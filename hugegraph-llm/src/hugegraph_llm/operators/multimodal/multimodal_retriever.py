"""
多模态联合检索器 — MultimodalGraphRAG Pipeline Stage 4

四通道 RRF 融合检索:
  1. 向量通道 (FAISS) — TextChunk + ImageDescription 文本嵌入
  2. BM25 通道 — 关键词全文匹配
  3. 图遍历通道 — HugeGraph k_neighbor 多跳关联
  4. 视觉通道 (★新增) — 图表类型过滤 + VLM 描述语义匹配 + 空间邻近

核心差异化：返回结果标注来源类型 (text/image/graph/mixed)，
让 LLM 在生成答案时知道哪些信息来自图片、哪些来自文本。
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Set
from enum import Enum

import requests

log = logging.getLogger(__name__)


# ========== 结果来源枚举 ==========


class SourceType(Enum):
    TEXT = "text"           # 来自 TextChunk
    IMAGE = "image"         # 来自 ImageDescription（通过图片）
    GRAPH = "graph"         # 来自图遍历发现的关联实体
    MIXED = "mixed"         # 多通道命中同一内容


@dataclass
class RetrievalResult:
    """单条检索结果"""
    id: str                  # 顶点 ID
    label: str               # 顶点标签 (TextChunk/ImageDescription/Image)
    score: float             # RRF 融合分数
    source_type: SourceType  # 来源类型
    channel_scores: Dict[str, float] = field(default_factory=dict)  # 各通道原始分
    properties: Dict[str, Any] = field(default_factory=dict)  # 顶点属性

    @property
    def is_from_image(self) -> bool:
        return self.source_type in (SourceType.IMAGE, SourceType.MIXED)

    @property
    def is_from_text(self) -> bool:
        return self.source_type in (SourceType.TEXT, SourceType.MIXED)


@dataclass
class MultiModalSearchResult:
    """多模态检索完整结果"""
    query: str
    query_type: str          # "text" / "image" / "mixed"
    results: List[RetrievalResult] = field(default_factory=list)

    # 各通道统计
    channel_stats: Dict[str, int] = field(default_factory=dict)

    # 来源分布
    source_distribution: Dict[str, int] = field(default_factory=dict)

    # 性能
    latency_ms: int = 0

    # 用于 LLM 上下文组装
    @property
    def text_context(self) -> str:
        """组装纯文本上下文（用于发送给 LLM）"""
        parts = []
        for r in self.results[:10]:  # top-10
            prefix = "[图]" if r.is_from_image else "[文]"
            content = (
                r.properties.get("content")
                or r.properties.get("caption")
                or r.properties.get("detailed_description", "")
            )
            parts.append(f"{prefix} {content[:300]}")
        return "\n\n".join(parts)

    @property
    def structured_context(self) -> Dict:
        """结构化上下文（包含来源标注，用于高级 Agent）"""
        visual_props = {"caption", "chart_type", "key_insights",
                        "object_labels", "detailed_description"}
        return {
            "query": self.query,
            "results": [
                {
                    "id": r.id,
                    "label": r.label,
                    "score": round(r.score, 4),
                    "source": r.source_type.value,
                    "content": (
                        r.properties.get("content") or r.properties.get("caption", "")
                    ),
                    **{
                        k: v for k, v in r.properties.items()
                        if k in visual_props and v
                    },
                }
                for r in self.results
            ],
            "source_distribution": self.source_distribution,
        }


class MultiModalRetriever:
    """
    多模态联合检索器 — 四通道 RRF 融合

    Usage:
        retriever = MultiModalRetriever(
            host="http://127.0.0.1:8080",
            graph="multimodal_demo",
            faiss_index_path="./faiss_multimodal.index",
            embedding_model="all-MiniLM-L6-v2",
        )

        # 文本查询
        result = retriever.search("2023年的营收趋势如何?")
        print(result.text_context)
        print(f"来源: {result.source_distribution}")

        # 图片查询（图片->VLM描述->匹配）
        result = retriever.search_by_image(image_base64_data)
    """

    RRF_K = 60  # RRF 常数（值越大，排名靠前的权重越高）

    # ---------- 视觉通道：图表类型关键词映射 ----------
    _CHART_TYPE_KEYWORDS: Dict[str, List[str]] = {
        "bar": ["柱状", "柱状图", "bar", "column", "对比", "比较"],
        "line": ["折线", "趋势", "line", "trend", "变化", "走势", "增减"],
        "pie": ["饼图", "占比", "pie", "proportion", "分布", "份额", "比例"],
        "table": ["表格", "数据", "table"],
        "scatter": ["散点", "相关性", "scatter", "correlation", "关系"],
        "flowchart": ["流程", "flow", "process", "步骤"],
        "architecture": ["架构", "系统", "architecture", "system", "拓扑"],
        "schema": ["模式", "schema", "模型", "结构图"],
        "map": ["地图", "map", "地理", "区域"],
        "screenshot": ["截图", "界面", "screenshot", "screen"],
        "photo": ["照片", "photo", "实物", "现场"],
        "other": [],
    }

    def __init__(
        self,
        host: str = "http://127.0.0.1:8080",
        graph: str = "hugegraph",
        auth: tuple = ("admin", "admin"),
        faiss_index_path: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        bm25_index: Optional[Any] = None,
        enable_vision_channel: bool = True,      # 是否启用视觉通道
        enable_graph_channel: bool = True,       # 是否启用图遍历通道
        top_k: int = 20,                         # 每通道返回数量
        final_top_k: int = 10,                   # 最终返回数量
        rrf_k: int = 60,                         # RRF常数
    ):
        self.host = host.rstrip("/")
        self.graph = graph
        self.auth = auth
        self.base_url = f"{host}/graphs/{graph}/graph"

        self.enable_vision = enable_vision_channel
        self.enable_graph = enable_graph_channel
        self.top_k = top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k

        # 向量通道初始化（懒加载）
        self._faiss_index = None
        self._embedding_model = None
        self._id_to_vertex: Dict[int, str] = {}   # FAISS index ID -> Vertex ID 映射
        self._vertex_to_id: Dict[str, int] = {}   # Vertex ID -> FAISS index ID 映射
        self.faiss_index_path = faiss_index_path
        self.embedding_dim = embedding_dim
        self.embedding_model_name = embedding_model

        # BM25 通道（懒加载）
        self._bm25 = bm25_index

    # ==================== 主入口 ====================

    def search(
        self,
        query: str,
        mode: str = "auto",       # auto / text_only / image_aware
        filters: Dict[str, Any] = None,  # 过滤条件 {chart_type: "bar", page_num: [1,2,3]}
    ) -> MultiModalSearchResult:
        """
        执行多模态联合搜索

        Args:
            query: 用户查询文本
            mode: 检索模式
                - auto: 自动判断是否涉及图像（含"图"/"图表"/"截图"等关键词时启用视觉通道）
                - text_only: 只用向量+BM25
                - image_aware: 强制启用所有4个通道
            filters: 属性过滤器

        Returns:
            MultiModalSearchResult
        """
        import time
        start = time.time()

        # 判断查询类型
        use_vision = self.enable_vision and (
            mode == "image_aware"
            or (mode == "auto" and self._is_visual_query(query))
        )

        all_results: Dict[str, List[Tuple[str, float]]] = {}

        # Channel 1: 向量检索
        vec_results = self._vector_search(query)
        if vec_results:
            all_results["vector"] = vec_results

        # Channel 2: BM25 检索
        bm25_results = self._bm25_search(query)
        if bm25_results:
            all_results["bm25"] = bm25_results

        # Channel 3: 视觉通道（核心差异化能力）
        if use_vision:
            vis_results = self._vision_search(query, filters)
            if vis_results:
                all_results["vision"] = vis_results

        # Channel 4: 图遍历通道
        if self.enable_graph:
            graph_results = self._graph_search(query)
            if graph_results:
                all_results["graph"] = graph_results

        # RRF 融合
        fused = self._rrf_fuse(all_results)

        # 获取顶点属性并构建最终结果
        results = self._build_results(fused, all_results)

        elapsed_ms = int((time.time() - start) * 1000)

        search_result = MultiModalSearchResult(
            query=query,
            query_type="mixed" if use_vision else "text",
            results=results[:self.final_top_k],
            channel_stats={ch: len(rlist) for ch, rlist in all_results.items()},
            source_distribution=self._count_sources(results),
            latency_ms=elapsed_ms,
        )

        log.info(
            "[MultiModal Search] query='%s' channels=%s "
            "results=%d distribution=%s latency=%dms",
            query[:50], list(all_results.keys()), len(results),
            search_result.source_distribution, elapsed_ms,
        )

        return search_result

    def search_by_image(
        self,
        image_base64: str,
        image_id: str = "user_image",
        page_context: str = "",
        text_query: str = "",
    ) -> MultiModalSearchResult:
        """
        以图片为输入的多模态搜索流程：
        1. 调用 VLM 对图片生成描述
        2. 用描述的 caption + key_insights 作为查询文本执行四通道检索

        Args:
            image_base64: 图片的 base64 编码数据
            image_id: 图片标识符
            page_context: 所在页面上下文
            text_query: 额外的文本查询（可选，会与VLM描述合并）

        Returns:
            MultiModalSearchResult
        """
        try:
            from .vlm_descriptor import VLMDescriptor
        except ImportError:
            log.error("search_by_image requires vlm_descriptor module")
            raise ImportError("请确保 vlm_descriptor.py 在同一目录下")

        descriptor = VLMDescriptor()
        desc = descriptor.describe(
            image_id=image_id,
            base64_data=image_base64,
            page_context=page_context,
        )

        # 用 VLM 描述的多个字段组合成查询
        combined_query_parts = [desc.caption, desc.detailed_description]
        combined_query_parts.extend(desc.key_insights)
        combined_query_parts.extend(desc.related_keywords)
        if text_query:
            combined_query_parts.insert(0, text_query)

        combined_query = " ".join(filter(None, combined_query_parts))

        log.info(
            "[SearchByImage] image_id=%s chart_type=%s "
            "combined_query='%s' (len=%d)",
            image_id, desc.chart_type, combined_query[:80],
            len(combined_query),
        )

        # 根据推断出的 chart_type 设置 filter，增强视觉通道效果
        filters = None
        if desc.chart_type and desc.chart_type != "other":
            filters = {"chart_type": desc.chart_type}

        return self.search(combined_query, mode="image_aware", filters=filters)

    # ==================== 四个通道实现 ====================

    def _vector_search(self, query: str) -> List[Tuple[str, float]]:
        """
        Channel 1: 向量语义检索

        搜索范围：
          - TextChunk.content 的嵌入
          - ImageDescription.detailed_description 的嵌入

        返回: [(vertex_id, similarity_score), ...]
        """
        try:
            import numpy as np
        except ImportError:
            log.warning("numpy not available, skipping vector search")
            return []

        embedding = self._get_embedding(query)
        if embedding is None:
            return []

        # 尝试使用 FAISS index
        faiss_index = self._load_faiss_index()
        if faiss_index is not None:
            return self._faiss_search(embedding)

        # 回退到 HugeGraph 属性扫描（无 FAISS 时的降级方案）
        return self._fallback_vector_scan()

    def _bm25_search(self, query: str) -> List[Tuple[str, float]]:
        """
        Channel 2: BM25 关键词检索

        搜索范围：
          - TextChunk.content
          - ImageDescription.caption
          - ImageDescription.related_keywords
          - ImageDescription.object_labels

        返回: [(vertex_id, bm25_score), ...]
        """
        # 如果已有 BM25 索引对象，直接使用
        if self._bm25 is not None:
            return self._bm25_index_search(query)

        # 回退到 HugeGraph contains 条件查询
        return self._fallback_bm25_contains(query)

    def _vision_search(
        self,
        query: str,
        filters: Dict[str, Any] = None,
    ) -> List[Tuple[str, float]]:
        """
        Channel 3: 视觉通道检索（核心差异化能力！）

        利用 VLM 生成的结构化描述做智能匹配，
        无需再调 VLM API，纯基于已存储的 ImageDescription 属性。

        匹配策略（按权重分配）：
          1. 图表类型匹配 (weight=0.40):
             query 含 "柱状图/趋势/bar chart" -> filter by chart_type
          2. 关键信息匹配 (weight=0.30):
             提取 query 中的数字/指标 -> match key_insights 字段
          3. 对象标签匹配 (weight=0.15):
             实体名匹配 -> match object_labels
          4. 关键词/标题匹配 (weight=0.15):
             related_keywords 与 query 的 token overlap

        Args:
            query: 用户查询文本
            filters: 额外属性过滤条件（如 {"chart_type": "bar"}）

        Returns:
            [(ImageDescription vertex_id, vision_score), ...]
        """
        # Step 1: 从 HugeGraph 获取所有 ImageDescription 顶点
        descriptions = self._fetch_all_image_descriptions(limit=2000)
        if not descriptions:
            log.warning("[Vision] No ImageDescription vertices found in graph")
            return []

        # Step 2: 推断用户期望的图表类型
        inferred_type = self._extract_chart_type_from_query(query)
        if filters and "chart_type" in filters:
            target_chart_type = filters["chart_type"]
        elif inferred_type:
            target_chart_type = inferred_type
        else:
            target_chart_type = None

        # Step 3: 预处理查询——提取 token 和数字特征
        query_tokens = set(self._tokenize(query))
        query_numbers = self._extract_numbers(query)

        scored: List[Tuple[str, float]] = []  # (vertex_id, score)

        for vid, props in descriptions:
            score = 0.0

            chart_type = str(props.get("chart_type", "")).strip().lower()
            caption = str(props.get("caption", ""))
            detailed_desc = str(props.get("detailed_description", ""))
            key_insights = props.get("key_insights", [])
            object_labels = props.get("object_labels", [])
            related_keywords = props.get("related_keywords", [])

            # --- a) 图表类型精确匹配 (+0 ~ 0.40) ---
            type_score = 0.0
            if target_chart_type and chart_type:
                if chart_type == target_chart_type:
                    type_score = 0.40
                else:
                    # 类型不完全匹配时，检查是否属于同一大类
                    type_score = self._chart_type_similarity(
                        target_chart_type, chart_type
                    ) * 0.20
            elif not target_chart_type:
                # 没有明确指定时，如果 query 中包含任意图表关键词且该顶点是图表，
                # 给一个基础分
                if chart_type and chart_type != "other":
                    if self._has_any_chart_keyword(query):
                        type_score = 0.05
            score += type_score

            # --- b) key_insights 匹配 (+0 ~ 0.30) ---
            insights_score = self._field_text_match_score(
                query_tokens, query_numbers, key_insights
            )
            score += insights_score * 0.30

            # --- c) object_labels 匹配 (+0 ~ 0.15) ---
            labels_score = self._field_text_match_score(
                query_tokens, query_numbers, object_labels
            )
            score += labels_score * 0.15

            # --- d) caption + related_keywords 匹配 (+0 ~ 0.15) ---
            caption_tokens = set(self._tokenize(caption))
            kw_texts = related_keywords if isinstance(related_keywords, list) else []
            kw_tokens = set()
            for kw in kw_texts:
                kw_tokens.update(self._tokenize(str(kw)))
            all_caption_kw_tokens = caption_tokens | kw_tokens

            if query_tokens and all_caption_kw_tokens:
                overlap = query_tokens & all_caption_kw_tokens
                keyword_jaccard = len(overlap) / len(query_tokens)
                score += min(keyword_jaccard, 1.0) * 0.15

            # 只保留有正向得分的候选结果
            if score > 1e-6:
                scored.append((vid, round(score, 6)))

        # Step 4: 按 score 降序排列，返回 top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        top_results = scored[:self.top_k]

        log.debug(
            "[Vision] query='%s' target_type=%s candidates=%d "
            "scored=%d returned=%d",
            query[:40], target_chart_type, len(descriptions),
            len(scored), len(top_results),
        )

        return top_results

    def _graph_search(self, query: str) -> List[Tuple[str, float]]:
        """
        Channel 4: 图遍历通道

        使用 HugeGraph k_neighbor API 从已知相关节点出发，
        发现隐式关联的内容。

        策略：
          1. 先用向量/BM25 找到 top-3 最相关的 seed nodes
          2. 以这些种子节点为起点，执行 2-hop k_neighbor
          3. 返回遍历到的节点（排除起始种子节点），用 depth 作为距离分数

        Returns:
            [(vertex_id, distance_score), ...] distance 越小越好
        """
        # 1) 快速获取 seed nodes（优先用已有的向量/BM25 结果缓存，
        #    否则做一个轻量级向量搜索）
        seeds = self._get_seed_nodes(query, max_seeds=3)
        if not seeds:
            log.debug("[Graph] No seed nodes found, skipping graph traversal")
            return []

        discovered: Dict[str, int] = {}  # vertex_id -> min_depth

        for seed_vid in seeds:
            neighbors = self._kneighbor_traverse(seed_vid, max_depth=2, limit=20)
            for neighbor_id, depth in neighbors.items():
                # 排除 seed 自身
                if neighbor_id == seed_vid:
                    continue
                old_depth = discovered.get(neighbor_id, 999)
                if depth < old_depth:
                    discovered[neighbor_id] = depth

        if not discovered:
            return []

        # 将 depth 转为分数：depth 越小分数越高，score = 1 / (depth + 1)
        scored = [
            (vid, round(1.0 / (depth + 1), 4))
            for vid, depth in sorted(discovered.items(), key=lambda x: x[1])
        ]
        # 按 score 降序排列
        scored.sort(key=lambda x: x[1], reverse=True)

        log.debug("[Graph] seeds=%d discovered=%d returned=%d",
                  len(seeds), len(discovered), min(len(scored), self.top_k))

        return scored[:self.top_k]

    # ==================== RRF 融合 ====================

    def _rrf_fuse(
        self,
        channel_results: Dict[str, List[Tuple[str, float]]],
    ) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion 多通道融合

        公式:
            RRF(id) = sum_{channel} 1 / (K + rank_i)
            其中 rank_i 是 item 在第 i 个通道中的排名（从1开始）

        K = 60（标准值）
        单通道第1名得分 ≈ 0.0164 (1/61)
        四通道均第1名得分 ≈ 0.0656 (4x 差距!)

        Args:
            channel_results: {channel_name: [(id, raw_score), ...]}

        Returns:
            {vertex_id: rrf_score}
        """
        k = self.rrf_k
        rrf_scores: Dict[str, float] = {}

        for channel_name, results in channel_results.items():
            # 按原始分数降序排序
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

            for rank, (vid, raw_score) in enumerate(sorted_results, start=1):
                rrf_score = 1.0 / (k + rank)
                rrf_scores[vid] = rrf_scores.get(vid, 0.0) + rrf_score

        return rrf_scores

    # ==================== 辅助方法：构建最终结果 ====================

    def _build_results(
        self,
        fused_scores: Dict[str, float],
        channel_raw: Dict[str, List[Tuple[str, float]]],
    ) -> List[RetrievalResult]:
        """将融合后的分数转换为带属性的 RetrievalResult 列表"""

        # 1) 按 rrf_score 降序排列
        sorted_ids = sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        )

        results: List[RetrievalResult] = []
        for vid, rrf_score in sorted_ids:
            # 2a) 获取顶点属性
            properties = self._get_vertex_properties(vid)
            label = properties.pop("label", "")

            # 2b) 判断 source_type（基于 ID 前缀和标签名）
            source_type = self._infer_source_type(vid, label, channel_raw)

            # 2c) 收集各通道原始分
            channel_scores = {}
            for ch_name, ch_list in channel_raw.items():
                for ch_vid, ch_score in ch_list:
                    if ch_vid == vid:
                        channel_scores[ch_name] = ch_score
                        break

            results.append(RetrievalResult(
                id=vid,
                label=label,
                score=round(rrf_score, 6),
                source_type=source_type,
                channel_scores=channel_scores,
                properties=properties,
            ))

        return results

    # ==================== 辅助方法：HugeGraph REST API ====================

    def _get_vertex_properties(self, vid: str) -> Dict[str, Any]:
        """从 HugeGraph 获取单个顶点的属性"""
        try:
            url = f"{self.base_url}/vertices/{vid}"
            r = requests.get(url, auth=self.auth, timeout=10)
            if r.status_code == 200:
                data = r.json()
                props = data.get("properties", {})
                # 将 label 放入属性中方便后续判断
                props["label"] = data.get("label", "")
                return props
            elif r.status_code == 404:
                log.debug("Vertex %s not found (404)", vid)
            else:
                log.debug("Get vertex %s failed: status=%d", vid, r.status_code)
        except requests.RequestException as e:
            log.debug("Failed to get vertex %s: %s", vid, e)
        return {"label": ""}

    def _fetch_all_image_descriptions(
        self, limit: int = 2000,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        从 HugeGraph 批量获取所有 ImageDescription 顶点及其属性

        Returns:
            [(vertex_id, properties_dict), ...]
        """
        results = []
        offset = 0
        batch_size = 500

        while True:
            params = {
                "label": "ImageDescription",
                "limit": batch_size,
            }
            if offset > 0:
                params[offset] = offset  # 分页偏移

            try:
                url = f"{self.base_url}/vertices"
                r = requests.get(url, auth=self.auth, params=params, timeout=15)
                if r.status_code != 200:
                    log.warning(
                        "[Vision] Fetch vertices failed: status=%d msg=%s",
                        r.status_code, r.text[:200],
                    )
                    break

                data = r.json()
                vertices = data.get("vertices", []) if isinstance(data, dict) else []
                if not vertices:
                    break

                for v in vertices:
                    vid = v.get("id")
                    if vid:
                        results.append((vid, v.get("properties", {})))

                if len(vertices) < batch_size:
                    break  # 已到最后

                offset += batch_size
                if limit and offset >= limit:
                    break

            except requests.RequestException as e:
                log.warning("[Vision] Fetch error: %s", e)
                break

        log.debug("[Vision] Fetched %d ImageDescription vertices", len(results))
        return results

    def _kneighbor_traverse(
        self,
        source_id: str,
        max_depth: int = 2,
        limit: int = 20,
    ) -> Dict[str, int]:
        """
        调用 HugeGraph k_neighbor API 进行图遍历

        Args:
            source_id: 起点 vertex ID
            max_depth: 最大跳数
            limit: 每层最大返回数

        Returns:
            {vertex_id: depth} 所有发现的节点及深度
        """
        try:
            url = f"{self.host}/graphs/{self.graph}/traversers/kneighbor"
            payload = {
                "source": {"id": source_id},
                "max_depth": max_depth,
                "limit": limit,
            }
            r = requests.post(url, json=payload, auth=self.auth, timeout=15)
            if r.status_code == 200:
                data = r.json()
                # k_neighbor 返回格式: {"vertices": [...]} 或类似
                # 兼容多种 HugeGraph 版本返回格式
                if isinstance(data, dict):
                    # 新版本: {"vertices": [{"id": "...", ...}]}
                    vertices = data.get("vertices", [])
                    if vertices:
                        # 尝试从响应中获取 depth 信息
                        result = {}
                        for v in vertices:
                            vid = v.get("id") or v
                            # k_neighbor 不直接返回 depth，
                            # 我们近似地认为都在 [1, max_depth] 内
                            # 这里保守估计 depth=2
                            if vid and vid != source_id:
                                result[vid] = 2
                        return result
                    # 另一种可能: 直接是 {id: distance}
                    if not vertices and "results" in data:
                        pass
        except requests.RequestException as e:
            log.debug("[Graph] k_neighbor failed for seed %s: %e", source_id, e)

        return {}

    # ==================== 辅助方法：Embedding / FAISS ====================

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """将文本转为嵌入向量（懒加载模型）"""
        if self._embedding_model is None:
            self._init_embedding_model()
        if self._embedding_model is None:
            return None

        try:
            embedding = self._embedding_model.encode(text)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            log.warning("Encoding failed: %s", e)
            return None

    def _init_embedding_model(self):
        """懒加载 sentence-transformer 嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            log.info(
                "Loaded embedding model: %s (dim=%d)",
                self.embedding_model_name, self.embedding_dim,
            )
        except ImportError:
            log.warning(
                "sentence_transformers not installed; "
                "vector search will use fallback"
            )
        except Exception as e:
            log.warning("Failed to load embedding model: %s", e)

    def _load_faiss_index(self):
        """懒加载 FAISS 索引"""
        if self._faiss_index is not None:
            return self._faiss_index
        if not self.faiss_index_path:
            return None

        try:
            import faiss
            self._faiss_index = faiss.read_index(self.faiss_index_path)
            log.info("Loaded FAISS index from %s (%d vectors)",
                     self.faiss_index_path, self._faiss_index.ntotal)
            return self._faiss_index
        except ImportError:
            log.warning("faiss not installed")
            self._faiss_index = None
        except Exception as e:
            log.warning("Failed to load FAISS index: %s", e)
            self._faiss_index = None
        return None

    def _faiss_search(
        self, query_embedding: List[float],
    ) -> List[Tuple[str, float]]:
        """使用 FAISS 索引进行向量相似度搜索"""
        import numpy as np
        faiss_index = self._faiss_index
        if faiss_index is None:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        k = min(self.top_k, faiss_index.ntotal)
        scores, indices = faiss_index.search(query_vec, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            vid = self._id_to_vertex.get(int(idx))
            if vid:
                results.append((vid, float(score)))

        return results

    def _fallback_vector_scan(self) -> List[Tuple[str, float]]:
        """
        FAISS 不可用时的降级方案：通过 HugeGraph 属性扫描获取文本内容。
        此时不计算真正的向量相似度，仅返回空列表让其他通道补充。
        """
        log.info("No FAISS index available, vector channel degraded")
        return []

    def _build_faiss_mapping(self, id_map: Dict[int, str]):
        """设置 FAISS index ID <-> Vertex ID 的映射关系"""
        self._id_to_vertex = {int(k): v for k, v in id_map.items()}
        self._vertex_to_id = {v: int(k) for k, v in id_map.items()}
        log.info("Built FAISS mapping: %d entries", len(self._id_to_vertex))

    # ==================== 辅助方法：BM25 ====================

    def _bm25_index_search(self, query: str) -> List[Tuple[str, float]]:
        """使用预构建的 BM25 索引搜索"""
        try:
            scores = self._bm25.get_scores(query)
            # 假设 _bm25 对象支持 get_scores + doc_id 映射
            # 具体实现取决于 BM25 库 (rank_bm25 / whoosh / jieba)
            top_indices = np.argsort(scores)[::-1][:self.top_k]
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    doc_id = getattr(self._bm25, f"doc_{idx}", f"doc_{idx}")
                    results.append((str(doc_id), float(scores[idx])))
            return results
        except Exception as e:
            log.warning("BM25 index search failed: %s", e)
            return []

    def _fallback_bm25_contains(
        self, query: str,
    ) -> List[Tuple[str, float]]:
        """
        BM25 索引不可用时回退到 HugeGraph contains 查询。

        使用 query 中的关键词对 TextChunk 和 ImageDescription 做
        contains 条件扫描，手动计算简化的 TF 打分。
        """
        tokens = self._tokenize(query)
        if not tokens:
            return []

        # 取前3个最有区分度的词作为搜索关键词
        keywords = [t for t in tokens if len(t) > 1][:3]
        if not keywords:
            keywords = list(tokens)[:1]
        if not keywords:
            return []

        all_candidates = []

        # 搜索 TextChunk
        for label in ("TextChunk", "ImageDescription"):
            for kw in keywords:
                try:
                    params = {
                        "label": label,
                        f"contains(content)": kw,
                        "limit": 50,
                    }
                    r = requests.get(
                        f"{self.base_url}/vertices",
                        auth=self.auth, params=params, timeout=10,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        vertices = data.get("vertices", [])
                        for v in vertices:
                            vid = v.get("id")
                            props = v.get("properties", {})
                            content = (
                                props.get("content")
                                or props.get("caption")
                                or props.get("detailed_description", "")
                                or ""
                            )
                            # 简化的 TF 得分：关键词出现次数 / 文本长度
                            tf = sum(content.count(k) for k in keywords)
                            if tf > 0:
                                norm_score = tf / max(len(content), 1) * 1000
                                all_candidates.append((vid, norm_score))
                except requests.RequestException:
                    continue

        # 按 score 降序去重
        seen = set()
        deduped = []
        for vid, score in sorted(all_candidates, key=lambda x: x[1], reverse=True):
            if vid not in seen:
                seen.add(vid)
                deduped.append((vid, score))

        return deduped[:self.top_k]

    # ==================== 辅助方法：Seed Nodes ====================

    def _get_seed_nodes(self, query: str, max_seeds: int = 3) -> List[str]:
        """为图遍历获取 seed nodes（复用轻量级向量或 BM25 结果的 top-K）"""
        seeds = []

        # 优先用向量搜索找 seed
        vec_res = self._vector_search(query)
        for vid, _ in vec_res[:max_seeds]:
            seeds.append(vid)

        # 如果不够，补 BM25
        if len(seeds) < max_seeds:
            bm25_res = self._bm25_search(query)
            for vid, _ in bm25_res:
                if vid not in seeds:
                    seeds.append(vid)
                    if len(seeds) >= max_seeds:
                        break

        return seeds

    # ==================== 辅助方法：Source Type 推断 ====================

    @staticmethod
    def _infer_source_type(
        vid: str,
        label: str,
        channel_raw: Dict[str, List[Tuple[str, float]]],
    ) -> SourceType:
        """
        根据 vertex ID、标签名、以及哪些通道命中了它来推断来源类型

        Rules:
          - ID 前缀 "desc_" 或 label 为 ImageDescription/Image => IMAGE
          - ID 前缀 "txt_" 或 label 为 TextChunk           => TEXT
          - 仅在 graph 通道命中                             => GRAPH
          - 多个通道同时命中                                 => MIXED
        """
        hit_channels = set()
        for ch_name, ch_list in channel_raw.items():
            for ch_vid, _ in ch_list:
                if ch_vid == vid:
                    hit_channels.add(ch_name)
                    break

        # 基于 ID/Label 判断基础类别
        if vid.startswith("desc_") or label in ("ImageDescription", "Image"):
            base_type = SourceType.IMAGE
        elif vid.startswith("txt_") or label in ("TextChunk",):
            base_type = SourceType.TEXT
        else:
            base_type = SourceType.GRAPH

        # 如果被多个通道命中，升级为 MIXED
        if len(hit_channels) >= 2:
            # 但如果所有命中的通道都是同一类别的，保持原类别
            has_image_like = base_type == SourceType.IMAGE
            has_text_like = base_type == SourceType.TEXT
            has_graph_only = base_type == SourceType.GRAPH and hit_channels == {"graph"}

            if has_graph_only:
                return SourceType.GRAPH

            if base_type == SourceType.GRAPH and len(hit_channels) > 1:
                # 图遍历发现的结果又被其他通道确认
                return SourceType.MIXED

        return base_type

    # ==================== 辅助方法：统计 ====================

    def _count_sources(
        self, results: List[RetrievalResult],
    ) -> Dict[str, int]:
        """统计来源分布"""
        dist: Dict[str, int] = {}
        for st in SourceType:
            dist[st.value] = 0
        for r in results:
            dist[r.source_type.value] += 1
        return {k: v for k, v in dist.items() if v > 0}

    @staticmethod
    def _is_visual_query(query: str) -> bool:
        """判断查询是否涉及视觉/图像内容"""
        visual_keywords = [
            '图', '图表', '截图', '照片', '图片', '示意图',
            'graph', 'chart', 'figure', 'diagram', 'plot',
            'trend', '趋势', '柱状', '饼图', '折线',
            'show me', 'look at', 'see the',
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in visual_keywords)

    def _has_any_chart_keyword(self, query: str) -> bool:
        """检查 query 中是否包含任何图表相关的关键词"""
        for keywords in self._CHART_TYPE_KEYWORDS.values():
            if any(kw in query.lower() for kw in keywords):
                return True
        return False

    def _extract_chart_type_from_query(self, query: str) -> Optional[str]:
        """从查询中推断期望的图表类型"""
        query_lower = query.lower()
        for ctype, keywords in self._CHART_TYPE_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return ctype
        return None

    @staticmethod
    def _chart_type_similarity(type_a: str, type_b: str) -> float:
        """
        计算两种图表类型的相似度 [0, 1]，用于部分匹配打分。

        相似分组:
          - 数据图表组: bar, line, pie, scatter, table (互似度 0.5)
          - 结构图组: flowchart, architecture, schema (互似度 0.5)
          - 其他: photo, screenshot, map, other (互似度低)
        """
        if type_a == type_b:
            return 1.0
        data_charts = {"bar", "line", "pie", "scatter", "table"}
        structure_charts = {"flowchart", "architecture", "schema"}
        if type_a in data_charts and type_b in data_charts:
            return 0.5
        if type_a in structure_charts and type_b in structure_charts:
            return 0.5
        return 0.1  # 不同大组的极低相似度

    # ==================== 辅助方法：文本处理工具 ====================

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        简单分词：中文按字符/常用词切分，英文按空格切分。
        生产环境建议替换为 jieba/spkac。
        """
        # 英文单词
        english_tokens = re.findall(r'[a-zA-Z]{2,}', text.lower())
        # 中文：保留长度>=2 的连续中文片段，和单字数字
        chinese_tokens = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        # 数字
        number_tokens = re.findall(
            r'[\d]+(?:\.[\d]+)?%?|[\d]{4}', text
        )
        return english_tokens + chinese_tokens + number_tokens

    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        """提取文本中的数字（含百分号、货币等）"""
        return re.findall(
            r'(?:[\d,]+(?:\.[\d+])?(?:%)?)'
            r'|(?:[\u00a5$\u20ac][\d,]+(?:\.[\d+])?)',
            text,
        )

    @staticmethod
    def _field_text_match_score(
        query_tokens: Set[str],
        query_numbers: List[str],
        field_values: Any,
    ) -> float:
        """
        计算查询与某个字段列表的文本匹配得分 [0, 1]。

        统计 query tokens/numbers 在字段值中的覆盖程度。
        """
        if not field_values:
            return 0.0
        if not isinstance(field_values, list):
            field_values = [str(field_values)]

        # 将字段值展平为 token 集合
        field_token_set: Set[str] = set()
        for val in field_values:
            val_str = str(val)
            field_token_set.update(MultiModalRetriever._tokenize(val_str))

        if not query_tokens and not query_numbers:
            return 0.0

        # 数字完全匹配权重更高
        matched_numbers = 0
        for qn in query_numbers:
            for fv in field_values:
                if qn in str(fv):
                    matched_numbers += 1
                    break
        number_score = (
            matched_numbers / max(len(query_numbers), 1)
            if query_numbers else 0.0
        )

        # Token Jaccard
        if query_tokens and field_token_set:
            overlap = query_tokens & field_token_set
            token_jaccard = len(overlap) / len(query_tokens)
        else:
            token_jaccard = 0.0

        # 综合：数字匹配占 60%，Token 匹配占 40%
        return number_score * 0.60 + token_jaccard * 0.40


# ========== 便捷函数 ==========


def multimodal_search(
    query: str,
    host: str = "http://127.0.0.1:8080",
    graph: str = "hugegraph",
    **kwargs,
) -> MultiModalSearchResult:
    """一键多模态搜索"""
    retriever = MultiModalRetriever(host=host, graph=graph, **kwargs)
    return retriever.search(query)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="多模态联合检索器")
    parser.add_argument("query", help="搜索查询")
    parser.add_argument("--host", default="http://127.0.0.1:8080")
    parser.add_argument("--graph", default="multimodal_poc")
    parser.add_argument(
        "--mode", default="auto", choices=["auto", "text_only", "image_aware"]
    )
    parser.add_argument("--top-k", type=int, default=10, dest="top_k")
    parser.add_argument("--json", action="store_true", dest="output_json",
                        help="输出JSON格式")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    retriever = MultiModalRetriever(
        host=args.host,
        graph=args.graph,
        final_top_k=args.top_k,
    )

    result = retriever.search(args.query, mode=args.mode)

    if args.output_json:
        print(json.dumps(result.structured_context, indent=2, ensure_ascii=False))
    else:
        print(f"\n{'='*50}")
        print(f"  查询: {args.query}")
        print(f"{'='*50}")
        print(f"耗时: {result.latency_ms}ms | 来源分布: {result.source_distribution}")
        print(f"各通道结果数: {result.channel_stats}")
        print(f"\n--- Top-{len(result.results)} Results ---\n")

        for i, r in enumerate(result.results, 1):
            prefix = "[IMG]" if r.is_from_image else "[TXT]"
            content = (
                r.properties.get('content')
                or r.properties.get('caption')
                or r.properties.get('detailed_description', '')
            )[:200]

            ch_info = ""
            if r.channel_scores:
                parts = [f"{ch}:{s:.3f}" for ch, s in r.channel_scores.items()]
                ch_info = f" channels=[{', '.join(parts)}]"

            print(
                f"{i}. {prefix} [{r.source_type.value}] "
                f"(rrf={r.score:.4f}){ch_info}"
            )
            print(f"   {content}\n")

        # 同时输出结构化上下文示例
        print("--- Structured Context Preview ---")
        ctx = result.text_context
        print(ctx[:800] + ("..." if len(ctx) > 800 else ""))
