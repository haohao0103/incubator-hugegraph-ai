"""
多模态感知检索通道 — Multimodal-Aware Retrieval Channel (P1)

核心问题：MultimodalEntityInjector 注入了 drawing/table/equation 实体到图谱，
但现有 RAG 检索管线 (KeywordExtract -> GraphQuery -> VectorQuery -> MergeRerank -> AnswerSynthesize)
完全不知道这些多模态实体的存在，导致它们无法被检索到。

解决方案（借鉴 LightRAG 设计）：
  - 多模态实体搭乘与常规实体相同的检索管线（语义搜索 + 图遍历）
  - 关键差异化：TYPE LABELING — 用 [图]/[表]/[公式] 标记结果，让 LLM 知道信息来源
  - association edges 提供跨模态上下文（文本实体 <-> 图像实体的关联）

管线集成位置：插入在 GraphQueryNode 之后、MergeRerankNode 之前，
  检索结果写入 context["multimodal_context"]，供 MergeRerankNode 和 AnswerSynthesizeNode 使用。

与 MultiModalRetriever 的关系：
  - MultiModalRetriever 是面向 HugeGraph REST API 的独立检索器（四通道 RRF 融合）
  - 本通道是面向 HG-AI RAG pipeline 的 operator，遵循 run(context) -> context 协议
  - 本通道可复用 MultiModalRetriever 的视觉通道逻辑，但专注于将多模态实体
    连接到已有的向量索引和图遍历管线

LightRAG 对应逻辑：
  - Entity VDB: content = f"{entity_name}\\n{description}" — 描述文本被嵌入
  - kg_query() _get_node_data(): 搜索 Entity VDB → 多模态实体自然参与
  - _find_related_text_unit_from_entities(): entity.source_id → chunks
  - _find_most_related_edges_from_entities(): "associated_with" edges
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import logging
log = logging.getLogger(__name__)

# Lazy-load HG-AI framework dependencies to avoid pulling full dependency chain
# when only the retrieval channel logic is needed (e.g., in unit tests).
def _get_vector_store_base():
    from hugegraph_llm.indices.vector_index.base import VectorStoreBase
    return VectorStoreBase

def _get_base_embedding():
    from hugegraph_llm.models.embeddings.base import BaseEmbedding
    return BaseEmbedding

# 多模态类型标签映射（中英文）
ENTITY_TYPE_LABELS: Dict[str, str] = {
    "drawing": "[图]",
    "table": "[表]",
    "equation": "[公式]",
    "Photo": "[图]",
    "Illustration": "[图]",
    "Screenshot": "[截图]",
    "Icon": "[图标]",
    "Chart": "[图表]",
    "Infographic": "[信息图]",
    "Flowchart": "[流程图]",
    "Chat Log": "[对话]",
    "Wireframe": "[线框图]",
    "Texture": "[纹理]",
    "Other": "[其他]",
}

# 多模态实体类型过滤器（与 MultimodalEntityInjector 一致）
MULTIMODAL_ENTITY_TYPES: Tuple[str, ...] = (
    "drawing", "table", "equation",
)

# HG-AI 图谱顶点中用于标识多模态实体的属性字段
_ENTITY_TYPE_PROP = "entity_type"
_SOURCE_ID_PROP = "source_id"
_DESCRIPTION_PROP = "description"
_FILE_PATH_PROP = "file_path"
_ENTITY_NAME_PROP = "entity_name"


# ========== 数据类 ==========


@dataclass
class MultimodalEntityResult:
    """检索到的多模态实体结果"""
    entity_name: str            # sidecar_id / image_id
    entity_type: str            # "drawing" / "table" / "equation" / LightRAG IMAGE_TYPE_ENUM
    description: str            # VLM 分析结果
    score: float                # 向量搜索相似度得分
    source_id: str              # 来源 chunk 引用
    file_path: str              # 来源文件路径

    @property
    def label(self) -> str:
        """中文类型标签"""
        return ENTITY_TYPE_LABELS.get(self.entity_type, "[其他]")

    @property
    def content_for_embedding(self) -> str:
        """用于嵌入的文本（与 LightRAG Entity VDB 格式一致）"""
        return f"{self.entity_name}\n{self.description}"


@dataclass
class MultimodalChunkResult:
    """通过 source_id 检索到的多模态原始 chunk"""
    chunk_id: str               # chunk key / block_id
    content: str                # VLM 渲染的 chunk 内容（含 [Image Name]... / [Table Name]... 标签）
    source_type: str            # "drawing" / "table" / "equation"
    score: float                # 间接得分（从关联实体的得分继承）

    @property
    def label(self) -> str:
        return ENTITY_TYPE_LABELS.get(self.source_type, "[其他]")


@dataclass
class MultimodalEdgeResult:
    """多模态实体与文本实体之间的关联边"""
    src_id: str                 # 多模态实体名
    tgt_id: str                 # 文本实体名
    description: str            # 关联描述
    edge_type: str = "associated_with"


@dataclass
class MultimodalRetrievalContext:
    """多模态检索完整上下文 — 写入 context["multimodal_context"]"""
    entities: List[MultimodalEntityResult] = field(default_factory=list)
    chunks: List[MultimodalChunkResult] = field(default_factory=list)
    edges: List[MultimodalEdgeResult] = field(default_factory=list)

    # LLM 可直接使用的格式化文本
    text_for_llm: str = ""

    # 检索统计
    latency_ms: int = 0
    entity_count: int = 0
    chunk_count: int = 0
    edge_count: int = 0


@dataclass
class MultimodalRetrievalConfig:
    """多模态检索通道配置"""
    enable_multimodal_channel: bool = True
    multimodal_top_k: int = 5
    enable_association_edges: bool = True
    enable_chunk_retrieval: bool = True
    entity_types_filter: Tuple[str, ...] = MULTIMODAL_ENTITY_TYPES
    # 图遍历深度（查找 associated_with 边时）
    graph_traversal_depth: int = 1
    # 向量搜索距离阈值
    vector_dis_threshold: float = 0.9


# ========== 核心逻辑 ==========


class MultimodalRetrievalChannel:
    """
    多模态感知检索通道 — 将注入的多模态实体连接到 RAG 检索管线。

    核心设计（与 LightRAG 一致）：
      多模态实体搭乘与常规实体相同的语义检索管线，
      差异化在于 TYPE LABELING — 标注 [图]/[表]/[公式] 让 LLM 知道信息来源。

    管线集成：
      在 GraphQueryNode 之后、MergeRerankNode 之前插入，
      检索结果写入 context["multimodal_context"]。

    Context 输入:
      - context["query"] — 用户查询
      - context["graph_client"] — HugeGraph REST 客户端 (PyHugeClient)
      - context["vector_index"] — 向量索引 (VectorStoreBase 子类实例或类)
      - context["embedding_model"] — 嵌入模型 (BaseEmbedding)
      - context["top_k"] — 返回结果数量 (可选, 默认 config.multimodal_top_k)
      - context["multimodal_entities"] — MultimodalEntityInjector 注入的实体 (可选)
      - context["vertices"] — 图谱顶点列表 (可选, 包含已注入的多模态实体)

    Context 输出 (新增):
      - context["multimodal_context"] — MultimodalRetrievalContext
      - context["multimodal_context_text"] — text_for_llm 的快捷引用
    """

    def __init__(self, config: Optional[MultimodalRetrievalConfig] = None):
        self.config = config or MultimodalRetrievalConfig()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行多模态检索，将结果写入 context["multimodal_context"]"""
        if not self.config.enable_multimodal_channel:
            log.debug("[MultimodalChannel] Disabled by config")
            return context

        query = context.get("query", "")
        if not query:
            log.debug("[MultimodalChannel] No query in context, skipping")
            return context

        start = time.time()

        # 1. 从 context 获取关键依赖
        embedding_model = context.get("embedding_model")
        if embedding_model is None:
            log.warning("[MultimodalChannel] No embedding_model in context, skipping")
            return context

        top_k = context.get("top_k", self.config.multimodal_top_k)

        # 2. 搜索多模态实体（向量语义搜索）
        entities = self._search_multimodal_entities(
            query=query,
            embedding_model=embedding_model,
            context=context,
            top_k=top_k,
        )

        # 3. 获取关联边（图遍历）
        edges: List[MultimodalEdgeResult] = []
        if self.config.enable_association_edges and entities:
            entity_names = [e.entity_name for e in entities]
            edges = self._follow_association_edges(
                entity_names=entity_names,
                context=context,
            )

        # 4. 检索多模态原始 chunks
        chunks: List[MultimodalChunkResult] = []
        if self.config.enable_chunk_retrieval and entities:
            entity_names = [e.entity_name for e in entities]
            entity_scores = {e.entity_name: e.score for e in entities}
            chunks = self._retrieve_multimodal_chunks(
                entity_names=entity_names,
                entity_scores=entity_scores,
                context=context,
            )

        # 5. 格式化为 LLM 上下文
        text_for_llm = self._format_context_for_llm(entities, chunks, edges)

        elapsed_ms = int((time.time() - start) * 1000)

        # 6. 构建并写入 context
        mm_context = MultimodalRetrievalContext(
            entities=entities,
            chunks=chunks,
            edges=edges,
            text_for_llm=text_for_llm,
            latency_ms=elapsed_ms,
            entity_count=len(entities),
            chunk_count=len(chunks),
            edge_count=len(edges),
        )

        context["multimodal_context"] = mm_context
        context["multimodal_context_text"] = text_for_llm

        log.info(
            "[MultimodalChannel] query='%s' entities=%d chunks=%d "
            "edges=%d latency=%dms",
            query[:50], len(entities), len(chunks),
            len(edges), elapsed_ms,
        )

        return context

    # ==================== 核心检索方法 ====================

    def _search_multimodal_entities(
        self,
        query: str,
        embedding_model: Any,
        context: Dict[str, Any],
        top_k: int,
    ) -> List[MultimodalEntityResult]:
        """
        向量语义搜索多模态实体。

        策略（与 LightRAG Entity VDB 一致）：
          1. 对 query 嵌入向量
          2. 在向量索引中搜索，过滤 entity_type ∈ {drawing, table, equation}
          3. 返回带得分的 MultimodalEntityResult 列表

        降级方案：
          - 若无向量索引，从 context["vertices"] 中做多模态实体属性匹配
        """
        # 优先使用向量索引搜索
        vector_index = context.get("vector_index")
        if vector_index is not None:
            return self._vector_search_entities(
                query, embedding_model, vector_index, top_k,
            )

        # 降级：从 context["vertices"] 属性扫描
        return self._fallback_entity_scan(query, context, top_k)

    def _vector_search_entities(
        self,
        query: str,
        embedding_model: Any,
        vector_index: Any,
        top_k: int,
    ) -> List[MultimodalEntityResult]:
        """通过向量索引搜索多模态实体"""
        try:
            # 获取 query 嵌入
            query_embedding = embedding_model.get_texts_embeddings([query])[0]
        except Exception as e:
            log.warning("[MultimodalChannel] Failed to embed query: %s", e)
            return []

        try:
            # 搜索向量索引（获取所有相关结果，然后过滤多模态类型）
            # VectorStoreBase.search 返回 List[Any] — 属性列表
            results = vector_index.search(
                query_embedding,
                top_k * 3,  # 取 3x 数量，过滤后保留 top_k
                dis_threshold=self.config.vector_dis_threshold,
            )
        except Exception as e:
            log.warning("[MultimodalChannel] Vector search failed: %s", e)
            return []

        if not results:
            return []

        # 过滤多模态实体
        entity_results: List[MultimodalEntityResult] = []
        types_filter = set(self.config.entity_types_filter)

        for item in results:
            # 向量索引结果的格式可能是 dict 或字符串
            if isinstance(item, dict):
                entity_type = str(item.get(_ENTITY_TYPE_PROP, ""))
                if entity_type not in types_filter:
                    continue
                entity_name = str(item.get(_ENTITY_NAME_PROP, item.get("id", "")))
                description = str(item.get(_DESCRIPTION_PROP, ""))
                source_id = str(item.get(_SOURCE_ID_PROP, ""))
                file_path = str(item.get(_FILE_PATH_PROP, ""))
                # 向量索引通常不直接返回 score，需要从搜索结果中估算
                # 搜索结果已经按相似度排序，用 rank 递减分
                score = 1.0 / (len(entity_results) + 1)
                entity_results.append(MultimodalEntityResult(
                    entity_name=entity_name,
                    entity_type=entity_type,
                    description=description,
                    score=score,
                    source_id=source_id,
                    file_path=file_path,
                ))
            elif isinstance(item, str):
                # 某些向量索引（如 FAISS）可能返回字符串格式
                # 尝试从字符串中解析多模态实体信息
                # 格式可能为: "entity_name{entity_type: drawing, ...}"
                parsed = self._parse_vertex_string(item)
                if parsed and parsed.get(_ENTITY_TYPE_PROP) in types_filter:
                    entity_results.append(MultimodalEntityResult(
                        entity_name=parsed.get("id", ""),
                        entity_type=parsed.get(_ENTITY_TYPE_PROP, ""),
                        description=parsed.get(_DESCRIPTION_PROP, ""),
                        score=1.0 / (len(entity_results) + 1),
                        source_id=parsed.get(_SOURCE_ID_PROP, ""),
                        file_path=parsed.get(_FILE_PATH_PROP, ""),
                    ))

        return entity_results[:top_k]

    def _fallback_entity_scan(
        self,
        query: str,
        context: Dict[str, Any],
        top_k: int,
    ) -> List[MultimodalEntityResult]:
        """
        无向量索引时的降级方案：从 context["vertices"] 扫描多模态实体。

        使用简单的关键词匹配：如果 query 中包含实体 description 的关键词，
        则认为该实体相关。

        这是 LightRAG Entity VDB 不可用时的降级策略。
        """
        vertices = context.get("vertices", [])
        if not vertices:
            # 也可以从 multimodal_entities 获取
            mm_entities_spec = context.get("multimodal_entities", [])
            if mm_entities_spec:
                vertices = [
                    {
                        _ENTITY_NAME_PROP: spec.entity_name,
                        _ENTITY_TYPE_PROP: spec.entity_type,
                        _DESCRIPTION_PROP: spec.description,
                        _SOURCE_ID_PROP: spec.source_id,
                        _FILE_PATH_PROP: spec.file_path,
                    }
                    for spec in mm_entities_spec
                ]

        if not vertices:
            return []

        types_filter = set(self.config.entity_types_filter)
        query_tokens = self._tokenize(query)

        scored_entities: List[Tuple[float, MultimodalEntityResult]] = []

        for v in vertices:
            if isinstance(v, dict):
                entity_type = str(v.get(_ENTITY_TYPE_PROP, ""))
                if entity_type not in types_filter:
                    continue

                entity_name = str(v.get(_ENTITY_NAME_PROP, ""))
                description = str(v.get(_DESCRIPTION_PROP, ""))
                source_id = str(v.get(_SOURCE_ID_PROP, ""))
                file_path = str(v.get(_FILE_PATH_PROP, ""))

                # 关键词重叠得分
                desc_tokens = self._tokenize(description)
                if query_tokens and desc_tokens:
                    overlap = query_tokens & desc_tokens
                    score = len(overlap) / len(query_tokens)
                else:
                    score = 0.0

                if score > 0:
                    scored_entities.append((
                        score,
                        MultimodalEntityResult(
                            entity_name=entity_name,
                            entity_type=entity_type,
                            description=description,
                            score=score,
                            source_id=source_id,
                            file_path=file_path,
                        ),
                    ))

        # 按得分降序排列
        scored_entities.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored_entities[:top_k]]

    def _follow_association_edges(
        self,
        entity_names: List[str],
        context: Dict[str, Any],
    ) -> List[MultimodalEdgeResult]:
        """
        图遍历获取关联边 — 查找多模态实体与文本实体的 "associated_with" 边。

        借鉴 LightRAG _find_most_related_edges_from_entities()，
        但使用 HugeGraph 的方式实现。

        策略：
          1. 从 context["edges"] 中查找包含多模态实体名的关联边
          2. 若有 graph_client，也可通过 HugeGraph REST API 查询边
        """
        edges_result: List[MultimodalEdgeResult] = []

        # 优先从 context 中已有的边数据查找（MultimodalEntityInjector 写入的）
        context_edges = context.get("edges", [])
        context_mm_associations = context.get("multimodal_associations", [])

        # 1. 从 multimodal_associations 中查找（精确格式）
        for assoc in context_mm_associations:
            if not isinstance(assoc, dict):
                # MultimodalAssociationSpec 对象
                src_id = getattr(assoc, "src_id", "")
                tgt_id = getattr(assoc, "tgt_id", "")
                desc = getattr(assoc, "description", "")
            else:
                src_id = str(assoc.get("src_id", ""))
                tgt_id = str(assoc.get("tgt_id", ""))
                desc = str(assoc.get("description", ""))

            if src_id in entity_names or tgt_id in entity_names:
                edges_result.append(MultimodalEdgeResult(
                    src_id=src_id,
                    tgt_id=tgt_id,
                    description=desc,
                    edge_type="associated_with",
                ))

        # 2. 从通用 edges 中补充查找
        for edge in context_edges:
            if not isinstance(edge, dict):
                continue
            src_id = str(edge.get("src_id", ""))
            tgt_id = str(edge.get("tgt_id", ""))
            desc = str(edge.get("description", ""))
            keywords = str(edge.get("keywords", ""))

            # 检查是否与多模态实体相关
            if src_id in entity_names or tgt_id in entity_names:
                # 避免重复
                already_found = any(
                    e.src_id == src_id and e.tgt_id == tgt_id
                    for e in edges_result
                )
                if not already_found:
                    edge_type = "associated_with" if "associated" in keywords.lower() else str(edge.get("edge_type", "related"))
                    edges_result.append(MultimodalEdgeResult(
                        src_id=src_id,
                        tgt_id=tgt_id,
                        description=desc,
                        edge_type=edge_type,
                    ))

        # 3. 若有 graph_client，通过 HugeGraph REST API 查询（补充更多边）
        graph_client = context.get("graph_client")
        if graph_client is not None and len(edges_result) < len(entity_names) * 2:
            self._query_association_edges_from_graph(
                entity_names, graph_client, edges_result,
            )

        return edges_result

    def _query_association_edges_from_graph(
        self,
        entity_names: List[str],
        graph_client: Any,
        edges_result: List[MultimodalEdgeResult],
    ) -> None:
        """通过 HugeGraph REST API 查询 associated_with 边"""
        existing_pairs: Set[Tuple[str, str]] = {
            (e.src_id, e.tgt_id) for e in edges_result
        }

        for entity_name in entity_names:
            try:
                # 尝试获取与该实体相关的边
                edges, _ = graph_client.graph().getEdgeByPage(
                    vertex_id=entity_name,
                    direction="BOTH",
                    limit=20,
                )
                for edge in edges:
                    if edge.label != "associated_with":
                        continue

                    # 确定方向
                    src = edge.outV if edge.outV == entity_name else edge.inV
                    tgt = edge.inV if edge.outV == entity_name else edge.outV

                    pair = (src, tgt)
                    reverse_pair = (tgt, src)
                    if pair in existing_pairs or reverse_pair in existing_pairs:
                        continue

                    desc = ""
                    if hasattr(edge, "properties") and edge.properties:
                        desc = str(edge.properties.get("description", ""))

                    edges_result.append(MultimodalEdgeResult(
                        src_id=src,
                        tgt_id=tgt,
                        description=desc,
                        edge_type="associated_with",
                    ))
                    existing_pairs.add(pair)
            except Exception as e:
                log.debug(
                    "[MultimodalChannel] Graph query for '%s' failed: %s",
                    entity_name, e,
                )

    def _retrieve_multimodal_chunks(
        self,
        entity_names: List[str],
        entity_scores: Dict[str, float],
        context: Dict[str, Any],
    ) -> List[MultimodalChunkResult]:
        """
        通过 source_id 检索多模态原始 chunk。

        借鉴 LightRAG _find_related_text_unit_from_entities()：
          entity.source_id → 获取 chunk 内容

        策略：
          1. 从 context["vertices"] 中查找多模态实体的 source_id
          2. 用 source_id 在向量索引或 HugeGraph 中获取 chunk 内容
        """
        chunks: List[MultimodalChunkResult] = []
        seen_chunk_ids: Set[str] = set()

        # 从 context["vertices"] 获取 source_id 映射
        vertices = context.get("vertices", [])
        source_id_map: Dict[str, Tuple[str, str, str]] = {}  # entity_name -> (source_id, description, entity_type)

        for v in vertices:
            if isinstance(v, dict):
                name = str(v.get(_ENTITY_NAME_PROP, ""))
                if name in entity_names:
                    source_id = str(v.get(_SOURCE_ID_PROP, ""))
                    desc = str(v.get(_DESCRIPTION_PROP, ""))
                    etype = str(v.get(_ENTITY_TYPE_PROP, ""))
                    source_id_map[name] = (source_id, desc, etype)

        # 也从 multimodal_entities spec 中补充
        mm_entities_spec = context.get("multimodal_entities", [])
        for spec in mm_entities_spec:
            if hasattr(spec, "entity_name") and spec.entity_name in entity_names:
                if spec.entity_name not in source_id_map:
                    source_id_map[spec.entity_name] = (
                        spec.source_id, spec.description, spec.entity_type,
                    )

        if not source_id_map:
            return chunks

        # 尝试通过向量索引获取 chunk 内容
        vector_index = context.get("vector_index")
        embedding_model = context.get("embedding_model")

        for entity_name, (source_id, description, etype) in source_id_map.items():
            if not source_id or source_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(source_id)

            # 直接得分继承（从关联实体）
            inherited_score = entity_scores.get(entity_name, 0.5)

            # 尝试从向量索引获取 chunk 内容
            chunk_content = self._get_chunk_content(
                source_id, vector_index, embedding_model, context,
            )

            # 若无法从向量索引获取，使用实体 description 作为 chunk 内容
            if not chunk_content:
                chunk_content = description

            chunks.append(MultimodalChunkResult(
                chunk_id=source_id,
                content=chunk_content,
                source_type=etype,
                score=inherited_score,
            ))

        return chunks

    def _get_chunk_content(
        self,
        chunk_id: str,
        vector_index: Any,
        embedding_model: Any,
        context: Dict[str, Any],
    ) -> str:
        """尝试从向量索引或 HugeGraph 获取 chunk 的内容文本"""
        # 1. 从向量索引的 properties 中查找
        if vector_index is not None:
            try:
                all_props = vector_index.get_all_properties()
                if chunk_id in all_props:
                    # 向量索引中 props 通常是字符串形式的 chunk 内容
                    return str(all_props[all_props.index(chunk_id)])
            except Exception:
                pass

        # 2. 从 HugeGraph 获取 TextChunk 顶点
        graph_client = context.get("graph_client")
        if graph_client is not None:
            try:
                v = graph_client.graph().getVertexById(chunk_id)
                if v and v.properties:
                    content = v.properties.get("content", "")
                    if content:
                        return str(content)
            except Exception:
                pass

        # 3. 从 context["raw_chunks"] 或 context["chunks"] 获取
        raw_chunks = context.get("raw_chunks", context.get("chunks", []))
        for chunk in raw_chunks:
            if isinstance(chunk, dict):
                cid = chunk.get("id", chunk.get("chunk_id", ""))
                if cid == chunk_id:
                    return str(chunk.get("content", ""))

        return ""

    # ==================== LLM 上下文格式化 ====================

    def _format_context_for_llm(
        self,
        entities: List[MultimodalEntityResult],
        chunks: List[MultimodalChunkResult],
        edges: List[MultimodalEdgeResult],
    ) -> str:
        """
        将所有检索结果格式化为 LLM 可直接使用的上下文文本。

        关键差异化：每个条目用 [图]/[表]/[公式] 标注类型，
        让 LLM 明确知道信息来自图像/表格/公式还是普通文本。

        格式：
          [图] 实体名: 描述内容 (来源: file_path)
          [表] chunk内容 (score: 0.85)
          [关联] 实体A <-> 实体B: 关联描述
        """
        parts: List[str] = []

        # 1. 多模态实体
        if entities:
            parts.append("--- 多模态实体 ---")
            for i, entity in enumerate(entities, 1):
                source_info = f" (来源: {entity.file_path})" if entity.file_path else ""
                parts.append(
                    f"{entity.label} {i}. {entity.entity_name}: "
                    f"{self._truncate(entity.description, 500)}"
                    f"{source_info}"
                )

        # 2. 多模态 chunks
        if chunks:
            parts.append("--- 多模态内容片段 ---")
            for i, chunk in enumerate(chunks, 1):
                parts.append(
                    f"{chunk.label} {i}. "
                    f"{self._truncate(chunk.content, 800)}"
                )

        # 3. 跨模态关联
        if edges:
            parts.append("--- 跨模态关联 ---")
            for i, edge in enumerate(edges, 1):
                parts.append(
                    f"[关联] {i}. {edge.src_id} <-> {edge.tgt_id}: "
                    f"{self._truncate(edge.description, 300)}"
                )

        return "\n\n".join(parts) if parts else ""

    # ==================== 工具方法 ====================

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """简单分词：中文按字符/常用词切分，英文按空格切分"""
        import re
        english_tokens = set(re.findall(r'[a-zA-Z]{2,}', text.lower()))
        chinese_tokens = set(re.findall(r'[\u4e00-\u9fff]{2,}', text))
        number_tokens = set(re.findall(r'[\d]+(?:\.[\d]+)?%?', text))
        return english_tokens | chinese_tokens | number_tokens

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        """截断文本，保留 max_len 字符"""
        if len(text) <= max_len:
            return text
        return text[:max_len - 1] + "..."

    @staticmethod
    def _parse_vertex_string(vertex_str: str) -> Optional[Dict[str, str]]:
        """
        从 HG-AI 格式的顶点字符串中解析属性。

        格式: "vertex_id{prop1: val1, prop2: val2, ...}"
        """
        if not vertex_str or "{" not in vertex_str:
            return None

        # 提取 ID 和属性部分
        id_part = vertex_str[:vertex_str.index("{")]
        props_part = vertex_str[vertex_str.index("{") + 1:vertex_str.rindex("}")]

        props: Dict[str, str] = {"id": id_part.strip()}
        for prop_pair in props_part.split(","):
            if ":" in prop_pair:
                key, value = prop_pair.split(":", 1)
                props[key.strip()] = value.strip()

        return props


# ========== 向量索引构建工具 ==========


def build_multimodal_vector_index(
    entities: List[Any],
    embedding_model: Any,
    vector_index: Any,
) -> int:
    """
    将多模态实体添加到向量索引，确保它们可以被语义搜索找到。

    借鉴 LightRAG Entity VDB 的存储方式：
      content_for_embedding = f"{entity_name}\\n{description}"
      — entity_name + VLM 分析描述一起嵌入，确保语义搜索可以命中

    Args:
        entities: 多模态实体列表（MultimodalEntitySpec 对象 或 vertex dict）
        embedding_model: 嵌入模型
        vector_index: 向量索引实例

    Returns:
        添加的向量数量
    """
    if not entities:
        return 0

    texts: List[str] = []
    props: List[str] = []

    for entity in entities:
        # 支持 MultimodalEntitySpec 对象和 dict 两种格式
        if hasattr(entity, "entity_name"):
            name = entity.entity_name
            desc = entity.description
            etype = getattr(entity, "entity_type", "drawing")
            source_id = getattr(entity, "source_id", "")
            file_path = getattr(entity, "file_path", "")
        elif isinstance(entity, dict):
            name = str(entity.get(_ENTITY_NAME_PROP, ""))
            desc = str(entity.get(_DESCRIPTION_PROP, ""))
            etype = str(entity.get(_ENTITY_TYPE_PROP, "drawing"))
            source_id = str(entity.get(_SOURCE_ID_PROP, ""))
            file_path = str(entity.get(_FILE_PATH_PROP, ""))
        else:
            continue

        if not name or not desc:
            continue

        # 与 LightRAG 一致：entity_name + description 作为嵌入文本
        embed_text = f"{name}\n{desc}"
        texts.append(embed_text)

        # 属性存储多模态实体的完整信息，供后续过滤和检索
        prop_str = f"{name}{{{_ENTITY_TYPE_PROP}: {etype}, {_DESCRIPTION_PROP}: {desc}, {_SOURCE_ID_PROP}: {source_id}, {_FILE_PATH_PROP}: {file_path}}}"
        props.append(prop_str)

    if not texts:
        return 0

    # 批量嵌入
    try:
        embeddings = embedding_model.get_texts_embeddings(texts)
    except Exception as e:
        log.error("[build_multimodal_vector_index] Embedding failed: %s", e)
        return 0

    # 添加到向量索引
    try:
        vector_index.add(embeddings, props)
        vector_index.save_index_by_name()
        log.info(
            "[build_multimodal_vector_index] Added %d multimodal entity vectors",
            len(texts),
        )
        return len(texts)
    except Exception as e:
        log.error("[build_multimodal_vector_index] Index add failed: %s", e)
        return 0


def ensure_multimodal_entities_in_vector_index(
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    便捷函数：在 MultimodalEntityInjector 之后调用，
    确保多模态实体已被添加到向量索引。

    应在 KG 构建管线中使用：
      injector.run(context)  →  ensure_multimodal_entities_in_vector_index(context)

    Args:
        context: 包含 multimodal_entities, embedding_model, vector_index 的 context

    Returns:
        更新后的 context（新增 context["mm_vector_index_count"]）
    """
    entities = context.get("multimodal_entities", [])
    if not entities:
        # 也尝试从 vertices 中提取多模态实体
        vertices = context.get("vertices", [])
        types_filter = set(MULTIMODAL_ENTITY_TYPES)
        entities = [
            v for v in vertices
            if isinstance(v, dict)
            and str(v.get(_ENTITY_TYPE_PROP, "")) in types_filter
        ]

    if not entities:
        log.debug("[ensure_mm_in_index] No multimodal entities found")
        return context

    embedding_model = context.get("embedding_model")
    vector_index = context.get("vector_index")

    if embedding_model is None or vector_index is None:
        log.warning(
            "[ensure_mm_in_index] Missing embedding_model or vector_index, "
            "cannot add multimodal entities to vector index"
        )
        return context

    count = build_multimodal_vector_index(entities, embedding_model, vector_index)
    context["mm_vector_index_count"] = count

    return context


# ========== 便捷函数 ==========


def multimodal_retrieval_channel(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """一键多模态检索通道的便捷函数"""
    config = MultimodalRetrievalConfig(**kwargs)
    channel = MultimodalRetrievalChannel(config=config)
    return channel.run(context)
