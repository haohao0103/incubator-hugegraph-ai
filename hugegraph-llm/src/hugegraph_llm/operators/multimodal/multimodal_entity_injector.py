"""
多模态实体注入器 — MultimodalGraphRAG Enhancement

将 PDF 中提取的图片、表格、公式作为图谱实体注入，与其他文本实体建立关联边。
借鉴 LightRAG operate.py 的 multimodal entity injection 逻辑（lines 3622-3690），
适配 HugeGraph-AI 的 operator 协议 (run(context) -> context)。

核心思路：
  - drawing/table/equation 不只是 vertex，而是参与图谱推理的实体节点
  - 与其他同 chunk 的文本实体建立 "associated_with" 关联边
  - 保留原始 VLM 描述作为实体 description
  - 保留 sidecar_id 作为实体名称

Usage:
    injector = MultimodalEntityInjector()
    context = injector.run(context)  # 注入多模态实体到 context["vertices"] / context["edges"]
"""

import re
import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

log = logging.getLogger(__name__)


# ========== 常量 ==========

# 多模态类型枚举 — 统一 LightRAG IMAGE_TYPE_ENUM + HG-AI VALID_CHART_TYPES
MULTIMODAL_TYPE_ENUM: Tuple[str, ...] = (
    # LightRAG 首级分类（图像本质）
    "Photo", "Illustration", "Screenshot", "Icon",
    "Chart", "Table", "Infographic", "Flowchart",
    "Chat Log", "Wireframe", "Texture", "Other",
)

# 二级图表类型分类（HG-AI 原有）
CHART_TYPE_ENUM: Tuple[str, ...] = (
    "bar", "line", "pie", "scatter", "table",
    "flowchart", "architecture", "schema", "map",
    "screenshot", "photo", "other",
)

MULTIMODAL_TYPE_FALLBACK = "Other"
CHART_TYPE_FALLBACK = "other"

# 多模态显示名提取 — 借鉴 LightRAG _parse_mm_display_name
_MM_DISPLAY_NAME_PATTERN = re.compile(
    r"^\[(?:Image|Table|Equation) Name\](.+)$",
    flags=re.MULTILINE,
)


# ========== 数据类 ==========

@dataclass
class MultimodalEntitySpec:
    """待注入的多模态实体规格"""
    entity_name: str            # sidecar_id / image_id
    entity_type: str            # "drawing" / "table" / "equation"
    description: str            # VLM 分析结果或 chunk content
    display_name: str           # 人类可读的名称 (从 [Image Name]... 提取或 fallback)
    source_id: str              # 来源 chunk key / block_id
    file_path: str = ""        # 来源文件路径
    timestamp: int = 0          # 创建时间戳


@dataclass
class MultimodalAssociationSpec:
    """多模态实体与文本实体的关联边规格"""
    src_id: str                 # 多模态实体名
    tgt_id: str                 # 文本实体名
    description: str            # 关联描述
    keywords: str               # 关联关键词
    weight: float = 1.0
    source_id: str = ""
    file_path: str = ""
    timestamp: int = 0


# ========== 核心逻辑 ==========

def parse_mm_display_name(content: str, fallback: str) -> str:
    """从多模态 chunk content 中提取人类可读的显示名。

    借鉴 LightRAG operate.py _parse_mm_display_name。
    匹配 [Image Name]... / [Table Name]... / [Equation Name]... 格式。
    """
    if content:
        match = _MM_DISPLAY_NAME_PATTERN.search(content)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate
    return fallback


def classify_multimodal_type(
    vlm_chart_type: str = "",
    vlm_description: str = "",
) -> str:
    """从 VLM 结果推断多模态一级类型。

    LightRAG 使用 IMAGE_TYPE_ENUM (12类)，HG-AI 使用 VALID_CHART_TYPES (12类)。
    此函数做映射：
      bar/line/pie/scatter → Chart
      table → Table
      flowchart/architecture/schema → Flowchart/Infographic
      photo → Photo
      screenshot → Screenshot
      other → Other
    """
    if vlm_chart_type and vlm_chart_type in CHART_TYPE_ENUM:
        mapping = {
            "bar": "Chart", "line": "Chart", "pie": "Chart",
            "scatter": "Chart",
            "table": "Table",
            "flowchart": "Flowchart", "architecture": "Infographic",
            "schema": "Infographic",
            "photo": "Photo", "screenshot": "Screenshot",
            "map": "Illustration",
            "other": "Other",
        }
        return mapping.get(vlm_chart_type, "Other")
    return MULTIMODAL_TYPE_FALLBACK


def build_association_description(
    tgt_entity: str,
    mm_type: str,
    mm_display_name: str,
    heading_label: str = "",
    file_path: str = "",
) -> str:
    """构建多模态关联边的描述文本。

    借鉴 LightRAG operate.py lines 3678-3684 的关联描述生成逻辑。
    """
    location = (
        f"in section {heading_label} of document"
        if heading_label
        else "of document"
    )
    return f"{tgt_entity} is associated with {mm_type} {mm_display_name} {location} \"{file_path}\""


class MultimodalEntityInjector:
    """
    多模态实体注入器 — 将 drawing/table/equation 注入图谱实体。

    直接借鉴 LightRAG operate.py 的 multimodal entity injection 逻辑，
    适配为 HugeGraph-AI operator 协议 (run(context))。

    Context 输入:
      - context["vertices"]: 已有的实体列表
      - context["edges"]: 已有的边列表
      - context["multimodal_items"]: 多模态元素列表，每项含:
          - type: "drawing" / "table" / "equation"
          - id: sidecar_id / image_id
          - content: VLM 分析结果或 chunk content
          - source_id: 来源 chunk key
          - file_path: 来源文件路径 (可选)
          - heading: 所在章节标题 (可选)
          - vlm_chart_type: VLM 推断的图表类型 (可选)
      - context["existing_entities"]: 当前 chunk 中已提取的文本实体名列表

    Context 输出:
      - context["vertices"]: 增加多模态实体节点
      - context["edges"]: 增加多模态关联边
      - context["multimodal_entities"]: 新增的多模态实体列表
      - context["multimodal_associations"]: 新增的关联边列表
    """

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """注入多模态实体和关联边到 context。"""
        multimodal_items = context.get("multimodal_items", [])
        existing_entities = context.get("existing_entities", [])

        if not multimodal_items:
            log.debug("[MultimodalEntityInjector] No multimodal items to inject")
            return context

        # 确保 vertices/edges 列表存在
        if "vertices" not in context:
            context["vertices"] = []
        if "edges" not in context:
            context["edges"] = []

        now_ts = int(time.time())
        mm_entities: List[MultimodalEntitySpec] = []
        mm_associations: List[MultimodalAssociationSpec] = []

        for item in multimodal_items:
            mm_type = item.get("type", "drawing")
            mm_id = item.get("id", "")
            content = item.get("content", "")
            source_id = item.get("source_id", "")
            file_path = item.get("file_path", "")
            heading = item.get("heading", "")
            vlm_chart_type = item.get("vlm_chart_type", "")

            if not mm_id:
                log.warning("[MultimodalEntityInjector] Skipping item with no id")
                continue

            # 1. 构建多模态实体节点
            display_name = parse_mm_display_name(content, mm_id)
            entity_type = classify_multimodal_type(vlm_chart_type, content)

            entity_spec = MultimodalEntitySpec(
                entity_name=mm_id,
                entity_type=entity_type,
                description=content,
                display_name=display_name,
                source_id=source_id,
                file_path=file_path,
                timestamp=now_ts,
            )
            mm_entities.append(entity_spec)

            # 转为 vertex dict 格式（兼容 HG-AI property_graph_extract 输出格式）
            vertex_dict = {
                "entity_name": mm_id,
                "entity_type": mm_type,  # 保留 drawing/table/equation 作为 HG-AI 原生类型
                "description": content,
                "source_id": source_id,
                "file_path": file_path,
                "timestamp": now_ts,
            }
            context["vertices"].append(vertex_dict)

            # 2. 构建关联边 — 与所有同 chunk 的文本实体建立 "associated_with"
            for tgt in existing_entities:
                if tgt == mm_id:
                    continue  # 不与自身关联

                assoc_desc = build_association_description(
                    tgt_entity=tgt,
                    mm_type=mm_type,
                    mm_display_name=display_name,
                    heading_label=heading,
                    file_path=file_path,
                )

                edge_dict = {
                    "src_id": mm_id,
                    "tgt_id": tgt,
                    "description": assoc_desc,
                    "keywords": "associated with, contained in",
                    "weight": 1.0,
                    "source_id": source_id,
                    "file_path": file_path,
                    "timestamp": now_ts,
                }

                mm_associations.append(MultimodalAssociationSpec(
                    src_id=mm_id,
                    tgt_id=tgt,
                    description=assoc_desc,
                    keywords="associated with, contained in",
                    weight=1.0,
                    source_id=source_id,
                    file_path=file_path,
                    timestamp=now_ts,
                ))

                context["edges"].append(edge_dict)

        # 3. 记录注入结果
        context["multimodal_entities"] = mm_entities
        context["multimodal_associations"] = mm_associations

        log.info(
            "[MultimodalEntityInjector] Injected %d multimodal entities "
            "and %d association edges",
            len(mm_entities),
            len(mm_associations),
        )
        return context


# ========== 便捷函数 ==========

def inject_multimodal_entities(context: Dict[str, Any]) -> Dict[str, Any]:
    """一键注入的便捷函数"""
    injector = MultimodalEntityInjector()
    return injector.run(context)
