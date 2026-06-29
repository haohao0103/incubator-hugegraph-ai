"""
多模态知识图谱构建器 — MultimodalGraphRAG Pipeline Stage 3

将 PDF 提取的图片、文本块 + VLM 描述 → 写入 HugeGraph 图数据库。
建立跨模态语义关联：文本↔图像↔描述 的完整图结构。
"""

import json
import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import requests

log = logging.getLogger(__name__)


@dataclass
class BuildStats:
    """构建统计"""
    start_time: float = 0.0
    end_time: float = 0.0
    document_pages: int = 0
    images: int = 0
    text_chunks: int = 0
    image_descriptions: int = 0
    contains_image_edges: int = 0
    contains_text_edges: int = 0
    describes_edges: int = 0
    cross_modal_edges: int = 0
    next_page_edges: int = 0
    vertex_errors: int = 0
    edge_errors: int = 0

    @property
    def total_vertices(self) -> int:
        return self.document_pages + self.images + self.text_chunks + self.image_descriptions

    @property
    def total_edges(self) -> int:
        return sum([self.contains_image_edges, self.contains_text_edges,
                   self.describes_edges, self.cross_modal_edges, self.next_page_edges])

    @property
    def duration_s(self) -> float:
        return round(self.end_time - self.start_time, 1) if self.end_time else 0

    def summary(self) -> Dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "vertices": {"total": self.total_vertices,
                        "pages": self.document_pages, "images": self.images,
                        "text_chunks": self.text_chunks, "descriptions": self.image_descriptions},
            "edges": {"total": self.total_edges,
                     "contains_image": self.contains_image_edges,
                     "contains_text": self.contains_text_edges,
                     "describes": self.describes_edges,
                     "cross_modal": self.cross_modal_edges,
                     "next_page": self.next_page_edges},
            "errors": {"vertex": self.vertex_errors, "edge": self.edge_errors},
        }


class MultimodalKGBuilder:
    """
    多模态知识图谱构建器

    Pipeline:
        PDFExtractionResult + BatchDescribeResult → HugeGraph

    Usage:
        builder = MultimodalKGBuilder(host="http://127.0.0.1:8080", graph="multimodal_poc")
        builder.init_schema()
        stats = builder.build(extraction_result, describe_result)
        print(json.dumps(stats.summary(), indent=2))
    """

    def __init__(
        self,
        host: str = "http://127.0.0.1:8080",
        graph: str = "multimodal_poc",
        auth: tuple = ("admin", "admin"),
        max_retries: int = 3,
        enable_cross_modal: bool = True,
        proximity_threshold: float = 150.0,
    ):
        self.host = host.rstrip("/")
        self.graph = graph
        self.auth = auth
        self.max_retries = max_retries
        self.enable_cross_modal = enable_cross_modal
        self.proximity_threshold = proximity_threshold
        self.base_url = f"{host}/graphs/{self.graph}/graph"
        self.schema_url = f"{host}/graphs/{self.graph}/schema"
        self.stats = BuildStats()

        # Property key 定义缓存
        self._pk_created: set = set()

    # ========== Schema 管理 ==========

    def init_schema(self) -> bool:
        """初始化多模态图谱的 Schema（幂等）"""
        log.info(f"[Schema] Initializing for graph '{self.graph}'...")

        # 1. Property Keys
        pks = [
            ("source_file", "TEXT"), ("page_num", "INT"), ("bbox", "TEXT"),
            ("pixel_size", "TEXT"), ("format", "TEXT"), ("base64_preview", "TEXT"),
            ("content", "TEXT"), ("char_length", "INT"), ("is_heading", "BOOLEAN"),
            ("chunk_index", "INT"),
            ("caption", "TEXT"), ("detailed_description", "TEXT"),
            ("object_labels", "TEXT"), ("chart_type", "TEXT"), ("key_insights", "TEXT"),
            ("related_keywords", "TEXT"), ("confidence", "FLOAT"),
            ("vlm_model", "TEXT"), ("generation_time_ms", "INT"),
            ("document_name", "TEXT"), ("page_width", "FLOAT"), ("page_height", "FLOAT"),
            ("image_count", "INT"), ("text_block_count", "INT"),
            ("position_index", "INT"), ("generated_at", "TEXT"),
            ("relation_type", "TEXT"), ("distance_px", "FLOAT"), ("score", "FLOAT"),
        ]
        for name, dtype in pks:
            self._ensure_property_key(name, dtype)

        # 2. Vertex Labels
        vls = [
            ("Image", ["source_file","page_num","bbox","pixel_size","format"]),
            ("TextChunk", ["content","page_num","bbox","char_length","is_heading","chunk_index"]),
            ("ImageDescription", ["caption","detailed_description","object_labels","chart_type",
                                  "key_insights","related_keywords","confidence","vlm_model","generation_time_ms"]),
            ("DocumentPage", ["document_name","page_num","page_width","page_height",
                             "image_count","text_block_count"]),
        ]
        for name, props in vls:
            self._create_vertex_label(name, props)

        # 3. Edge Labels
        els = [
            ("contains_image", "DocumentPage", "Image", ["position_index"]),
            ("contains_text", "DocumentPage", "TextChunk", ["position_index"]),
            ("describes", "ImageDescription", "Image", ["generated_at"]),
            ("cross_modal_ref", "TextChunk", "ImageDescription", ["relation_type","distance_px","score"]),
            ("next_page", "DocumentPage", "DocumentPage", []),
        ]
        for name, src, tgt, props in els:
            self._create_edge_label(name, src, tgt, props)

        log.info("[Schema] ✅ Initialization complete")
        return True

    # ========== Schema 原子操作 ==========

    def _ensure_property_key(self, name: str, data_type: str) -> bool:
        if name in self._pk_created:
            return True
        try:
            r = requests.post(f"{self.schema_url}/propertykeys/{name}",
                            json={"name": name, "data_type": data_type, "cardinality": "SINGLE"},
                            auth=self.auth, timeout=10)
            if r.status_code in (200, 201, 202):
                self._pk_created.add(name)
                return True
            if r.status_code == 400 and ("already" in r.text.lower() or "existed" in r.text.lower()):
                self._pk_created.add(name)
                return True
            log.debug(f"PK {name}: {r.status_code} {r.text[:80]}")
        except Exception as e:
            log.warning(f"PK {name} error: {e}")
        return False

    def _create_vertex_label(self, name: str, prop_names: List[str]) -> bool:
        props = [{"name": p} for p in prop_names]
        try:
            r = requests.post(
                f"{self.schema_url}/vertexlabels/{name}",
                json={"name": name, "id_strategy": "CUSTOMIZE_STRING",
                      "properties": props, "primary_keys": [], "nullable_keys": [],
                      "enable_label_index": True},
                auth=self.auth, timeout=10,
            )
            ok = r.status_code in (200, 201, 202) or "already" in r.text.lower() or "existed" in r.text.lower()
            if ok:
                log.info(f"[Schema] VertexLabel '{name}' ✓")
            return ok
        except Exception as e:
            log.warning(f"VertexLabel '{name}' error: {e}")
            return False

    def _create_edge_label(self, name: str, source: str, target: str, prop_names=[]) -> bool:
        props = [{"name": p} for p in prop_names]
        try:
            r = requests.post(
                f"{self.schema_url}/edgelabels/{name}",
                json={"name": name, "source_label": source, "target_label": target,
                      "properties": props, "sort_keys": [], "nullable_keys": [],
                      "enable_label_index": True, "frequency": "SINGLE"},
                auth=self.auth, timeout=10,
            )
            ok = r.status_code in (200, 201, 202) or "already" in r.text.lower() or "existed" in r.text.lower()
            if ok:
                log.info(f"[Schema] EdgeLabel '{name}' ({source}->{target}) ✓")
            return ok
        except Exception as e:
            log.warning(f"EdgeLabel '{name}' error: {e}")
            return False

    # ========== 核心：构建流程 ==========

    def build(self, extraction_result, describe_result=None, document_name: str = "") -> BuildStats:
        """
        完整构建流程

        Args:
            extraction_result: PDFExtractionResult 对象 (需有 .pages 属性)
            describe_result: BatchDescribeResult 对象 (需有 .descriptions 属性), 可选
            document_name: 文档名(用于 DocumentPage)

        Returns:
            BuildStats 统计信息
        """
        self.stats = BuildStats()
        self.stats.start_time = time.time()
        pages = getattr(extraction_result, 'pages', extraction_result) if hasattr(extraction_result, 'pages') else []

        try:
            log.info(f"[Build] Starting for {len(pages)} pages...")

            # Phase 1: DocumentPage
            log.info("[Phase 1/7] DocumentPage vertices")
            for page in pages:
                vid = f"page_{page.page_num}"
                ok = self._post_vertex("DocumentPage", vid, {
                    "document_name": document_name,
                    "page_num": page.page_num,
                    "page_width": page.page_size[0],
                    "page_height": page.page_size[1],
                    "image_count": page.image_count,
                    "text_block_count": page.text_block_count,
                })
                if ok: self.stats.document_pages += 1
                else: self.stats.vertex_errors += 1

            # Phase 2: Image
            log.info("[Phase 2/7] Image vertices")
            for page in pages:
                for img in page.images:
                    ok = self._post_vertex("Image", img.image_id, {
                        "source_file": document_name, "page_num": img.page_num,
                        "bbox": f"{img.bbox[0]},{img.bbox[1]},{img.bbox[2]},{img.bbox[3]}",
                        "pixel_size": f"{img.size[0]}x{img.size[1]}",
                        "format": img.format,
                    })
                    if ok: self.stats.images += 1
                    else: self.stats.vertex_errors += 1

            # Phase 3: TextChunk
            log.info("[Phase 3/7] TextChunk vertices")
            for page in pages:
                for idx, blk in enumerate(page.text_blocks):
                    ok = self._post_vertex("TextChunk", blk.block_id, {
                        "content": blk.text, "page_num": blk.page_num,
                        "bbox": f"{blk.bbox[0]},{blk.bbox[1]},{blk.bbox[2]},{blk.bbox[3]}",
                        "char_length": len(blk.text),
                        "is_heading": str(blk.is_heading).lower(),
                        "chunk_index": idx,
                    })
                    if ok: self.stats.text_chunks += 1
                    else: self.stats.vertex_errors += 1

            # Phase 4: Containment edges
            log.info("[Phase 4/7] Containment edges")
            for page in pages:
                page_vid = f"page_{page.page_num}"
                for idx, img in enumerate(page.images):
                    if self._post_edge("contains_image", page_vid, img.image_id, {"position_index": idx}):
                        self.stats.contains_image_edges += 1
                    else: self.stats.edge_errors += 1
                for idx, blk in enumerate(page.text_blocks):
                    if self._post_edge("contains_text", page_vid, blk.block_id, {"position_index": idx}):
                        self.stats.contains_text_edges += 1
                    else: self.stats.edge_errors += 1

            # Phase 5: ImageDescription + describes edges
            descriptions = []
            if describe_result and hasattr(describe_result, 'descriptions'):
                descriptions = describe_result.descriptions
            elif isinstance(describe_result, list):
                descriptions = describe_result

            if descriptions:
                log.info("[Phase 5/7] ImageDescription vertices + describes edges")
                desc_map = {}
                for desc in descriptions:
                    desc_vid = f"desc_{desc.image_id}"
                    desc_map[desc.image_id] = desc
                    ok = self._post_vertex("ImageDescription", desc_vid, {
                        "caption": desc.caption,
                        "detailed_description": desc.detailed_description,
                        "object_labels": json.dumps(getattr(desc, 'object_labels', []), ensure_ascii=False),
                        "chart_type": getattr(desc, 'chart_type', 'other'),
                        "key_insights": json.dumps(getattr(desc, 'key_insights', []), ensure_ascii=False),
                        "related_keywords": json.dumps(getattr(desc, 'related_keywords', []), ensure_ascii=False),
                        "confidence": getattr(desc, 'confidence', 0.0),
                        "vlm_model": getattr(desc, 'vlm_model', ''),
                        "generation_time_ms": getattr(desc, 'generation_time_ms', 0),
                    })
                    if ok:
                        self.stats.image_descriptions += 1
                        if self._post_edge("describes", desc_vid, desc.image_id,
                                          {"generated_at": datetime.now(timezone.utc).isoformat()}):
                            self.stats.describes_edges += 1
                    else:
                        self.stats.vertex_errors += 1

                # Phase 6: Cross-modal edges
                if self.enable_cross_modal:
                    log.info("[Phase 6/7] Cross-modal reference edges")
                    for page in pages:
                        for txt_blk in page.text_blocks:
                            best_img = None; best_dist = float('inf')
                            for img in page.images:
                                d = self._bbox_distance(txt_blk.bloc, img.bbox)
                                if d < best_dist and d < self.proximity_threshold:
                                    best_dist = d; best_img = img
                            if best_img and best_img.image_id in desc_map:
                                rel = "caption_of" if best_dist < 50 and txt_blk.is_heading else (
                                    "spatial_adjacent" if best_dist < 100 else "semantic_related")
                                if self._post_edge("cross_modal_ref", txt_blk.block_id,
                                                   f"desc_{best_img.image_id}",
                                                   {"relation_type": rel, "distance_px": round(best_dist,1),
                                                    "score": round(1-best_dist/self.proximity_threshold, 3)}):
                                    self.stats.cross_modal_edges += 1
                                else:
                                    self.stats.edge_errors += 1

            # Phase 7: Page navigation
            log.info("[Phase 7/7] Page navigation edges")
            for i in range(len(pages)-1):
                if self._post_edge("next_page", f"page_{i}", f"page_{i+1}", {}):
                    self.stats.next_page_edges += 1
                else:
                    self.stats.edge_errors += 1

        except Exception as e:
            log.error(f"[Build] Fatal error: {e}", exc_info=True)

        self.stats.end_time = time.time()
        log.info(f"[Build] ✅ Complete! {json.dumps(self.stats.summary(), indent=2)}")
        return self.stats

    # ========== REST API 底层方法 ==========

    def _post_vertex(self, label: str, vid: str, properties: dict) -> bool:
        for attempt in range(self.max_retries):
            try:
                r = requests.post(f"{self.base_url}/vertices",
                                json={"label": label, "id": vid, "properties": properties},
                                auth=self.auth, timeout=15)
                if r.status_code in (200, 201, 202):
                    return True
                if attempt == self.max_retries - 1:
                    log.debug(f"V [{label}/{vid}]: {r.status_code} {r.text[:100]}")
                    return False
            except Exception:
                if attempt == self.max_retries - 1:
                    return False
                time.sleep(0.5 * (attempt+1))
        return False

    def _post_edge(self, label: str, out_v: str, in_v: str, properties=None) -> bool:
        body = {"label": label, "outV": out_v, "inV": in_v}
        if properties:
            body["properties"] = properties
        for attempt in range(self.max_retries):
            try:
                r = requests.post(f"{self.base_url}/edges", json=body, auth=self.auth, timeout=15)
                if r.status_code in (200, 201, 202):
                    return True
                if attempt == self.max_retries - 1:
                    log.debug(f"E [{label}: {out_v}->{in_v}]: {r.status_code} {r.text[:100]}")
                    return False
            except Exception:
                if attempt == self.max_retries - 1:
                    return False
                time.sleep(0.5 * (attempt+1))
        return False

    @staticmethod
    def _bbox_distance(b1: tuple, b2: tuple) -> float:
        c1 = ((b1[0]+b1[2])/2, (b1[1]+b1[3])/2)
        c2 = ((b2[0]+b2[2])/2, (b2[1]+b2[3])/2)
        return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5


def build_multimodal_kg(extraction_result, describe_result=None, **kwargs) -> BuildStats:
    builder = MultimodalKGBuilder(**kwargs)
    builder.init_schema()
    return builder.build(extraction_result, describe_result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="多模态 KG 构建器")
    parser.add_argument("--host", default="http://127.0.0.1:8080")
    parser.add_argument("--graph", default="multimodal_poc")
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    builder = MultimodalKGBuilder(host=args.host, graph=args.graph)
    if args.init_only:
        builder.init_schema()
    else:
        parser.print_help()
