# Multimodal Capability Gap Closure — 2026-07-01

## 3项差距补齐状态

| 优先级 | 差距 | 补齐状态 | 实现文件 | Commit |
|--------|------|---------|----------|--------|
| 🔴 P0 | 统一文档解析器 — HG-AI 仅 PDF，LightRAG 覆盖 DOCX/MD/PDF | ✅ CLOSED | `unified_document_parser.py` (4个子解析器) | 9dcf449 |
| 🟡 P1 | VLM Provider 多后端 — 仅 OpenAI-compatible | ✅ CLOSED | `vlm_provider_registry.py` (5个 adapter) | 9dcf449 |
| 🟡 P1 | Multimodal Chunk 检索通道 — 实体注入但图遍历找不到 | ✅ CLOSED | `multimodal_retrieval_channel.py` | 9dcf449 |

## 能力矩阵对比（补齐后）

| # | 能力 | LightRAG | HugeGraph-AI (补齐前) | HugeGraph-AI (补齐后) | 状态 |
|---|------|---------|----------------------|---------------------|------|
| 1 | PDF 文本+图片提取 | PyMuPDF fitz | PyMuPDF fitz | 同前 | ✅ 已有 |
| 2 | DOCX 文档解析 | NativeDocxParser (python-docx + heading/table/image/OMML) | ❌ 无 | `UnifiedDocumentParser._parse_docx` (python-docx) | ✅ 新增 |
| 3 | Markdown 解析 | NativeMarkdownParser (ATX + GFM table + $$equation + ![image]) | ❌ 无 | `UnifiedDocumentParser._parse_markdown` (regex) | ✅ 新增 |
| 4 | 统一输出格式 | IRDoc → 4种 IRBlock | 仅 PDF PageResult | `DocumentExtractionResult` (blocks+images+tables+equations) | ✅ 新增 |
| 5 | VLM 图片描述生成 | 4角色VLM (OpenAI/Ollama/Gemini/Anthropic/Bedrock) | 仅 OpenAI-compatible (xiaomimo/openai) | 5 provider adapters + VLMMultiBackendCaller | ✅ 新增 |
| 6 | 图像输入标准化 | NormalizedImage (raw_bytes+base64+mime+sha256+dims) | 直接 base64 string | `NormalizedVLMImage` + `normalize_vlm_image_inputs` | ✅ 新增 |
| 7 | 多模态实体注入 | extract_entities → mm entity + associated_with edges | `MultimodalEntityInjector` | 同前 + 现有检索通道 | ✅ 已有 |
| 8 | 实体向量嵌入 | Entity VDB content = entity_name + description | 仅文本实体嵌入 | `build_multimodal_vector_index` (mm entity embedding) | ✅ 新增 |
| 9 | 多模态实体检索 | kg_query → _get_node_data → mm entities ride same pipeline | ❌ 无检索通道 | `MultimodalRetrievalChannel._search_multimodal_entities` | ✅ 新增 |
| 10 | 跨模态关联边检索 | _find_most_related_edges_from_entities | ❌ 无 | `_follow_association_edges` (HugeGraph REST + context edges) | ✅ 新增 |
| 11 | 多模态 Chunk 回溯 | entity.source_id → chunk | ❌ 无 | `_retrieve_multimodal_chunks` | ✅ 新增 |
| 12 | LLM 类型标注 | 无特殊标注 (entity_type 自然保留) | ❌ 无 | `[图]/[表]/[公式]` type labeling in text_for_llm | ✅ 新增 |
| 13 | Sidecar IR 系统 | IRImage/IRTable/IREquation + AssetSpec | `sidecar_ir.py` | 同前 | ✅ 已有 |
| 14 | Async VLM Pipeline | asyncio.Semaphore + priority + retry | `AsyncVLMPipeline` | 同前 | ✅ 已有 |
| 15 | 周围上下文增强 | enrich_sidecars_with_surrounding | `SurroundingContextEnricher` | 同前 | ✅ 已有 |

## 剩余差距 (暂未实现)

| # | 差距 | 优先级 | 原因 |
|---|------|--------|------|
| 1 | DOCX OMML→LaTeX 完整转换 | P2 | 已实现简化版，完整版需 LightRAG omml 子包 |
| 2 | DOCX 嵌入图片导出 (DrawingExtractionContext) | P2 | 已实现基础版，完整版需 word/_rels 关系解析 |
| 3 | Markdown 图片 SSRF 防护 | P2 | LightRAG 有 _MarkdownImageResolver + NativeImageRawCache |

## 关键设计借鉴

| LightRAG 设计 | HugeGraph-AI 借用 |
|---------------|-------------------|
| Template method pattern (NativeParserBase) | UnifiedDocumentParser.run() → 路由 → 子解析器 |
| Heading-level block splitting | _parse_docx: Paragraph.style → outlineLvl |
| Provider adapter pattern | VLMProviderAdapter ABC + 5 implementations |
| NormalizedImage dataclass | NormalizedVLMImage (identical structure) |
| Entity VDB content = name + description | build_multimodal_vector_index same format |
| kg_query rides same pipeline | MultimodalRetrievalChannel rides same pipeline |
| Lazy import for dependency isolation | `__getattr__` in __init__.py + lazy wrappers |

## Commit

- **Branch**: `feature/graphrag-baseline`
- **Commit**: `9dcf449`
- **Pushed**: ✅ to origin/feature/graphrag-baseline
- **Files**: 4 (3 new + 1 modified __init__.py)
- **Lines**: ~4,448 insertions
