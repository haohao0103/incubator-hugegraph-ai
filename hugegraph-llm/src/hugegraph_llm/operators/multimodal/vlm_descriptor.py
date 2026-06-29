"""
VLM 图片描述生成器 — MultimodalGraphRAG Pipeline Stage 2

将 PDF 提取的图片发送给视觉语言模型(VLM)，生成结构化语义描述。
支持小米 MiMo v2.5 Pro 和 OpenAI GPT-4o。

输出格式（每个图片生成一个 ImageDescription）:
  - caption: 一句话概括（中文）
  - detailed_description: 2-3句详细描述（中文）
  - object_labels: 检测到的对象标签列表
  - chart_type: 如果是图表, 类型(bar/line/pie/scatter/table/flowchart/architecture/photo/other)
  - key_insights: 关键信息点列表（如"2023年营收增长23%"）
  - related_keywords: 相关关键词（用于文本-图像跨模态关联）
  - confidence: VLM 对描述的置信度(0-1)
"""

import base64
import logging
import time
import json
import hashlib
import re
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import requests

log = logging.getLogger(__name__)


# ========== VLM 配置 ==========
DEFAULT_VLM_CONFIG = {
    "xiaomimo": {
        "base_url": "https://api.xiaomimimo.com/v1",
        "model": "mimo-v2.5-pro",
        "api_key": "",  # 从环境变量 XIAOMI_MIMO_API_KEY 获取
        "supports_vision": True,
        "max_image_size": 20 * 1024 * 1024,  # 20MB
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key": "",  # 从环境变量 OPENAI_API_KEY 获取
        "supports_vision": True,
        "max_image_size": 20 * 1024 * 1024,
    },
}

# chart_type 允许的枚举值
VALID_CHART_TYPES = {
    "bar", "line", "pie", "scatter", "table",
    "flowchart", "architecture", "schema", "map",
    "screenshot", "photo", "other"
}


@dataclass
class ImageDescription:
    """VLM 生成的图片结构化描述"""
    image_id: str              # 来源图片 ID (如 "img_2_3")
    caption: str               # 一句话概括
    detailed_description: str   # 详细描述
    object_labels: List[str]    # 检测到的对象 ["bar_chart", "revenue", "2023"]
    chart_type: str            # 图表类型分类
    key_insights: List[str]     # 关键信息点
    related_keywords: List[str]  # 跨模态关联关键词
    confidence: float           # 置信度 0-1
    vlm_model: str              # 使用的模型名
    generation_time_ms: int     # 生成耗时(ms)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_chart(self) -> bool:
        return self.chart_type in ("bar", "line", "pie", "scatter", "table")

    @property
    def is_diagram(self) -> bool:
        return self.chart_type in ("flowchart", "architecture", "schema")


@dataclass
class BatchDescribeResult:
    """批量描述结果"""
    total_images: int
    success_count: int
    fail_count: int
    descriptions: List[ImageDescription] = field(default_factory=list)
    failed_ids: List[str] = field(default_factory=list)
    total_time_ms: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.total_images, 1)


class VLMDescriptor:
    """
    视觉语言模型图片描述生成器

    Usage:
        descriptor = VLMDescriptor(
            provider="xiaomimo",          # 或 "openai"
            api_key="your-api-key",
            batch_size=5,                  # 并发批处理
            max_retries=2,                 # 失败重试
            cache_dir=".vlm_cache",        # 缓存目录（避免重复调用）
        )

        # 单张描述
        desc = descriptor.describe(
            image_id="img_0_0",
            base64_data="...",
            page_context="这是第1页，标题是年度财务报告"
        )
        print(desc.caption, desc.chart_type)

        # 批量描述
        result = descriptor.describe_batch(extracted_images)
        print(f"成功: {result.success_count}/{result.total_images}")
    """

    def __init__(
        self,
        provider: str = "xiaomimo",
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        batch_size: int = 5,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        timeout: int = 60,
        cache_dir: Optional[str] = None,
        language: str = "zh",              # 输出语言: "zh" / "en"
    ):
        self.provider = provider
        config = DEFAULT_VLM_CONFIG.get(provider, DEFAULT_VLM_CONFIG["xiaomimo"])

        self.base_url = base_url or config["base_url"]
        self.model = model or config["model"]
        # API key 优先级: 参数 > 环境变量 > 配置默认值
        self.api_key = api_key or os.environ.get(
            "XIAOMI_MIMO_API_KEY" if provider == "xiaomimo" else "OPENAI_API_KEY",
            ""
        ) or config["api_key"]
        self.supports_vision = config.get("supports_vision", True)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.language = language

        # 缓存系统（基于图片内容的 hash）
        self._cache: Dict[str, ImageDescription] = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()

    # ========== 核心：单张图片描述 ==========

    def describe(
        self,
        image_id: str,
        base64_data: str,
        page_context: str = "",
        nearby_text: str = "",
    ) -> ImageDescription:
        """
        为单张图片生成结构化描述

        Args:
            image_id: 图片标识符
            base64_data: JPEG 格式的 base64 编码数据
            page_context: 所在页面的上下文（标题、章节等）
            nearby_text: 图片附近的文本内容（用于跨模态理解）

        Returns:
            ImageDescription 结构化描述
        """
        start_time = time.time()

        # 1. 检查缓存 (基于 base64 的 md5 hash)
        cache_k = self._cache_key(base64_data)
        if cache_k in self._cache:
            log.debug(f"Cache hit for {image_id} (key={cache_k})")
            cached = self._cache[cache_k]
            # 返回副本，覆盖 image_id（同一张图可能出现在不同位置）
            return ImageDescription(
                image_id=image_id,
                caption=cached.caption,
                detailed_description=cached.detailed_description,
                object_labels=list(cached.object_labels),
                chart_type=cached.chart_type,
                key_insights=list(cached.key_insights),
                related_keywords=list(cached.related_keywords),
                confidence=cached.confidence,
                vlm_model=cached.vlm_model,
                generation_time_ms=0,  # 缓存命中不计耗时
            )

        # 2. 构建 prompt
        messages = self._build_prompt(page_context, nearby_text)

        # 3-6. 调用 API、解析、构建结果或兜底
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response_text = self._call_vlm_api(messages, base64_data)
                parsed = self._parse_response(response_text, image_id)

                desc = ImageDescription(
                    image_id=image_id,
                    caption=parsed.get("caption", ""),
                    detailed_description=parsed.get("detailed_description", ""),
                    object_labels=parsed.get("object_labels", []),
                    chart_type=parsed.get("chart_type", "other"),
                    key_insights=parsed.get("key_insights", []),
                    related_keywords=parsed.get("related_keywords", []),
                    confidence=float(parsed.get("confidence", 0.0)),
                    vlm_model=self.model,
                    generation_time_ms=int((time.time() - start_time) * 1000),
                )

                # 写入缓存
                self._cache[cache_k] = desc
                self._save_cache()
                return desc

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = self.retry_delay * (2 ** attempt)  # 指数退避
                    log.warning(
                        f"VLM call failed for {image_id} "
                        f"(attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    log.error(f"All retries exhausted for {image_id}: {e}")

        # 全部失败 → 返回兜底描述
        log.warning(f"Using fallback description for {image_id}")
        fallback = self._fallback_description(image_id)
        fallback.generation_time_ms = int((time.time() - start_time) * 1000)
        return fallback

    def describe_extracted_images(
        self,
        images: list,  # List[ImageExtract] from pdf_image_extractor
        text_blocks: list = None,  # 同页的文本块（用于找邻近文本）
    ) -> BatchDescribeResult:
        """
        批量处理从 PDF 提取的图片

        Args:
            images: ImageExtract 对象列表，每个需有 id, base64, page_num, bbox 属性
            text_blocks: 同页的文本块列表（用于跨模态上下文增强）

        Returns:
            BatchDescribeResult 批量结果
        """
        items = []
        for img in images:
            b64 = getattr(img, "base64", getattr(img, "base64_data", None))
            img_id = getattr(img, "id", getattr(img, "image_id", str(img)))
            if not b64:
                log.warning(f"Skipping image {img_id}: no base64 data")
                continue

            # 尝试找邻近文本
            nearby = ""
            if text_blocks and hasattr(img, "bbox") and img.bbox:
                nearby = self._find_nearby_text(img.bbox, text_blocks)

            page_ctx = ""
            if hasattr(img, "page_num"):
                page_ctx = f"第{img.page_num + 1}页"

            items.append((img_id, b64, page_ctx, nearby))

        return self._describe_batch_with_context(items)

    def describe_batch(
        self,
        items: List[Tuple[str, str]],  # [(image_id, base64_data), ...]
    ) -> BatchDescribeResult:
        """
        批量描述（基础方法）

        Args:
            items: (image_id, base64_data) 元组列表

        Returns:
            BatchDescribeResult 批量结果
        """
        enriched = [(iid, b64, "", "") for iid, b64 in items]
        return self._describe_batch_with_context(enriched)

    def _describe_batch_with_context(
        self,
        items: List[Tuple[str, str, str, str]],  # [(id, b64, ctx, nearby)]
    ) -> BatchDescribeResult:
        """内部批量处理方法，支持带上下文的元组"""
        total_start = time.time()
        result = BatchDescribeResult(
            total_images=len(items),
            success_count=0,
            fail_count=0,
        )

        # 分批处理
        for batch_start in range(0, len(items), self.batch_size):
            batch = items[batch_start:batch_start + self.batch_size]
            log.info(
                f"Processing batch {batch_start // self.batch_size + 1}: "
                f"images [{batch_start}-{batch_start + len(batch) - 1}]"
            )

            for img_id, b64, ctx, nearby in batch:
                try:
                    desc = self.describe(img_id, b64, page_context=ctx, nearby_text=nearby)
                    result.descriptions.append(desc)
                    result.success_count += 1
                except Exception as e:
                    log.error(f"Unexpected error describing {img_id}: {e}")
                    result.failed_ids.append(img_id)
                    result.fail_count += 1

        result.total_time_ms = int((time.time() - total_start) * 1000)
        log.info(
            f"Batch describe complete: {result.success_count}/{result.total_images} "
            f"success ({result.success_rate:.1%}), {result.total_time_ms}ms"
        )
        return result

    # ========== 内部方法 ==========

    def _build_prompt(self, page_context: str, nearby_text: str) -> List[Dict]:
        """构建 VLM 对话 prompt"""
        lang_instruction = "中文" if self.language == "zh" else "English"

        system_prompt = (
            f"你是一个专业的文档视觉分析助手。分析给定的图片，返回严格JSON格式的描述。"
            f"所有文本输出使用{lang_instruction}。\n\n"
            "你必须返回以下JSON结构（不要包含markdown代码块标记）:\n"
            "{\n"
            '  "caption": "一句话概括图片内容",\n'
            '  "detailed_description": "2-3句详细描述图片内容、数据、趋势等",\n'
            '  "object_labels": ["检测到的对象/元素标签"],\n'
            '  "chart_type": "图表类型, 必须是以下之一: '
            f"{'/'.join(sorted(VALID_CHART_TYPES))}" + '",\n'
            '  "key_insights": ["关键信息点, 如具体数字、结论等"],\n'
            '  "related_keywords": ["用于文本检索匹配的关键词"],\n'
            '  "confidence": 0.95\n'
            "}\n\n"
            "规则:\n"
            "1. caption 要简洁有力, 包含最核心的信息\n"
            "2. 如果是图表, key_insights 必须包含具体的数值和趋势\n"
            "3. related_keywords 要包含中英文, 方便后续检索匹配\n"
            "4. confidence 反映你对描述的准确程度 (0.0-1.0)\n"
            "5. 如果图片模糊不清, confidence 设低并说明"
        )

        user_content_parts = []
        if page_context:
            user_content_parts.append({
                "type": "text",
                "text": f"文档上下文: {page_context}"
            })
        if nearby_text:
            user_content_parts.append({
                "type": "text",
                "text": f"图片附近的文字: {nearby_text}"
            })
        user_content_parts.append({
            "type": "text",
            "text": "请分析这张图片, 返回JSON格式的描述:"
        })
        # 图片部分由 _call_vlm_api 追加

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content_parts},
        ]

    def _call_vlm_api(
        self,
        messages: List[Dict],
        image_base64: str,
    ) -> str:
        """
        调用 VLM API（OpenAI 兼容格式，支持 vision）

        Args:
            messages: 对话消息列表（最后一条 user msg 需要追加图片）
            image_base64: JPEG base64 数据

        Returns:
            VLM 响应文本（应该是 JSON 字符串）

        Raises:
            requests.RequestException: 网络/API 错误
            ValueError: 响应解析错误
        """
        # 1. 在最后一条 user message 追加图片 part
        request_messages = json.loads(json.dumps(messages))  # 深拷贝
        last_user_msg = request_messages[-1]
        if last_user_msg.get("role") != "user":
            raise ValueError("Last message must be from user role")

        content = list(last_user_msg["content"])
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        })
        last_user_msg["content"] = content

        # 2. 构建请求
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": request_messages,
            "max_tokens": 1024,
            "temperature": 0.3,
        }

        log.debug(f"Calling VLM API: {url}, model={self.model}")

        # 3. 发送请求
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        # 4. 处理响应
        if response.status_code != 200:
            error_detail = ""
            try:
                error_body = response.json()
                error_detail = error_body.get("error", {}).get("message", "")
            except Exception:
                error_detail = response.text[:500]
            raise requests.RequestException(
                f"VLM API returned status {response.status_code}: {error_detail}"
            )

        # 5. 解析响应体
        try:
            body = response.json()
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response: {response.text[:500]}")

        if "choices" not in body or not body["choices"]:
            raise ValueError(f"No choices in response: {body}")

        message = body["choices"][0].get("message", {})
        content_text = message.get("content", "")

        if not content_text:
            raise ValueError("Empty content in VLM response")

        # 6. 记录 token 使用量
        usage = body.get("usage")
        if usage:
            log.debug(
                f"VLM token usage: prompt={usage.get('prompt_tokens')}, "
                f"completion={usage.get('completion_tokens')}, "
                f"total={usage.get('total_tokens')}"
            )

        return content_text.strip()

    def _parse_response(self, response_text: str, image_id: str) -> Dict:
        """
        解析 VLM 返回的 JSON，容忍 markdown 代码块包装

        Args:
            response_text: VLM 原始响应文本
            image_id: 图片ID（用于日志）

        Returns:
            解析后的字典，保证所有必需字段存在

        Raises:
            ValueError: 无法解析为有效 JSON
        """
        text = response_text.strip()

        # Strategy 1: 直接 JSON 解析
        parsed = self._try_json_parse(text)
        if parsed is not None:
            return self._validate_and_fill(parsed, image_id)

        # Strategy 2: 提取 ```json ... ``` 代码块
        json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_block_match:
            parsed = self._try_json_parse(json_block_match.group(1).strip())
            if parsed is not None:
                return self._validate_and_fill(parsed, image_id)

        # Strategy 3: 提取第一个 {...} 花括号对
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            parsed = self._try_json_parse(brace_match.group(0))
            if parsed is not None:
                return self._validate_and_fill(parsed, image_id)

        # 全部失败
        raise ValueError(
            f"Failed to parse VLM response as JSON for {image_id}. "
            f"Response preview: {text[:200]}"
        )

    @staticmethod
    def _try_json_parse(text: str) -> Optional[Dict]:
        """尝试 JSON 解析，失败返回 None 而非抛异常"""
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _validate_and_fill(self, parsed: Dict, image_id: str) -> Dict:
        """
        验证并填充缺失字段，确保返回符合 schema 的字典
        """
        result = {}

        # caption
        result["caption"] = self._coerce_str(parsed.get("caption", ""), 200)

        # detailed_description
        result["detailed_description"] = self._coerce_str(
            parsed.get("detailed_description", ""), 1000
        )

        # object_labels → 保证是 list[str]
        raw_labels = parsed.get("object_labels", [])
        if isinstance(raw_labels, list):
            result["object_labels"] = [
                str(l) for l in raw_labels if l
            ]
        elif isinstance(raw_labels, str):
            result["object_labels"] = [raw_labels]
        else:
            result["object_labels"] = []

        # chart_type → 枚举校验
        raw_chart = str(parsed.get("chart_type", "other")).strip().lower()
        if raw_chart in VALID_CHART_TYPES:
            result["chart_type"] = raw_chart
        else:
            log.warning(
                f"Invalid chart_type '{raw_chart}' for {image_id}, defaulting to 'other'"
            )
            result["chart_type"] = "other"

        # key_insights → list[str]
        raw_insights = parsed.get("key_insights", [])
        if isinstance(raw_insights, list):
            result["key_insights"] = [str(i) for i in raw_insights if i]
        elif isinstance(raw_insights, str):
            result["key_insights"] = [raw_insights]
        else:
            result["key_insights"] = []

        # related_keywords → list[str]
        raw_kw = parsed.get("related_keywords", [])
        if isinstance(raw_kw, list):
            result["related_keywords"] = [str(k) for k in raw_kw if k]
        elif isinstance(raw_kw, str):
            result["related_keywords"] = [raw_kw]
        else:
            result["related_keywords"] = []

        # confidence → float [0, 1]
        raw_conf = parsed.get("confidence", 0.5)
        try:
            conf_val = float(raw_conf)
            result["confidence"] = max(0.0, min(1.0, conf_val))
        except (TypeError, ValueError):
            result["confidence"] = 0.5

        return result

    @staticmethod
    def _coerce_str(value: Any, max_length: int = 500) -> str:
        """强制转为字符串并截断"""
        if value is None:
            return ""
        s = str(value).strip()
        return s[:max_length] if len(s) > max_length else s

    def _find_nearby_text(
        self,
        image_bbox: tuple,
        text_blocks: list,
        proximity_threshold: float = 100.0,
    ) -> str:
        """
        找到图片附近（上下左右）的文本块，作为 VLM 上下文

        Args:
            image_bbox: 图片边界框 (x0, y0, x1, y1)
            text_blocks: 文本块列表，每个需有 bbox 属性和 text 属性
            proximity_threshold: 距离阈值（像素）

        Returns:
            合并后的邻近文本
        """
        if not image_bbox or not text_blocks:
            return ""

        img_x0, img_y0, img_x1, img_y1 = image_bbox
        img_cx = (img_x0 + img_x1) / 2
        img_cy = (img_y0 + img_y1) / 2

        nearby_texts = []
        for block in text_blocks:
            bbox = getattr(block, "bbox", None)
            text = getattr(block, "text", getattr(block, "content", ""))
            if not bbox or not text:
                continue

            try:
                bx0, by0, bx1, by1 = bbox
                bcx = (bx0 + bx1) / 2
                bcy = (by0 + by1) / 2

                # 计算 bbox 中心到图片中心的距离
                import math
                dist = math.sqrt((bcx - img_cx) ** 2 + (bcy - img_cy) ** 2)

                if dist <= proximity_threshold:
                    nearby_texts.append(text.strip())
            except (TypeError, ValueError, IndexError):
                continue

        result = " ".join(nearby_texts)
        # 截断避免 prompt 过长
        return result[:500] if len(result) > 500 else result

    # ========== 缓存系统 ==========

    def _cache_key(self, base64_data: str) -> str:
        """生成缓存键（基于内容的 hash）"""
        return hashlib.md5(base64_data.encode()).hexdigest()[:16]

    def _load_cache(self):
        """从磁盘加载缓存（如果启用）"""
        if not self.cache_dir or not self.cache_dir.exists():
            return
        cache_file = self.cache_dir / "vlm_descriptions_cache.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                # 反序列化回 ImageDescription
                self._cache = {
                    k: ImageDescription(**v) if isinstance(v, dict) else v
                    for k, v in data.items()
                }
                log.info(f"Loaded {len(self._cache)} cached descriptions from {cache_file}")
            except Exception as e:
                log.warning(f"Failed to load cache: {e}")
                self._cache = {}

    def _save_cache(self):
        """保存缓存到磁盘"""
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / "vlm_descriptions_cache.json"
        serializable = {
            k: v.to_dict() if hasattr(v, 'to_dict') else v
            for k, v in self._cache.items()
        }
        try:
            cache_file.write_text(
                json.dumps(serializable, ensure_ascii=False, indent=2)
            )
        except Exception as e:
            log.warning(f"Failed to save cache: {e}")

    def _fallback_description(self, image_id: str) -> ImageDescription:
        """VLM 调用失败时的兜底描述"""
        return ImageDescription(
            image_id=image_id,
            caption="[VLM调用失败, 使用兜底描述]",
            detailed_description="无法获取视觉语言模型的描述。图片可能需要人工审核。",
            object_labels=["unknown"],
            chart_type="other",
            key_insights=[],
            related_keywords=[],
            confidence=0.0,
            vlm_model=f"{self.provider}-fallback",
            generation_time_ms=0,
        )


# ========== 便捷函数 ==========

def describe_images(
    images: list,  # List[ImageExtract]
    **kwargs
) -> BatchDescribeResult:
    """一键批量描述的便捷函数"""
    desc = VLMDescriptor(**kwargs)
    return desc.describe_extracted_images(images)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="VLM 图片描述生成器")
    parser.add_argument("--image", help="单张图片的 base64 文件路径（每行一个 base64）")
    parser.add_argument("--provider", default="xiaomimo", choices=["xiaomimo", "openai"])
    parser.add_argument("--api-key", default="")
    parser.add_argument("--language", default="zh")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if args.image:
        with open(args.image) as f:
            b64 = f.read().strip()
        desc = VLMDescriptor(
            provider=args.provider,
            api_key=args.api_key,
            language=args.language
        )
        result = desc.describe("test_img", b64)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        parser.print_help()
        print("\n示例用法:")
        print("  echo 'base64data...' > image.b64 && python vlm_descriptor.py --image image.b64")
