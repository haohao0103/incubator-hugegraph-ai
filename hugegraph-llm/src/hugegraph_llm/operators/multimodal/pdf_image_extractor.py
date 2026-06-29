"""
PDF 图片+文本提取器 — MultimodalGraphRAG Pipeline Stage 1

使用 PyMuPDF (fitz) 解析 PDF，提取：
1. 每页的所有图片（base64编码 + 位置 + 尺寸）
2. 每页的文本块（内容 + 位置）
3. 页面之间的结构关系

输出格式：List[PageResult]，每个 PageResult 包含:
  - page_num: int (0-indexed)
  - page_size: (width, height)
  - images: List[ImageExtract]
  - text_blocks: List[TextBlockExtract]

ImageExtract:
  - image_id: str (自动生成, 格式 "img_{page}_{index}")
  - base64_data: str (JPEG base64 编码, 控制在 512KB 以内)
  - bbox: (x0, y0, x1, y1) 在页面上的坐标
  - size: (width, height) 像素尺寸
  - page_num: int
  
TextBlockExtract:
  - block_id: str (格式 "txt_{page}_{index}")
  - text: str (文本内容, 去除首尾空白)
  - bbox: (x0, y0, x1, y1)
  - page_num: int
"""

import fitz  # PyMuPDF
import base64
import io
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from PIL import Image

log = logging.getLogger(__name__)

@dataclass
class ImageExtract:
    """从 PDF 中提取的图片"""
    image_id: str
    base64_data: str          # JPEG base64
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    size: Tuple[int, int]     # (width, height) pixels
    page_num: int
    # 元数据
    format: str = "jpeg"      # 原始格式
    file_size_bytes: int = 0  # 原始大小
    
    @property
    def data_uri(self) -> str:
        """返回 data URI 格式，可直接用于 VLM API"""
        return f"data:image/jpeg;base64,{self.base64_data}"

@dataclass 
class TextBlockExtract:
    """从 PDF 中提取的文本块"""
    block_id: str
    text: str
    bbox: Tuple[float, float, float, float]
    page_num: int
    # 空间属性
    area: float = 0.0         # bbox 面积
    is_heading: bool = False  # 是否像标题（短文本+大字体）

@dataclass
class PageResult:
    """单页解析结果"""
    page_num: int
    page_size: Tuple[float, float]  # (width, height) points
    images: List[ImageExtract] = field(default_factory=list)
    text_blocks: List[TextBlockExtract] = field(default_factory=list)
    
    @property
    def image_count(self) -> int:
        return len(self.images)
    
    @property
    def text_block_count(self) -> int:
        return len(self.text_blocks)

@dataclass
class PDFExtractionResult:
    """完整 PDF 解析结果"""
    source_path: str
    total_pages: int
    pages: List[PageResult] = field(default_factory=list)
    
    @property
    def total_images(self) -> int:
        return sum(p.image_count for p in self.pages)
    
    @property
    def total_text_blocks(self) -> int:
        return sum(p.text_block_count for p in self.pages)
    
    @property  
    def total_text_length(self) -> int:
        return sum(len(b.text) for p in self.pages for b in p.text_blocks)
    
    def summary(self) -> Dict[str, Any]:
        """返回摘要统计"""
        return {
            "source": self.source_path,
            "pages": self.total_pages,
            "images": self.total_images,
            "text_blocks": self.total_text_blocks,
            "total_chars": self.total_text_length,
        }


class PDFImageExtractor:
    """
    PDF 图片与文本提取器
    
    Usage:
        extractor = PDFImageExtractor(
            max_image_size_kb=512,      # 控制 base64 大小
            min_image_dim=50,            # 忽略过小的图
            extract_images=True,
            extract_text=True,
        )
        
        result = extractor.extract("/path/to/document.pdf")
        print(result.summary())
        
        # 访问第3页的所有图片
        for img in result.pages[2].images:
            print(img.image_id, img.size, img.bbox)
    """
    
    def __init__(
        self,
        max_image_size_kb: int = 512,       # 单张图片最大 KB（压缩阈值）
        jpeg_quality: int = 75,              # JPEG 压缩质量
        min_image_dim: int = 50,             # 最小边长（忽略更小的）
        max_images_per_page: int = 50,       # 每页最大图片数（防爆炸）
        extract_images: bool = True,
        extract_text: bool = True,
    ):
        self.max_image_size_kb = max_image_size_kb
        self.jpeg_quality = jpeg_quality
        self.min_image_dim = min_image_dim
        self.max_images_per_page = max_images_per_page
        self.extract_images = extract_images
        self.extract_text = extract_text
    
    def extract(self, pdf_path: str, pages: Optional[List[int]] = None) -> PDFExtractionResult:
        """
        提取 PDF 中的所有图片和文本
        
        Args:
            pdf_path: PDF 文件路径
            pages: 可选，只提取指定页码列表（0-indexed）。None=全部
            
        Returns:
            PDFExtractionResult 完整结构化数据
        """
        pdf_path = str(Path(pdf_path).resolve())
        
        log.info(f"Opening PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        
        total_pages = len(doc)
        log.info(f"Total pages: {total_pages}")
        
        if pages is not None:
            pages_to_process = [p for p in pages if 0 <= p < total_pages]
        else:
            pages_to_process = list(range(total_pages))
        
        result = PDFExtractionResult(source_path=pdf_path, total_pages=total_pages)
        seen_xrefs = set()  # 用于去重，避免同一图片多次引用
        
        for page_num in pages_to_process:
            if page_num % 10 == 0 and page_num > 0:
                log.info(f"Processing page {page_num + 1}/{total_pages}...")
            
            page = doc.load_page(page_num)
            page_rect = page.rect
            
            page_result = PageResult(
                page_num=page_num,
                page_size=(page_rect.width, page_rect.height)
            )
            
            if self.extract_images:
                images = self._extract_page_images(page, page_num, seen_xrefs)
                page_result.images = images
            
            if self.extract_text:
                text_blocks = self._extract_page_text(page, page_num)
                page_result.text_blocks = text_blocks
            
            result.pages.append(page_result)
        
        doc.close()
        log.info(f"Extraction complete: {result.summary()}")
        
        return result
    
    def _extract_page_images(
        self, 
        page: fitz.Page, 
        page_num: int,
        seen_xrefs: set
    ) -> List[ImageExtract]:
        """提取单页所有图片"""
        images = []
        
        try:
            image_list = page.get_images(full=True)
        except Exception as e:
            log.warning(f"Failed to get images from page {page_num}: {e}")
            return images
        
        for idx, img_info in enumerate(image_list):
            if len(images) >= self.max_images_per_page:
                log.warning(f"Page {page_num}: reached max image limit ({self.max_images_per_page})")
                break
            
            # img_info: (xref, smask, width, height, bcs, colorspace, alt., ...)
            xref = img_info[0]
            width = img_info[2]
            height = img_info[3]
            
            # 去重检查
            if xref in seen_xrefs:
                continue
            
            # 过滤过小的图片
            if width < self.min_image_dim or height < self.min_image_dim:
                continue
            
            try:
                # 提取图片数据
                img_data = page.parent.extract_image(xref)
                
                if img_data is None:
                    continue
                
                raw_bytes = img_data["image"]
                original_ext = img_data["ext"]  # e.g., "png", "jpeg"
                
                # 获取图片在页面上的位置（查找该 xref 的图像区域）
                img_bbox = self._find_image_bbox(page, xref)
                
                # 压缩并转换为 JPEG
                format_str, compressed_bytes = self._compress_image(raw_bytes, original_ext)
                
                # Base64 编码
                b64_data = base64.b64encode(compressed_bytes).decode("ascii")
                
                # 创建 ImageExtract 对象
                image_extract = ImageExtract(
                    image_id=f"img_{page_num}_{idx}",
                    base64_data=b64_data,
                    bbox=img_bbox,
                    size=(width, height),
                    page_num=page_num,
                    format=format_str,
                    file_size_bytes=len(raw_bytes),
                )
                
                images.append(image_extract)
                seen_xrefs.add(xref)
                
            except Exception as e:
                log.warning(f"Failed to extract image {xref} on page {page_num}: {e}")
                continue
        
        return images
    
    def _find_image_bbox(self, page: fitz.Page, xref: int) -> Tuple[float, float, float, float]:
        """查找指定 xref 图片在页面上的边界框"""
        try:
            # 方法1：通过搜索图片引用来找到位置
            for img in page.get_images(full=True):
                if img[0] == xref:
                    # 尝试获取图片的位置信息
                    # 注意：get_images 不直接返回 bbox，我们需要通过其他方式获取
                    
                    # 搜索页面中包含该图片的区域
                    blocks = page.get_text("dict", flags=fitz.TEXT_MEDIABOX)["blocks"]
                    
                    # 遍历所有图像块查找匹配的
                    for block in blocks:
                        if block.get("type") == 1:  # 图像类型
                            # 检查这个图像块是否包含目标 xref（通过位置近似匹配）
                            if hasattr(block, 'image') and block.get('image', {}).get('xref') == xref:
                                return tuple(block['bbox'])
                    
                    # 如果找不到精确匹配，返回页面中心区域作为 fallback
                    rect = page.rect
                    return (
                        rect.width * 0.1,
                        rect.height * 0.1,
                        rect.width * 0.9,
                        rect.height * 0.9
                    )
            
            # 如果没找到，返回默认值
            rect = page.rect
            return (0, 0, rect.width, rect.height)
            
        except Exception:
            rect = page.rect
            return (0, 0, rect.width, rect.height)
    
    def _compress_image(self, raw_bytes: bytes, original_format: str) -> Tuple[str, bytes]:
        """
        压缩图片到目标大小以内
        
        Args:
            raw_bytes: 原始图片字节
            original_format: 原始格式 (PNG/JPEG/etc.)
            
        Returns:
            (format_str, compressed_bytes) 格式和压缩后字节
        """
        target_size = self.max_image_size_kb * 1024  # 转换为字节
        
        try:
            img = Image.open(io.BytesIO(raw_bytes))
            
            # 转换为 RGB（如果需要）
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            current_quality = self.jpeg_quality
            current_img = img.copy()
            
            # 第一次尝试压缩
            buffer = io.BytesIO()
            current_img.save(buffer, format="JPEG", quality=current_quality, optimize=True)
            compressed = buffer.getvalue()
            
            # 如果已经在大小限制内，直接返回
            if len(compressed) <= target_size:
                return ("jpeg", compressed)
            
            # 否则逐步降低质量和缩小尺寸
            for quality in range(current_quality, 20, -5):
                buffer = io.BytesIO()
                current_img.save(buffer, format="JPEG", quality=quality, optimize=True)
                compressed = buffer.getvalue()
                
                if len(compressed) <= target_size:
                    return ("jpeg", compressed)
            
            # 如果降低质量还不够，缩小尺寸
            scale_factor = 0.8
            while scale_factor > 0.3:
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                resized = img.resize((new_width, new_height), Image.LANCZOS)
                
                buffer = io.BytesIO()
                resized.save(buffer, format="JPEG", quality=self.jpeg_quality // 2, optimize=True)
                compressed = buffer.getvalue()
                
                if len(compressed) <= target_size:
                    return ("jpeg", compressed)
                
                scale_factor -= 0.1
            
            # 最后的兜底：强制压缩到很小
            buffer = io.BytesIO()
            small_img = img.resize((img.width // 4, img.height // 4), Image.LANCZOS)
            small_img.save(buffer, format="JPEG", quality=30, optimize=True)
            return ("jpeg", buffer.getvalue())
            
        except Exception as e:
            log.error(f"Failed to compress image: {e}")
            # 兜底：直接返回原始数据的 base64（不压缩）
            return (original_format, raw_bytes)
    
    def _extract_page_text(self, page: fitz.Page, page_num: int) -> List[TextBlockExtract]:
        """提取单页所有文本块"""
        text_blocks = []
        
        try:
            page_dict = page.get_text("dict")
            blocks = page_dict.get("blocks", [])
        except Exception as e:
            log.warning(f"Failed to get text dict from page {page_num}: {e}")
            return text_blocks
        
        for idx, block in enumerate(blocks):
            # 只处理文本块 (type=0)，忽略图片块 (type=1)
            if block.get("type") != 0:
                continue
            
            bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
            
            # 合并所有行的文本
            lines = block.get("lines", [])
            full_text_parts = []
            font_sizes = []
            
            for line in lines:
                spans = line.get("spans", [])
                line_text = ""
                for span in spans:
                    span_text = span.get("text", "")
                    line_text += span_text
                    # 收集字体大小用于判断是否是标题
                    font_size = span.get("size", 0)
                    if font_size > 0:
                        font_sizes.append(font_size)
                
                if line_text.strip():
                    full_text_parts.append(line_text.strip())
            
            # 合并所有行文本
            text = "\n".join(full_text_parts).strip()
            
            # 跳过空白文本块
            if not text:
                continue
            
            # 计算 area
            x0, y0, x1, y1 = bbox
            area = abs((x1 - x0) * (y1 - y0))
            
            # 判断是否是标题：短文本 + 较大字体
            is_heading = False
            if font_sizes:
                avg_font_size = sum(font_sizes) / len(font_sizes)
                # 标题特征：文本较短且字体相对较大
                if len(text) < 100 and avg_font_size > 11:
                    is_heading = True
            
            text_block = TextBlockExtract(
                block_id=f"txt_{page_num}_{idx}",
                text=text,
                bbox=bbox,
                page_num=page_num,
                area=area,
                is_heading=is_heading,
            )
            
            text_blocks.append(text_block)
        
        return text_blocks


# ========== 便捷函数 ==========

def extract_pdf(pdf_path: str, **kwargs) -> PDFExtractionResult:
    """一键提取 PDF 的便捷函数"""
    extractor = PDFImageExtractor(**kwargs)
    return extractor.extract(pdf_path)


def extract_single_page(pdf_path: str, page_num: int) -> PageResult:
    """只提取某一页（用于调试）"""
    extractor = PDFImageExtractor()
    result = extractor.extract(pdf_path, pages=[page_num])
    return result.pages[0]


# ========== CLI 入口 ==========
if __name__ == "__main__":
    import sys
    import json
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_image_extractor.py <pdf_path> [page_num]")
        print("  page_num: optional, extract only this page (0-indexed)")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    page_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if page_arg is not None:
        page = extract_single_page(pdf_path, page_arg)
        print(f"\n=== Page {page_arg} ===")
        print(f"Images: {page.image_count}, Text blocks: {page.text_block_count}")
        for img in page.images:
            print(f"  📷 {img.image_id}: {img.size}px, bbox={img.bbox}, "
                  f"base64_len={len(img.base64_data)}")
        for blk in page.text_blocks:
            preview = blk.text[:80].replace('\n', ' ')
            print(f"  📝 {blk.block_id}: [{blk.area:.0f}px²] {preview}...")
    else:
        result = extract_pdf(pdf_path)
        print(json.dumps(result.summary(), indent=2, ensure_ascii=False))
