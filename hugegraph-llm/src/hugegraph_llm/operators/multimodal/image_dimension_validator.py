# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Image dimension validation — check image size before sending to VLM.

Adapted from LightRAG ``llm/_vision_utils.py``. Reads only the file header
(no Pillow dependency). Supports PNG, JPEG, GIF, and WebP (VP8/VP8L/VP8X).
Rejects images that are too small (< min_width/min_height) or too large
(> max_width/max_height) before burning VLM API tokens.
"""

from __future__ import annotations

import base64
import hashlib
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Magic byte signatures
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"
_GIF_SIGNATURES = (b"GIF87a", b"GIF89a")
_WEBP_RIFF = b"RIFF"
_WEBP_TAG = b"WEBP"

# Data URL regex
DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[\w./+-]+);base64,(?P<data>[A-Za-z0-9+/=\s]+)$"
)


@dataclass(frozen=True)
class ImageValidationResult:
    """Result of image dimension validation."""

    path_or_id: str
    width: int | None = None
    height: int | None = None
    mime_type: str | None = None
    sha256: str | None = None
    bytes_size: int | None = None
    accepted: bool = True
    reason: str | None = None


@dataclass
class ImageDimensionValidatorConfig:
    """Configuration for image dimension validation."""

    min_width: int = 50
    min_height: int = 50
    max_width: int = 8192
    max_height: int = 8192
    max_file_bytes: int = 20 * 1024 * 1024  # 20 MB
    allowed_mime_types: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
    )


def _detect_mime(raw: bytes) -> str:
    """Infer MIME type from magic bytes."""
    if raw.startswith(_PNG_SIGNATURE):
        return "image/png"
    if raw.startswith(_JPEG_SIGNATURE):
        return "image/jpeg"
    if any(raw.startswith(sig) for sig in _GIF_SIGNATURES):
        return "image/gif"
    if len(raw) >= 12 and raw[0:4] == _WEBP_RIFF and raw[8:12] == _WEBP_TAG:
        return "image/webp"
    return "image/png"


def _read_png_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or not data.startswith(_PNG_SIGNATURE):
        return None
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _read_gif_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 10 or not any(data.startswith(sig) for sig in _GIF_SIGNATURES):
        return None
    width, height = struct.unpack("<HH", data[6:10])
    return width, height


def _read_jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4 or not data.startswith(_JPEG_SIGNATURE):
        return None
    i = 2
    n = len(data)
    while i < n:
        if data[i] != 0xFF:
            return None
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            return None
        marker = data[i]
        i += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if i + 2 > n:
            return None
        segment_len = struct.unpack(">H", data[i : i + 2])[0]
        if segment_len < 2 or i + segment_len > n:
            return None
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 7 > n:
                return None
            height, width = struct.unpack(">HH", data[i + 3 : i + 7])
            return width, height
        i += segment_len
    return None


def _read_webp_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 30 or data[0:4] != _WEBP_RIFF or data[8:12] != _WEBP_TAG:
        return None
    chunk_type = data[12:16]
    if chunk_type == b"VP8 ":
        if len(data) < 30:
            return None
        width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return width, height
    if chunk_type == b"VP8L":
        if len(data) < 25 or data[20] != 0x2F:
            return None
        b0, b1, b2, b3 = data[21], data[22], data[23], data[24]
        width = ((b1 & 0x3F) << 8 | b0) + 1
        height = ((b3 & 0x0F) << 10 | b2 << 2 | (b1 & 0xC0) >> 6) + 1
        return width, height
    if chunk_type == b"VP8X":
        if len(data) < 30:
            return None
        width = (data[24] | data[25] << 8 | data[26] << 16) + 1
        height = (data[27] | data[28] << 8 | data[29] << 16) + 1
        return width, height
    return None


def _dimensions_from_bytes(data: bytes) -> tuple[int, int] | None:
    """Run the four header readers against a byte buffer."""
    if not data:
        return None
    for reader in (
        _read_png_dimensions,
        _read_gif_dimensions,
        _read_jpeg_dimensions,
        _read_webp_dimensions,
    ):
        try:
            dims = reader(data)
        except (struct.error, IndexError, ValueError):
            continue
        if dims:
            return dims
    return None


def read_image_dimensions_from_path(path: Path) -> tuple[int, int] | None:
    """Return (width, height) for a raster image file, reading only the header."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(64 * 1024)
    except OSError:
        return None
    return _dimensions_from_bytes(header)


def read_image_dimensions_from_base64(b64_data: str) -> tuple[int, int] | None:
    """Return (width, height) for a base64-encoded image."""
    try:
        cleaned = re.sub(r"\s+", "", b64_data)
        raw_bytes = base64.b64decode(cleaned, validate=True)
    except (base64.binascii.Error, ValueError):
        return None
    return _dimensions_from_bytes(raw_bytes)


class ImageDimensionValidator:
    """Validate image dimensions before sending to VLM.

    Adapted from LightRAG ``llm/_vision_utils.py``. Supports PNG, JPEG, GIF,
    WebP. No Pillow dependency — reads only file headers.

    Usage::

        validator = ImageDimensionValidator()
        result = validator.validate_path("/path/to/image.png")
        # or
        result = validator.validate_base64("iVBORw0KGgo...")
        # or
        results = validator.run({"images": ["/path1.png", "/path2.jpg"]})
    """

    def __init__(self, config: ImageDimensionValidatorConfig | None = None) -> None:
        self.config = config or ImageDimensionValidatorConfig()

    def _check_dims(
        self,
        width: int | None,
        height: int | None,
        path_or_id: str,
        mime_type: str | None,
        file_bytes: int | None,
    ) -> ImageValidationResult:
        """Apply dimension/size rules."""
        c = self.config

        # Check MIME type
        if mime_type and mime_type not in c.allowed_mime_types:
            return ImageValidationResult(
                path_or_id=path_or_id,
                width=width,
                height=height,
                mime_type=mime_type,
                accepted=False,
                reason=f"mime type {mime_type} not in allowed types",
            )

        # Check file size
        if file_bytes is not None and file_bytes > c.max_file_bytes:
            return ImageValidationResult(
                path_or_id=path_or_id,
                width=width,
                height=height,
                mime_type=mime_type,
                bytes_size=file_bytes,
                accepted=False,
                reason=f"file size {file_bytes} exceeds max {c.max_file_bytes}",
            )

        # Check dimensions
        if width is not None and height is not None:
            if width < c.min_width or height < c.min_height:
                return ImageValidationResult(
                    path_or_id=path_or_id,
                    width=width,
                    height=height,
                    mime_type=mime_type,
                    accepted=False,
                    reason=f"dimensions {width}x{height} below minimum "
                    f"{c.min_width}x{c.min_height}",
                )
            if width > c.max_width or height > c.max_height:
                return ImageValidationResult(
                    path_or_id=path_or_id,
                    width=width,
                    height=height,
                    mime_type=mime_type,
                    accepted=False,
                    reason=f"dimensions {width}x{height} exceed maximum "
                    f"{c.max_width}x{c.max_height}",
                )

        return ImageValidationResult(
            path_or_id=path_or_id,
            width=width,
            height=height,
            mime_type=mime_type,
            bytes_size=file_bytes,
            accepted=True,
        )

    def validate_path(self, path: str | Path) -> ImageValidationResult:
        """Validate an image file by path."""
        p = Path(path)
        dims = read_image_dimensions_from_path(p)
        width, height = (dims[0], dims[1]) if dims else (None, None)

        # Read raw bytes for mime + size + sha256
        try:
            raw = p.read_bytes()
            mime_type = _detect_mime(raw)
            file_bytes = len(raw)
            sha256 = hashlib.sha256(raw).hexdigest()
        except OSError as exc:
            return ImageValidationResult(
                path_or_id=str(p),
                accepted=False,
                reason=f"cannot read file: {exc}",
            )

        result = self._check_dims(width, height, str(p), mime_type, file_bytes)
        # Patch in sha256
        return ImageValidationResult(
            path_or_id=result.path_or_id,
            width=result.width,
            height=result.height,
            mime_type=result.mime_type,
            sha256=sha256,
            bytes_size=result.bytes_size,
            accepted=result.accepted,
            reason=result.reason,
        )

    def validate_base64(self, b64_data: str, *, path_or_id: str = "") -> ImageValidationResult:
        """Validate a base64-encoded image."""
        dims = read_image_dimensions_from_base64(b64_data)
        width, height = (dims[0], dims[1]) if dims else (None, None)

        try:
            cleaned = re.sub(r"\s+", "", b64_data)
            # Handle data URL
            match = DATA_URL_RE.match(cleaned)
            if match:
                mime_type = match.group("mime")
                raw_bytes = base64.b64decode(match.group("data"), validate=True)
            else:
                raw_bytes = base64.b64decode(cleaned, validate=True)
                mime_type = _detect_mime(raw_bytes)
        except (base64.binascii.Error, ValueError) as exc:
            return ImageValidationResult(
                path_or_id=path_or_id,
                accepted=False,
                reason=f"invalid base64: {exc}",
            )

        file_bytes = len(raw_bytes)
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if not mime_type:
            mime_type = _detect_mime(raw_bytes)

        result = self._check_dims(width, height, path_or_id, mime_type, file_bytes)
        return ImageValidationResult(
            path_or_id=result.path_or_id,
            width=result.width,
            height=result.height,
            mime_type=result.mime_type,
            sha256=sha256,
            bytes_size=result.bytes_size,
            accepted=result.accepted,
            reason=result.reason,
        )

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """HG-AI operator protocol: validate images from context.

        Context keys:
        - ``images``: list of paths (str/Path) or base64 strings
        - ``image_mode``: ``"path"`` or ``"base64"`` (default: auto-detect)
        """
        images = context.get("images", [])
        mode = context.get("image_mode", "auto")
        results: list[ImageValidationResult] = []

        for item in images:
            if mode == "path" or (mode == "auto" and isinstance(item, (str, Path)) and not item.startswith("data:")):
                # Treat as file path
                try:
                    p = Path(item)
                    if p.exists():
                        results.append(self.validate_path(item))
                    else:
                        # Maybe it's base64 that looks like a path string
                        results.append(self.validate_base64(str(item), path_or_id=str(item)))
                except Exception:
                    results.append(ImageValidationResult(
                        path_or_id=str(item),
                        accepted=False,
                        reason="auto-detection failed",
                    ))
            else:
                # Treat as base64
                results.append(self.validate_base64(str(item), path_or_id=str(item)[:80]))

        context["image_validation_results"] = results
        context["accepted_images"] = [r for r in results if r.accepted]
        context["rejected_images"] = [r for r in results if not r.accepted]
        return context
