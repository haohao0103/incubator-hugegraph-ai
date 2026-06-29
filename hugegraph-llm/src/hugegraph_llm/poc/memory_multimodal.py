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

"""
Multimodal memory support for HugeGraph-AI-Memory.

Converts non-text modalities into text descriptions that can be stored as
normal memory nodes in HugeGraph, FAISS and BM25.

Supported:
  - Image → caption (via LLM vision model or local BLIP/CLIP fallback)
  - Audio → transcript (via Whisper API or local whisper.cpp)

The text description is prefixed with a modality marker so downstream
retrieval can distinguish it.
"""

import base64
import os
from typing import Optional

from openai import OpenAI

from hugegraph_llm.config.memory_config import memory_settings
from hugegraph_llm.utils.log import log


class MultimodalEncoder:
    """Facade for image captioning and audio transcription."""

    def __init__(self):
        self._blip = None
        self._whisper = None

    # ------------------------------------------------------------------
    # Image → caption
    # ------------------------------------------------------------------
    def image_to_text(self, image_path: str) -> str:
        """Generate a text caption for an image."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        # Try vision-capable LLM first
        if memory_settings.vision_model:
            return self._llm_vision_caption(image_path)

        # Fallback to local BLIP
        return self._blip_caption(image_path)

    def _llm_vision_caption(self, image_path: str) -> str:
        client = OpenAI(
            api_key=memory_settings.llm_api_key or "",
            base_url=memory_settings.llm_base_url,
        )
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        if ext not in ("png", "jpg", "jpeg", "gif", "webp"):
            ext = "png"
        try:
            resp = client.chat.completions.create(
                model=memory_settings.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "用一句话描述这张图片的内容。"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/{ext};base64,{b64}"},
                            },
                        ],
                    }
                ],
                max_completion_tokens=256,
                temperature=0.3,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            log.error("LLM vision caption failed: %s", e)
            return self._blip_caption(image_path)

    def _blip_caption(self, image_path: str) -> str:
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "transformers and Pillow are required for local image captioning"
            ) from exc
        if self._blip is None:
            model_name = "Salesforce/blip-image-captioning-base"
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name)
            self._blip = (processor, model)
        processor, model = self._blip
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=64)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()

    # ------------------------------------------------------------------
    # Audio → transcript
    # ------------------------------------------------------------------
    def audio_to_text(self, audio_path: str) -> str:
        """Transcribe an audio file to text."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(audio_path)

        # Try Whisper API if configured
        if memory_settings.asr_model:
            return self._whisper_api_transcribe(audio_path)

        # Fallback to local whisper
        return self._local_whisper_transcribe(audio_path)

    def _whisper_api_transcribe(self, audio_path: str) -> str:
        client = OpenAI(
            api_key=memory_settings.llm_api_key or "",
            base_url=memory_settings.llm_base_url,
        )
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=memory_settings.asr_model, file=f
            )
        return (resp.text or "").strip()

    def _local_whisper_transcribe(self, audio_path: str) -> str:
        try:
            import whisper
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is required for local audio transcription"
            ) from exc
        if self._whisper is None:
            self._whisper = whisper.load_model("base")
        result = self._whisper.transcribe(audio_path)
        return result.get("text", "").strip()


def encode_multimodal(path: str) -> str:
    """
    Convert an image or audio file to a text description suitable for memory.

    Returns:
        A text string prefixed with modality marker, e.g.
        "[image] A person sitting at a desk with a laptop."
    """
    ext = os.path.splitext(path)[1].lower()
    encoder = MultimodalEncoder()
    if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
        caption = encoder.image_to_text(path)
        return f"[image] {caption}"
    if ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"):
        transcript = encoder.audio_to_text(path)
        return f"[audio] {transcript}"
    raise ValueError(f"Unsupported multimodal file extension: {ext}")
