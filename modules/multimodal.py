"""
Task 2: Multi-modal Handler using Groq vision model (free)
"""

import base64, io, logging
from typing import Dict, List, Optional, Tuple
from groq import Groq
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_MIME = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_BYTES = 4 * 1024 * 1024
MAX_DIM   = 1568


class MultimodalHandler:
    def __init__(self, api_key: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        self.client     = Groq(api_key=api_key)
        self.model      = model

    def process_image(self, image_file) -> Tuple[str, str]:
        """Returns (base64_string, mime_type)"""
        try:
            if hasattr(image_file, "type"):
                mime = image_file.type if image_file.type in SUPPORTED_MIME else "image/png"
                raw  = image_file.getvalue()
            else:
                raw  = bytes(image_file)
                mime = "image/png"
            if len(raw) > MAX_BYTES:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                raw  = buf.getvalue()
                mime = "image/jpeg"
            return base64.standard_b64encode(raw).decode("utf-8"), mime
        except Exception as e:
            raise ValueError(f"Image processing failed: {e}") from e

    def get_image_info(self, image_file) -> Dict:
        try:
            raw = image_file.getvalue() if hasattr(image_file, "getvalue") else image_file
            img = Image.open(io.BytesIO(raw))
            return {"width": img.width, "height": img.height,
                    "format": img.format, "mode": img.mode, "size_kb": round(len(raw)/1024, 1)}
        except Exception:
            return {}

    def analyze_image(self, b64: str, mime: str,
                      question: str = "", conversation_context: str = "") -> str:
        user_text = question or (
            "Describe this image in detail. Note any text, diagrams, issues, "
            "or features relevant for customer support."
        )
        if conversation_context:
            user_text = f"[Context: {conversation_context}]\n\n{user_text}"
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=800,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": user_text},
                    ],
                }],
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"⚠️ Image analysis failed: {e}"

    def build_vision_content(self, text: str, b64: Optional[str] = None,
                             mime: Optional[str] = None) -> List[Dict]:
        content = []
        if b64 and mime:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}})
        content.append({"type": "text", "text": text})
        return content
