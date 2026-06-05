"""
Task 6: Multi-language Support using Groq (free)
"""

import logging
from typing import Dict, List
from groq import Groq

logger = logging.getLogger(__name__)

LANGUAGE_PROFILES: Dict[str, Dict] = {
    "en":    {"name": "English",              "flag": "🇺🇸", "dir": "ltr", "system_note": "Respond naturally and warmly in English."},
    "es":    {"name": "Spanish",              "flag": "🇪🇸", "dir": "ltr", "system_note": "Responde íntegramente en español. Usa 'usted' para un tono formal."},
    "fr":    {"name": "French",               "flag": "🇫🇷", "dir": "ltr", "system_note": "Réponds entièrement en français. Utilise 'vous'."},
    "de":    {"name": "German",               "flag": "🇩🇪", "dir": "ltr", "system_note": "Antworte vollständig auf Deutsch. Verwende 'Sie'."},
    "ja":    {"name": "Japanese",             "flag": "🇯🇵", "dir": "ltr", "system_note": "日本語で完全に応答してください。丁寧語を使用してください。"},
    "zh-cn": {"name": "Chinese (Simplified)", "flag": "🇨🇳", "dir": "ltr", "system_note": "请完全用简体中文回答。使用正式、礼貌的语气。"},
    "ar":    {"name": "Arabic",               "flag": "🇸🇦", "dir": "rtl", "system_note": "أجب بالكامل باللغة العربية الفصحى."},
    "hi":    {"name": "Hindi",                "flag": "🇮🇳", "dir": "ltr", "system_note": "पूरी तरह हिंदी में जवाब दें। आदरपूर्ण भाषा का उपयोग करें।"},
    "pt":    {"name": "Portuguese",           "flag": "🇧🇷", "dir": "ltr", "system_note": "Responda completamente em português brasileiro."},
    "it":    {"name": "Italian",              "flag": "🇮🇹", "dir": "ltr", "system_note": "Rispondi completamente in italiano. Usa 'Lei'."},
}

LANGDETECT_MAP = {"zh": "zh-cn", "zh-tw": "zh-cn"}


class LanguageHandler:
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.client  = Groq(api_key=api_key)
        self.model   = model
        self._cache: Dict[str, str] = {}
        self.history: List[str]     = []

    def detect_language(self, text: str) -> str:
        text = text.strip()
        if not text or len(text) < 4: return "en"
        if text in self._cache:       return self._cache[text]
        try:
            from langdetect import detect
            code = detect(text)
            code = LANGDETECT_MAP.get(code, code)
            if code not in LANGUAGE_PROFILES: code = "en"
        except Exception:
            code = "en"
        self._cache[text] = code
        self.history.append(code)
        return code

    def build_language_instruction(self, lang_code: str) -> str:
        p = LANGUAGE_PROFILES.get(lang_code, LANGUAGE_PROFILES["en"])
        return (
            f"\n\n## LANGUAGE REQUIREMENT\n{p['system_note']}\n"
            f"The user is communicating in **{p['name']}**. "
            f"You MUST respond ENTIRELY in {p['name']}. Do NOT switch languages."
        )

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if source_lang == target_lang or not text.strip(): return text
        src = LANGUAGE_PROFILES.get(source_lang, {}).get("name", source_lang)
        tgt = LANGUAGE_PROFILES.get(target_lang, {}).get("name", target_lang)
        try:
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=800,
                messages=[{"role": "user", "content": f"Translate from {src} to {tgt}. Return ONLY the translated text:\n\n{text}"}],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text

    def get_profile(self, lang_code: str) -> Dict:
        return LANGUAGE_PROFILES.get(lang_code, LANGUAGE_PROFILES["en"])

    def all_languages(self) -> Dict[str, Dict]:
        return LANGUAGE_PROFILES

    def dominant_session_language(self) -> str:
        if not self.history: return "en"
        return max(set(self.history), key=self.history.count)
