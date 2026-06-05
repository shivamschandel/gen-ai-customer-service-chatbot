"""
Task 5: Sentiment Analysis using Groq (free, fast LLaMA models)
"""

import json, re, datetime, logging
from typing import Dict, List
from groq import Groq

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    PROMPT = """Analyze the sentiment of this customer service message.
Return ONLY a valid JSON object. No markdown, no explanation, no code fences.

Message: "{text}"

Return exactly this JSON structure:
{{
  "sentiment": "positive or negative or neutral or mixed",
  "confidence": 0.0,
  "intensity": "low or medium or high",
  "emotions": ["emotion1", "emotion2"],
  "key_phrases": ["phrase1", "phrase2"],
  "requires_escalation": false,
  "urgency": "low or medium or high"
}}"""

    TONE_MAP = {
        ("positive","low"):    "Be warm and friendly.",
        ("positive","medium"): "Be enthusiastic and appreciative.",
        ("positive","high"):   "Share their excitement! Be very upbeat.",
        ("negative","low"):    "Be calm, understanding, and solution-focused.",
        ("negative","medium"): "Show genuine empathy before offering help.",
        ("negative","high"):   "Lead with a sincere apology. Be patient and gentle.",
        ("neutral", "low"):    "Be professional, clear, and helpful.",
        ("neutral", "medium"): "Be informative and thorough.",
        ("neutral", "high"):   "Be very detailed and comprehensive.",
        ("mixed",   "low"):    "Acknowledge both positive and negative aspects.",
        ("mixed",   "medium"): "Be empathetic about negatives while building on positives.",
        ("mixed",   "high"):   "Carefully address all aspects with balanced empathy.",
    }

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.client  = Groq(api_key=api_key)
        self.model   = model
        self.history: List[Dict] = []

    def analyze(self, text: str) -> Dict:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=300,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis engine. Return only valid JSON."},
                    {"role": "user",   "content": self.PROMPT.format(text=text[:500])},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$",     "", raw)
            result = json.loads(raw)
        except Exception as e:
            logger.warning(f"Sentiment fallback: {e}")
            result = {
                "sentiment": "neutral", "confidence": 0.5, "intensity": "low",
                "emotions": [], "key_phrases": [], "requires_escalation": False, "urgency": "low",
            }
        result["text"]      = text
        result["timestamp"] = datetime.datetime.now().isoformat()
        self.history.append(result)
        return result

    def get_response_tone(self, s: Dict) -> str:
        key  = (s.get("sentiment", "neutral"), s.get("intensity", "low"))
        tone = self.TONE_MAP.get(key, "Be professional and helpful.")
        if s.get("urgency") == "high":          tone += " This is URGENT — be direct and fast."
        if s.get("requires_escalation"):        tone += " Offer to connect with a human agent."
        return tone

    def get_session_summary(self) -> Dict:
        if not self.history:
            return {"total": 0, "breakdown": {}, "avg_confidence": 0.0, "trend": "stable"}
        breakdown: Dict[str, int] = {}
        total_conf = 0.0
        for h in self.history:
            s = h.get("sentiment", "neutral")
            breakdown[s] = breakdown.get(s, 0) + 1
            total_conf   += h.get("confidence", 0.5)
        recent = [h.get("sentiment") for h in self.history[-3:]]
        trend  = "declining" if recent.count("negative") > 1 else ("improving" if recent.count("positive") > 1 else "stable")
        return {
            "total": len(self.history), "breakdown": breakdown,
            "avg_confidence": round(total_conf / len(self.history), 2), "trend": trend,
            "escalation_needed": any(h.get("requires_escalation") for h in self.history[-3:]),
        }

    def get_history(self):   return self.history
    def clear_history(self): self.history = []
