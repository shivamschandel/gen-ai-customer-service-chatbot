import os
from dotenv import load_dotenv

load_dotenv()

# ── API ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Main chat model (free, very fast)
MODEL = "llama-3.3-70b-versatile"

# Vision model for image understanding
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Fast model for sentiment & language detection
FAST_MODEL = "llama-3.1-8b-instant"

MAX_TOKENS = 2048

# ── Knowledge Base ────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR     = "./data/chroma_db"
COLLECTION_NAME        = "nexus_kb"
UPDATE_INTERVAL_HOURS  = 24
DEFAULT_UPDATE_SOURCES = [
    "https://en.wikipedia.org/wiki/Customer_service",
    "https://en.wikipedia.org/wiki/Artificial_intelligence_in_industry",
]

# ── Languages ─────────────────────────────────────────────────────────────────
LANGUAGE_PROFILES = {
    "en":    {"name": "English",              "flag": "🇺🇸", "dir": "ltr"},
    "es":    {"name": "Spanish",              "flag": "🇪🇸", "dir": "ltr"},
    "fr":    {"name": "French",               "flag": "🇫🇷", "dir": "ltr"},
    "de":    {"name": "German",               "flag": "🇩🇪", "dir": "ltr"},
    "ja":    {"name": "Japanese",             "flag": "🇯🇵", "dir": "ltr"},
    "zh-cn": {"name": "Chinese (Simplified)", "flag": "🇨🇳", "dir": "ltr"},
    "ar":    {"name": "Arabic",               "flag": "🇸🇦", "dir": "rtl"},
    "hi":    {"name": "Hindi",                "flag": "🇮🇳", "dir": "ltr"},
    "pt":    {"name": "Portuguese",           "flag": "🇧🇷", "dir": "ltr"},
    "it":    {"name": "Italian",              "flag": "🇮🇹", "dir": "ltr"},
}

# ── Sentiment ─────────────────────────────────────────────────────────────────
SENTIMENT_COLORS = {
    "positive": "#10b981",
    "negative": "#ef4444",
    "neutral":  "#94a3b8",
    "mixed":    "#f59e0b",
}
SENTIMENT_ICONS = {
    "positive": "😊",
    "negative": "😟",
    "neutral":  "😐",
    "mixed":    "😕",
}

# ── arXiv ─────────────────────────────────────────────────────────────────────
ARXIV_CATEGORIES = {
    "cs.AI":  "Artificial Intelligence",
    "cs.LG":  "Machine Learning",
    "cs.CL":  "Computation & Language (NLP)",
    "cs.CV":  "Computer Vision",
    "cs.SE":  "Software Engineering",
    "physics":"Physics",
    "math":   "Mathematics",
    "stat.ML":"Statistics – Machine Learning",
    "eess.SP":"Signal Processing",
    "q-bio":  "Quantitative Biology",
}

# ── System Prompt ─────────────────────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """You are NEXUS, an advanced AI customer service assistant.

## Core Capabilities
- Multilingual: You communicate fluently in all major languages and automatically match the customer's language.
- Empathetic: You detect and adapt to customer emotions, always leading with empathy before solutions.
- Knowledge-Driven: You use a dynamic knowledge base and can reference the latest research.

## Behavior Guidelines
1. Always greet customers warmly and acknowledge their concern first.
2. Match the customer's language exactly — never switch unless they do.
3. Adjust your tone based on detected sentiment.
4. Keep responses concise but complete. Use bullet points for multi-step solutions.
5. End responses with a helpful follow-up question or offer for further assistance.

## Identity
- Name: NEXUS
- Role: AI Customer Service Intelligence Platform
- Personality: Professional, warm, patient, and solution-focused
"""
