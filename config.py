# config.py 

import os

DEBUG = True

USE_GROK = False


def _require_env(name: str, prefix: str = None) -> str:
    """
    Ortam değişkeninden değer okur.
    Zorunlu, yoksa veya format yanlışsa RuntimeError fırlatır.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} ortam değişkeni ayarlı değil. Lütfen sistemine ekle.")
    if prefix and not value.startswith(prefix):
        raise RuntimeError(f"{name} beklenen formatta değil (prefix: {prefix}).")
    return value


# ========= OpenAI =========
OPENAI_API_KEY = _require_env("OPENAI_API_KEY", "sk-")
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"

# ========= GEMINI =========
GEMINI_API_KEY = _require_env("GEMINI_API_KEY", "AIza")
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)

# ========= CLAUDE (Anthropic) =========
CLAUDE_API_KEY = _require_env("ANTHROPIC_API_KEY", "sk-ant-")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_BASE_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_VERSION = "2023-06-01"

# ========= GROK / xAI (şimdilik opsiyonel) =========
GROK_API_KEY = os.getenv("XAI_API_KEY")  # zorunlu değil
GROK_MODEL = "grok-2-latest"
GROK_BASE_URL = "https://api.x.ai/v1/chat/completions"


