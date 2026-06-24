from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
DB_PATH = BASE_DIR / "course_study_assistant.db"

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except Exception:
    pass

APP_TITLE = "专业课程资料辅助学习系统 API"

LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat").strip()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "35"))

# DeepSeek / OpenAI / 通义千问等 OpenAI-compatible API 通常都用 /v1/chat/completions。
def chat_completion_url() -> str:
    if LLM_BASE_URL.endswith("/v1"):
        return f"{LLM_BASE_URL}/chat/completions"
    if LLM_BASE_URL.endswith("/chat/completions"):
        return LLM_BASE_URL
    return f"{LLM_BASE_URL}/v1/chat/completions"

SUPPORTED_EXTENSIONS = [".txt", ".md", ".csv", ".pdf", ".docx", ".pptx"]
