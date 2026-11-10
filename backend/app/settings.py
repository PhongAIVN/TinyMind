"""Paths and constants for the backend. Every path is relative to backend/, so the app runs from any cwd."""

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND_DIR / "models"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.json"
DATA_DIR = BACKEND_DIR / "data"
DB_PATH = DATA_DIR / "tinymind.db"
MCP_CONFIG_PATH = BACKEND_DIR / "mcp_servers.json"

# What each exported variant is for (shown in the client's model picker).
# chat=False: plain text continuation, no chat template.
# memory=True: stored user facts go into the prompt. Toy-math variants get confused by them
# (the tool model turned "Tuổi: 22" into calculator(8 × 22)), so only the general chat model gets them.
VARIANTS: dict[str, dict] = {
    "sft": {"label": "TinyMind Chat", "description": "Hỏi đáp chung, code", "chat": True, "think": False, "tools": False,
            "memory": True},
    "tool": {"label": "TinyMind Tools", "description": "Gọi tool qua MCP (máy tính)", "chat": True, "think": False, "tools": True},
    "cispo": {"label": "TinyMind Reason", "description": "Toán đồ chơi, có tư duy", "chat": True, "think": True, "tools": False},
    "think": {"label": "TinyMind Think", "description": "Bật/tắt tư duy", "chat": True, "think": True, "tools": False},
    "lora_merged": {"label": "TinyMind LoRA", "description": "Toán đồ chơi", "chat": True, "think": False, "tools": False},
    "student": {"label": "TinyMind Student 13M", "description": "Bản chưng cất, chỉ biết toán đồ chơi", "chat": True, "think": True, "tools": False},
    "pretrain": {"label": "TinyMind Base", "description": "Viết tiếp văn bản", "chat": False, "think": False, "tools": False},
}
DEFAULT_MODEL = "sft"

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 40
DEFAULT_MAX_NEW_TOKENS = 200
MAX_NEW_TOKENS_LIMIT = 400
MAX_TOOL_TURNS = 3
TITLE_LENGTH = 60
