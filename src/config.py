import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_ENDPOINT: str = os.getenv("API_ENDPOINT", "http://localhost:11434")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "llama3")
    DRY_RUN: bool = os.getenv("DRY_RUN", "false").lower() in ("1", "true", "yes")
    ML_MODEL_DIR: str = os.getenv("ML_MODEL_DIR", "models/input_classifier")
    ML_CLASSIFIER_THRESHOLD: float = float(os.getenv("ML_CLASSIFIER_THRESHOLD", "0.5"))
    ML_MIN_MACRO_F1: float = float(os.getenv("ML_MIN_MACRO_F1", "0.75"))
    ML_MIN_EVAL_ACCURACY: float = float(os.getenv("ML_MIN_EVAL_ACCURACY", "0.8"))
    # Conversation monitor (docs/specs/conversation-monitor.md). Placeholder
    # values until calibrated on benign multi-turn chats + generated attacks.
    CONV_ALPHA: float = 0.8  # ledger decay per turn
    CONV_BETA: float = 0.5  # weight of the newest turn's risk
    CONV_FLAG_THRESHOLD: float = 0.6
    CONV_WINDOW_TURNS: int = 4
    CONV_WINDOW_MAX_CHARS: int = 2000
    CONV_TTL_SECONDS: int = 3600
    CONV_MAX_CONVERSATIONS: int = 10_000
    HF_TOKEN: str = ""
    HF_MODEL_REPO: str = "Kimdokja5149/guardproxy-input-classifier"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
