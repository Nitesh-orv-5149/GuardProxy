import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_ENDPOINT: str = os.getenv("API_ENDPOINT", "http://localhost:11434")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "llama3")
    DRY_RUN: bool = os.getenv("DRY_RUN", "false").lower() in ("1", "true", "yes")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
