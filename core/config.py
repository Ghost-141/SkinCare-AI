import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    """Base configuration common to all environments."""
    APP_NAME: str = "SkinDiseaseAnalysis"
    ENV_MODE: str = "dev"  # dev, staging, prod
    
    # LLM Settings
    LLM_PROVIDER: str = "Groq"  # Groq or Ollama
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    OLLAMA_KEEP_ALIVE: int = -1
    
    # Model & Storage Settings
    MODEL_PATH: str = "models/weights/resnet_v1.pt"
    DATABASE_URL: str = "sqlite:///./data/db/skin_app.db"
    UPLOAD_DIR: str = "data/uploads"

    # Pydantic Settings V2 configuration
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" # Ignore extra env vars
    )

class DevSettings(Settings):
    ENV_MODE: str = "dev"
    model_config = SettingsConfigDict(env_file=".env.dev")

class StagingSettings(Settings):
    ENV_MODE: str = "staging"
    model_config = SettingsConfigDict(env_file=".env.staging")

class ProdSettings(Settings):
    ENV_MODE: str = "prod"
    model_config = SettingsConfigDict(env_file=".env.prod")

@lru_cache()
def get_settings():
    """Factory to return environment-specific settings."""
    # Use environment variable to determine which settings to load
    # Defaults to 'dev' if ENV_MODE is not set
    env_mode = os.getenv("ENV_MODE", "dev").lower()
    
    if env_mode == "prod":
        return ProdSettings()
    elif env_mode == "staging":
        return StagingSettings()
    else:
        # Fallback to .env.dev, and then to .env if .env.dev doesn't exist
        # Pydantic-settings handles the actual file loading logic
        return DevSettings()

settings = get_settings()
