## backend/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME:    str = os.getenv("APP_NAME",    "AI SaaS API")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG:       bool = os.getenv("DEBUG", "True") == "True"
    SECRET_KEY:  str = os.getenv("SECRET_KEY",  "dev-secret-key")
    ALGORITHM:   str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite:///./ai_saas.db"
    )
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "*")

settings = Settings()