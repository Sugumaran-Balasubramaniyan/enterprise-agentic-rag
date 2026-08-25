import os
try:
    from pydantic_settings import BaseSettings
    class Settings(BaseSettings):
        PROJECT_NAME: str = "Enterprise Agentic RAG"
        VERSION: str = "0.1.0"
        API_V1_STR: str = "/api/v1"
        DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/agentic_rag")
        LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")
        EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
        CLOUD_REGION: str = os.getenv("CLOUD_REGION", "eu-west-3")
        GUARDRAIL_STRICT_MODE: bool = True
        MAX_EXECUTION_STEPS: int = 5
        SIMILARITY_THRESHOLD: float = 0.70
except ImportError:
    class Settings:
        PROJECT_NAME: str = "Enterprise Agentic RAG"
        VERSION: str = "0.1.0"
        API_V1_STR: str = "/api/v1"
        DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/agentic_rag")
        LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")
        EMBEDDING_DIMENSION: int = 1536
        CLOUD_REGION: str = os.getenv("CLOUD_REGION", "eu-west-3")
        GUARDRAIL_STRICT_MODE: bool = True
        MAX_EXECUTION_STEPS: int = 5
        SIMILARITY_THRESHOLD: float = 0.70

settings = Settings()
