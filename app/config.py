import os
try:
    from pydantic_settings import BaseSettings

    def _env_bool(key: str, default: bool) -> bool:
        val = os.getenv(key)
        if val is None:
            return default
        return str(val).strip().lower() in {"1", "true", "yes", "on"}

    class Settings(BaseSettings):
        # --- Core service ---
        PROJECT_NAME: str = "Enterprise Agentic RAG"
        VERSION: str = "0.2.0"
        API_V1_STR: str = "/api/v1"
        DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/agentic_rag")
        CLOUD_REGION: str = os.getenv("CLOUD_REGION", "eu-west-3")
        GUARDRAIL_STRICT_MODE: bool = True
        MAX_EXECUTION_STEPS: int = 5
        SIMILARITY_THRESHOLD: float = 0.70

        # --- Embeddings ---
        EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
        EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "mock")
        EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        # --- LLM ---
        LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")
        LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")
        LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")
        LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

        # --- OpenAI ---
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        # --- Retrieval / rerank ---
        RERANK_PROVIDER: str = os.getenv("RERANK_PROVIDER", "linear")
        RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        TOP_K: int = int(os.getenv("TOP_K", "4"))
        RRV_K: int = int(os.getenv("RRV_K", "60"))
        RETRIEVAL_TIMEOUT_SECONDS: float = float(os.getenv("RETRIEVAL_TIMEOUT_SECONDS", "30"))

        # --- Semantic cache ---
        ENABLE_SEMANTIC_CACHE: bool = _env_bool("ENABLE_SEMANTIC_CACHE", False)
        CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))

        # --- API security / rate limits ---
        API_AUTH_ENABLED: bool = _env_bool("API_AUTH_ENABLED", False)
        API_AUTH_TOKEN: str = os.getenv("API_AUTH_TOKEN", "")
        RATE_LIMIT_ENABLED: bool = _env_bool("RATE_LIMIT_ENABLED", False)
        RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
        RATE_LIMIT_PERIOD_SECONDS: int = int(os.getenv("RATE_LIMIT_PERIOD_SECONDS", "60"))

        # --- Observability / audit ---
        TRACE_ENABLED: bool = _env_bool("TRACE_ENABLED", True)
        AUDIT_LOG_ENABLED: bool = _env_bool("AUDIT_LOG_ENABLED", True)
except ImportError:
    def _env_bool(key: str, default: bool) -> bool:
        val = os.getenv(key)
        if val is None:
            return default
        return str(val).strip().lower() in {"1", "true", "yes", "on"}

    class Settings:
        # --- Core service ---
        PROJECT_NAME: str = "Enterprise Agentic RAG"
        VERSION: str = "0.2.0"
        API_V1_STR: str = "/api/v1"
        DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/agentic_rag")
        CLOUD_REGION: str = os.getenv("CLOUD_REGION", "eu-west-3")
        GUARDRAIL_STRICT_MODE: bool = True
        MAX_EXECUTION_STEPS: int = 5
        SIMILARITY_THRESHOLD: float = 0.70

        # --- Embeddings ---
        EMBEDDING_DIMENSION: int = 1536
        EMBEDDING_PROVIDER: str = "mock"
        EMBEDDING_MODEL: str = "text-embedding-3-small"

        # --- LLM ---
        LLM_PROVIDER: str = "mock"
        LLM_MODEL: str = "gpt-4.1-mini"
        LLM_BASE_URL: str = ""
        LLM_MAX_TOKENS: int = 1024
        LLM_TEMPERATURE: float = 0.1

        # --- OpenAI ---
        OPENAI_API_KEY: str = ""
        OPENAI_BASE_URL: str = "https://api.openai.com/v1"

        # --- Retrieval / rerank ---
        RERANK_PROVIDER: str = "linear"
        RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
        TOP_K: int = 4
        RRV_K: int = 60
        RETRIEVAL_TIMEOUT_SECONDS: float = 30.0

        # --- Semantic cache ---
        ENABLE_SEMANTIC_CACHE: bool = False
        CACHE_SIMILARITY_THRESHOLD: float = 0.95

        # --- API security / rate limits ---
        API_AUTH_ENABLED: bool = False
        API_AUTH_TOKEN: str = ""
        RATE_LIMIT_ENABLED: bool = False
        RATE_LIMIT_REQUESTS: int = 60
        RATE_LIMIT_PERIOD_SECONDS: int = 60

        # --- Observability / audit ---
        TRACE_ENABLED: bool = True
        AUDIT_LOG_ENABLED: bool = True

settings = Settings()