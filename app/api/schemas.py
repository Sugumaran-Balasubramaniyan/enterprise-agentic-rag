from typing import List, Dict, Any, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def dict(self):
            return self.__dict__
    def Field(default=None, **kwargs):
        return default

class DocumentIngestRequest(BaseModel):
    title: str = Field(default="", description="Document title")
    content: str = Field(default="", description="Full text content of document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata tags")

class DocumentIngestResponse(BaseModel):
    document_id: str = ""
    chunks_created: int = 0
    status: str = "success"

class QueryRequest(BaseModel):
    query: str = Field(default="", description="User query to the enterprise agent")
    user_role: str = Field(default="standard_user", description="RBAC Role")
    stream: bool = False

class ToolExecutionTrace(BaseModel):
    tool_name: str = ""
    arguments: Dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    latency_ms: float = 0.0

class QueryResponse(BaseModel):
    answer: str = ""
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_steps: List[str] = Field(default_factory=list)
    tool_traces: List[ToolExecutionTrace] = Field(default_factory=list)
    guardrail_metrics: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    db_connected: bool = True
