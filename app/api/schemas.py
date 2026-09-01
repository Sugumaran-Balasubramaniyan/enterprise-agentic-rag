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

class DocumentSummaryResponse(BaseModel):
    document_id: str = Field(default="", description="Unique document ID")
    title: str = Field(default="", description="Document title or identifier")
    department: str = Field(default="", description="Associated department")
    chunk_count: int = Field(default=0, description="Total indexed chunks for document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")

class DocumentChunkDetail(BaseModel):
    id: str = Field(default="", description="Unique chunk ID")
    document_id: Optional[str] = Field(default="", description="Parent document ID")
    content: str = Field(default="", description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")

class DocumentDetailResponse(BaseModel):
    document_id: str = Field(default="", description="Unique document ID")
    title: str = Field(default="", description="Document title")
    department: str = Field(default="", description="Associated department")
    total_chunks: int = Field(default=0, description="Total chunk count")
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="List of chunk objects")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")

class DeleteDocumentResponse(BaseModel):
    document_id: str = Field(default="", description="Deleted document ID")
    chunks_deleted: int = Field(default=0, description="Number of chunks deleted")
    status: str = Field(default="success", description="Deletion status")
    message: str = Field(default="", description="Detailed status message")

class SystemMetricsResponse(BaseModel):
    total_queries: int = Field(default=0, description="Total queries executed")
    blocked_queries: int = Field(default=0, description="Total queries blocked by guardrails")
    avg_latency_ms: float = Field(default=0.0, description="Average query latency in milliseconds")
    p95_latency_ms: float = Field(default=0.0, description="95th percentile query latency in milliseconds")
    total_documents: int = Field(default=0, description="Total indexed documents")
    total_chunks: int = Field(default=0, description="Total indexed chunks across all documents")
    active_backend: str = Field(default="in_memory", description="Active storage engine backend")

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
