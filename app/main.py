import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import router


class RequestContextMiddleware:
    """
    Plain-ASGI middleware: generates/accepts a correlation (X-Request-ID), stashes
    it on ``request.state.correlation_id``, mirrors it into the ``X-Correlation-ID``
    response header, and times each request.

    Implemented as pure ASGI (rather than ``BaseHTTPMiddleware``) so that
    server-sent-event / streaming responses are relayed without buffering.
    When an exception propagates before a response starts, the outer Starlette
    error middleware builds the 500; the correlation id is still present on the
    request scope for downstream handlers.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        correlation_id = ""
        for key, value in scope.get("headers", []):
            if key.lower() == b"x-correlation-id":
                correlation_id = value.decode("utf-8", "replace")
                break
        if not correlation_id:
            correlation_id = uuid.uuid4().hex[:12]

        # Stash on request.state (idempotent; State supports item access).
        try:
            from starlette.datastructures import State

            state = scope.setdefault("state", State())
            state["correlation_id"] = correlation_id
        except Exception:
            scope.setdefault("state", {})["correlation_id"] = correlation_id

        start_ns = time.perf_counter_ns()
        started = {"done": False}

        async def send_wrapper(message):
            if message["type"] == "http.response.start" and not started["done"]:
                started["done"] = True
                elapsed_ms = round((time.perf_counter_ns() - start_ns) / 1e6, 3)
                headers = list(message.get("headers", []))
                headers = [
                    h for h in headers
                    if h[0].lower() not in (b"x-correlation-id", b"x-response-time-ms")
                ]
                headers.append((b"x-correlation-id", correlation_id.encode("utf-8")))
                headers.append((b"x-response-time-ms", str(elapsed_ms).encode("utf-8")))
                message = dict(message)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
    Enterprise Agentic RAG with PGVector Semantic Retrieval and Deterministic Guardrails.
    Designed for high-throughput, low-latency enterprise AI workflows with strict safety boundaries.
    """
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestContextMiddleware)

app.include_router(router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health"
    }