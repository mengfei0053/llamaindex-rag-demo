"""FastAPI entrypoint for the LlamaIndex RAG demo."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import AppConfig, load_config
from app.rag import RagError, RagService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)


class SourceSnippet(BaseModel):
    file_name: str | None = None
    score: float | None = None
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceSnippet]


class IngestResponse(BaseModel):
    status: str
    documents: int
    chunks: int


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    chunks: int
    error: str | None = None


def create_app(
    config: AppConfig | None = None,
    *,
    index_on_startup: bool = True,
) -> FastAPI:
    config = config or load_config()
    service = RagService(config)

    def _build_index_on_startup() -> None:
        try:
            documents, chunks = service.ensure_index()
            logger.info(
                "index ready documents=%s chunks=%s provider=%s",
                documents,
                chunks,
                config.embedding_provider,
            )
        except Exception as exc:
            service.last_error = str(exc)
            logger.error("index init failed: %s", exc)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Keep /health responsive while the first embedding pass runs.
        if index_on_startup and config.api_key:
            threading.Thread(
                target=_build_index_on_startup,
                name="rag-index",
                daemon=True,
            ).start()
        yield

    app = FastAPI(title="LlamaIndex RAG Demo", lifespan=lifespan)
    app.state.rag = service

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        chunks = 0
        try:
            chunks = service.chunk_count()
        except Exception as exc:
            service.last_error = str(exc)
        return HealthResponse(
            status="ok",
            index_ready=chunks > 0,
            chunks=chunks,
            error=service.last_error,
        )

    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest) -> QueryResponse:
        try:
            result = service.query(body.question)
        except RagError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        return QueryResponse(**result)

    @app.post("/ingest", response_model=IngestResponse)
    def ingest() -> IngestResponse:
        try:
            documents, chunks = service.ensure_index(rebuild=True)
        except RagError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("ingest failed")
            raise HTTPException(status_code=502, detail=f"Ingest failed: {exc}") from exc
        return IngestResponse(status="ok", documents=documents, chunks=chunks)

    return app


app = create_app()
