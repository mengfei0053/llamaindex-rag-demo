"""Build and query a Chroma-backed LlamaIndex vector index."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import AppConfig

logger = logging.getLogger(__name__)

COLLECTION_NAME = "company_docs"
SNIPPET_CHARS = 1200

QA_TEMPLATE = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "You are a company knowledge assistant. "
    "Answer only from the context above. "
    "If the context does not contain the answer, say you do not know. "
    "Reply in the same language as the query (Chinese or English).\n"
    "Query: {query_str}\n"
    "Answer: "
)


class RagError(Exception):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class RagService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._index: VectorStoreIndex | None = None
        self._client: chromadb.PersistentClient | None = None

    def _chroma(self) -> chromadb.PersistentClient:
        if self._client is None:
            self.config.storage_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.config.storage_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def chunk_count(self) -> int:
        try:
            collection = self._chroma().get_collection(COLLECTION_NAME)
        except Exception:
            return 0
        return int(collection.count())

    @property
    def index_ready(self) -> bool:
        try:
            return self.chunk_count() > 0
        except Exception:
            return False

    def _configure_models(self) -> None:
        if not self.config.api_key:
            raise RagError(
                "OPENAI_API_KEY is required. Set it in the environment or .env.",
                status_code=503,
            )
        client_kwargs = {"api_key": self.config.api_key}
        if self.config.base_url:
            client_kwargs["api_base"] = self.config.base_url
        Settings.llm = OpenAI(
            model=self.config.chat_model,
            temperature=0.1,
            **client_kwargs,
        )
        Settings.embed_model = OpenAIEmbedding(
            model=self.config.embed_model,
            **client_kwargs,
        )
        Settings.chunk_size = 512
        Settings.chunk_overlap = 64

    def _load_documents(self):
        data_dir = self.config.data_dir
        if not data_dir.is_dir():
            raise RagError(f"Data directory does not exist: {data_dir}", status_code=400)
        reader = SimpleDirectoryReader(
            input_dir=str(data_dir),
            required_exts=[".md", ".txt", ".markdown"],
            recursive=True,
        )
        documents = reader.load_data()
        if not documents:
            raise RagError(
                f"No markdown or text documents found in {data_dir}",
                status_code=400,
            )
        return documents

    def _collection(self, *, reset: bool):
        client = self._chroma()
        if reset:
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:
                logger.info("No existing collection to delete")
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _index_from_collection(self, collection) -> VectorStoreIndex:
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )

    def ensure_index(self, *, rebuild: bool = False) -> tuple[int, int]:
        """Load the persisted index, or build it when missing or rebuild is set.

        Returns ``(document_count, chunk_count)``. ``document_count`` is 0 when
        an existing index is reused without reading ``data/``.
        """
        with self._lock:
            try:
                self._configure_models()
                if rebuild or self.chunk_count() == 0:
                    documents = self._load_documents()
                    collection = self._collection(reset=True)
                    vector_store = ChromaVectorStore(chroma_collection=collection)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    self._index = VectorStoreIndex.from_documents(
                        documents,
                        storage_context=storage_context,
                        show_progress=False,
                    )
                    document_count = len(documents)
                else:
                    if self._index is None:
                        self._index = self._index_from_collection(
                            self._collection(reset=False)
                        )
                    document_count = 0
            except RagError as exc:
                self.last_error = str(exc)
                raise
            except Exception as exc:
                logger.exception("index build failed")
                self.last_error = str(exc)
                raise RagError(f"Index build failed: {exc}", status_code=502) from exc
            self.last_error = None
            return document_count, self.chunk_count()

    def query(self, question: str) -> dict:
        question = question.strip()
        if not question:
            raise RagError("question must not be empty", status_code=422)
        self.ensure_index(rebuild=False)
        assert self._index is not None
        engine = self._index.as_query_engine(
            similarity_top_k=self.config.top_k,
            text_qa_template=QA_TEMPLATE,
        )
        try:
            response = engine.query(question)
        except RagError:
            raise
        except Exception as exc:
            logger.exception("query failed")
            raise RagError(f"Query failed: {exc}", status_code=502) from exc

        sources = []
        for node in response.source_nodes:
            content = node.node.get_content().strip()
            if len(content) > SNIPPET_CHARS:
                content = content[:SNIPPET_CHARS] + "..."
            metadata = node.node.metadata or {}
            file_name = metadata.get("file_name")
            if not file_name and metadata.get("file_path"):
                file_name = Path(str(metadata["file_path"])).name
            sources.append(
                {
                    "file_name": file_name,
                    "score": None if node.score is None else float(node.score),
                    "text": content,
                }
            )
        return {"answer": str(response), "sources": sources}
