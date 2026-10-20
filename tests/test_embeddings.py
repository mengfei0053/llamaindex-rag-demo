"""Embedding provider selection. No network: fastembed is imported, never constructed."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chromadb
from chromadb.config import Settings as ChromaSettings
from fastapi.testclient import TestClient

from app.config import DEFAULT_LOCAL_EMBED_MODEL, AppConfig, load_config
from app.embeddings import _load_fastembed
from app.main import create_app
from app.rag import RagService

try:
    from test_api import FakeEmbedding, FakeLLM
except ImportError:  # `python -m unittest tests.test_embeddings`
    from tests.test_api import FakeEmbedding, FakeLLM


def _config(data: Path, storage: Path, **overrides) -> AppConfig:
    values = {
        "api_key": "sk-test",
        "base_url": None,
        "chat_model": "fake",
        "embed_model": "text-embedding-3-small",
        "data_dir": data,
        "storage_dir": storage,
        "embedding_provider": "openai",
        "embed_cache_dir": storage.parent / "fastembed-cache",
    }
    values.update(overrides)
    return AppConfig(**values)


ENV_KEYS = (
    "EMBEDDING_PROVIDER",
    "LOCAL_EMBED_MODEL",
    "FASTEMBED_CACHE_PATH",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "OPENAI_EMBED_MODEL",
    "DATA_DIR",
    "STORAGE_DIR",
)


class EnvGuard(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {key: os.environ.get(key) for key in ENV_KEYS}
        for key in ENV_KEYS:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class ConfigTests(EnvGuard):
    def test_default_provider_is_openai(self) -> None:
        config = load_config()
        self.assertEqual(config.embedding_provider, "openai")
        self.assertEqual(config.embed_model, "text-embedding-3-small")
        self.assertEqual(config.local_embed_model, DEFAULT_LOCAL_EMBED_MODEL)
        self.assertEqual(config.chat_model, "gpt-4o-mini")

    def test_local_provider_reads_cache_and_model(self) -> None:
        os.environ["EMBEDDING_PROVIDER"] = "Local"
        os.environ["FASTEMBED_CACHE_PATH"] = "/tmp/fastembed-cache"
        os.environ["LOCAL_EMBED_MODEL"] = "custom/model"
        config = load_config()
        self.assertEqual(config.embedding_provider, "local")
        self.assertEqual(config.embed_cache_dir, Path("/tmp/fastembed-cache"))
        self.assertEqual(config.local_embed_model, "custom/model")

    def test_blank_provider_stays_openai(self) -> None:
        os.environ["EMBEDDING_PROVIDER"] = "  "
        self.assertEqual(load_config().embedding_provider, "openai")

    def test_unknown_provider_is_rejected(self) -> None:
        os.environ["EMBEDDING_PROVIDER"] = "huggingface"
        with self.assertRaises(ValueError):
            load_config()


class ProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data = root / "data"
        self.data.mkdir()
        (self.data / "product.md").write_text(
            "专业版最多 50 名成员，每人每月 99 元。\n",
            encoding="utf-8",
        )
        self.storage = root / "storage"
        self.cache = root / "fastembed-cache"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _local_config(self, model: str = DEFAULT_LOCAL_EMBED_MODEL):
        return _config(
            self.data,
            self.storage,
            api_key="sk-test",
            base_url="http://chat.example/v1",
            chat_model="proxy-chat",
            embedding_provider="local",
            local_embed_model=model,
            embed_cache_dir=self.cache,
        )

    def test_local_ingest_and_query_do_not_call_openai_embeddings(self) -> None:
        captured: dict = {}

        def fake_fastembed(**kwargs):
            captured.update(kwargs)
            return FakeEmbedding()

        with (
            patch("app.embeddings._load_fastembed", return_value=fake_fastembed),
            patch("app.rag.OpenAI", return_value=FakeLLM()) as openai,
            patch("app.rag.OpenAIEmbedding") as openai_embed,
        ):
            app = create_app(self._local_config(), index_on_startup=False)
            with TestClient(app) as client:
                ingest = client.post("/ingest")
                self.assertEqual(ingest.status_code, 200, ingest.text)
                body = ingest.json()
                self.assertGreaterEqual(body["documents"], 1)
                self.assertGreaterEqual(body["chunks"], 1)

                answer = client.post("/query", json={"question": "专业版多少钱？"})
                self.assertEqual(answer.status_code, 200, answer.text)
                payload = answer.json()
                self.assertIn("99", payload["answer"])
                self.assertTrue(payload["sources"])
                self.assertEqual(payload["sources"][0]["file_name"], "product.md")

        openai_embed.assert_not_called()
        self.assertEqual(captured["model_name"], DEFAULT_LOCAL_EMBED_MODEL)
        self.assertEqual(captured["cache_dir"], str(self.cache))
        self.assertTrue(self.cache.is_dir())
        llm_kwargs = openai.call_args.kwargs
        self.assertEqual(llm_kwargs["api_key"], "sk-test")
        self.assertEqual(llm_kwargs["api_base"], "http://chat.example/v1")
        self.assertEqual(llm_kwargs["model"], "proxy-chat")

    def test_local_query_still_requires_api_key(self) -> None:
        config = _config(
            self.data,
            self.storage,
            api_key="",
            embedding_provider="local",
            embed_cache_dir=self.cache,
        )
        app = create_app(config, index_on_startup=False)
        with (
            patch("app.embeddings._load_fastembed", side_effect=AssertionError("download")),
            TestClient(app) as client,
        ):
            response = client.post("/query", json={"question": "专业版多少钱？"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("OPENAI_API_KEY", response.json()["detail"])

    def test_openai_provider_still_uses_openai_embeddings(self) -> None:
        captured: dict = {}

        def fake_embed(**kwargs):
            captured.update(kwargs)
            return FakeEmbedding()

        config = _config(
            self.data,
            self.storage,
            api_key="sk-test",
            base_url="http://chat.example/v1",
            embed_model="text-embedding-3-small",
            embedding_provider="openai",
        )
        with (
            patch("app.rag.OpenAIEmbedding", side_effect=fake_embed),
            patch("app.rag.OpenAI", return_value=FakeLLM()),
            patch(
                "app.embeddings._load_fastembed",
                side_effect=AssertionError("fastembed should not load"),
            ),
        ):
            app = create_app(config, index_on_startup=False)
            with TestClient(app) as client:
                ingest = client.post("/ingest")
                self.assertEqual(ingest.status_code, 200, ingest.text)
                answer = client.post("/query", json={"question": "专业版多少钱？"})
                self.assertEqual(answer.status_code, 200, answer.text)
                self.assertIn("99", answer.json()["answer"])
                self.assertTrue(answer.json()["sources"])

        self.assertEqual(captured["model"], "text-embedding-3-small")
        self.assertEqual(captured["api_key"], "sk-test")
        self.assertEqual(captured["api_base"], "http://chat.example/v1")

    def test_legacy_index_without_stamp_is_reused(self) -> None:
        self.storage.mkdir()
        client = chromadb.PersistentClient(
            path=str(self.storage),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.create_collection(
            "company_docs",
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(ids=["1"], documents=["已有片段"], embeddings=[[0.1, 0.2, 0.3]])
        client.close()

        config = _config(
            self.data,
            self.storage,
            api_key="sk-test",
            embedding_provider="openai",
        )
        service = RagService(config)
        with (
            patch("app.rag.OpenAIEmbedding", return_value=FakeEmbedding()),
            patch("app.rag.OpenAI", return_value=FakeLLM()),
        ):
            documents, chunks = service.ensure_index()
        self.assertEqual(documents, 0)
        self.assertEqual(chunks, 1)

    def test_changing_local_model_rebuilds_stamped_index(self) -> None:
        def fake_fastembed(**kwargs):
            return FakeEmbedding()

        with (
            patch("app.embeddings._load_fastembed", return_value=fake_fastembed),
            patch("app.rag.OpenAI", return_value=FakeLLM()),
        ):
            first = RagService(self._local_config("BAAI/bge-small-zh-v1.5"))
            documents, _chunks = first.ensure_index()
            self.assertGreaterEqual(documents, 1)
            meta = first._chroma().get_collection("company_docs").metadata
            self.assertEqual(meta["embedding_provider"], "local")
            self.assertEqual(meta["embedding_model"], "BAAI/bge-small-zh-v1.5")
            first._client.close()

            second = RagService(self._local_config("sentence-transformers/all-MiniLM-L6-v2"))
            documents, _chunks = second.ensure_index()
            self.assertGreaterEqual(documents, 1)
            meta = second._chroma().get_collection("company_docs").metadata
            self.assertEqual(meta["embedding_model"], "sentence-transformers/all-MiniLM-L6-v2")


class FastEmbedPackageTests(unittest.TestCase):
    def test_default_model_is_supported_without_downloading_it(self) -> None:
        from fastembed import TextEmbedding

        names = {row["model"] for row in TextEmbedding.list_supported_models()}
        self.assertIn(DEFAULT_LOCAL_EMBED_MODEL, names)

    def test_loader_returns_class_without_constructing_it(self) -> None:
        cls = _load_fastembed()
        self.assertEqual(cls.__name__, "FastEmbedEmbedding")
