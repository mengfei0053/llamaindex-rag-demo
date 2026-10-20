"""API smoke test with local fake models. No network and no API key."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)

from app.config import AppConfig
from app.main import create_app


class FakeEmbedding(BaseEmbedding):
    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [byte / 255.0 for byte in digest]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vec(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._vec(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)


class FakeLLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=8192, num_output=64, model_name="fake")

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> CompletionResponse:
        return CompletionResponse(text="专业版每人每月 99 元。")

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs
    ) -> CompletionResponseGen:
        text = "专业版每人每月 99 元。"
        yield CompletionResponse(text=text, delta=text)


class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        data = root / "data"
        data.mkdir()
        (data / "product.md").write_text(
            "专业版最多 50 名成员，每人每月 99 元。\n",
            encoding="utf-8",
        )
        config = AppConfig(
            api_key="",
            base_url=None,
            chat_model="fake",
            embed_model="fake",
            data_dir=data,
            storage_dir=root / "storage",
        )
        self.app = create_app(config)
        self.app.state.rag._configure_models = self._use_fakes

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def _use_fakes() -> None:
        from llama_index.core import Settings

        Settings.llm = FakeLLM()
        Settings.embed_model = FakeEmbedding()
        Settings.chunk_size = 128
        Settings.chunk_overlap = 16

    def test_health_query_and_ingest(self) -> None:
        with TestClient(self.app) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["status"], "ok")
            self.assertFalse(health.json()["index_ready"])

            ingest = client.post("/ingest")
            self.assertEqual(ingest.status_code, 200, ingest.text)
            body = ingest.json()
            self.assertEqual(body["status"], "ok")
            self.assertGreaterEqual(body["documents"], 1)
            self.assertGreaterEqual(body["chunks"], 1)

            answer = client.post("/query", json={"question": "专业版多少钱？"})
            self.assertEqual(answer.status_code, 200, answer.text)
            payload = answer.json()
            self.assertIn("99", payload["answer"])
            self.assertTrue(payload["sources"])
            self.assertIn("99", payload["sources"][0]["text"])
            self.assertEqual(payload["sources"][0]["file_name"], "product.md")

            ready = client.get("/health")
            self.assertTrue(ready.json()["index_ready"])
            self.assertGreaterEqual(ready.json()["chunks"], 1)

            empty = client.post("/query", json={"question": "   "})
            self.assertEqual(empty.status_code, 422)

    def test_missing_api_key_returns_503(self) -> None:
        root = Path(self.tmp.name)
        config = AppConfig(
            api_key="",
            base_url=None,
            chat_model="gpt-4o-mini",
            embed_model="text-embedding-3-small",
            data_dir=root / "data",
            storage_dir=root / "storage-nokey",
        )
        app = create_app(config)
        with TestClient(app) as client:
            response = client.post("/query", json={"question": "专业版多少钱？"})
            self.assertEqual(response.status_code, 503)
            self.assertIn("OPENAI_API_KEY", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
