"""ProxyChatLLM accepts non-OpenAI model ids used by chat proxies."""

from __future__ import annotations

import unittest

from llama_index.llms.openai import OpenAI

from app.rag import ProxyChatLLM


class ProxyChatLLMTests(unittest.TestCase):
    def test_known_openai_model_uses_stock_metadata(self) -> None:
        llm = ProxyChatLLM(model="gpt-4o-mini", api_key="sk-test")
        self.assertEqual(llm.metadata.model_name, "gpt-4o-mini")
        self.assertGreater(llm.metadata.context_window, 0)

    def test_claude_proxy_id_does_not_raise(self) -> None:
        llm = ProxyChatLLM(model="claude-sonnet-4-6", api_key="sk-test")
        meta = llm.metadata
        self.assertEqual(meta.model_name, "claude-sonnet-4-6")
        self.assertEqual(meta.context_window, 200_000)
        self.assertTrue(meta.is_chat_model)

    def test_stock_openai_still_rejects_unknown(self) -> None:
        llm = OpenAI(model="claude-sonnet-4-6", api_key="sk-test")
        with self.assertRaises(ValueError):
            _ = llm.metadata


if __name__ == "__main__":
    unittest.main()
