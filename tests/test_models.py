from __future__ import annotations

import sys
from types import SimpleNamespace

from hiddenbench.models import build_model_client


def test_ollama_provider_uses_local_defaults_without_api_key(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    client = build_model_client(provider="ollama", model="gemma3:4b-it-qat")

    assert client.model_name == "gemma3:4b-it-qat"
    assert captured["api_key"] == "ollama"
    assert captured["base_url"] == "http://127.0.0.1:11434/v1"
