from __future__ import annotations

import sys
from types import SimpleNamespace

from hiddenbench.models import OpenAICompatibleClient, build_model_client


def test_ollama_provider_uses_local_defaults_without_api_key(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    client = build_model_client(provider="ollama", model="gemma3:4b-it-qat", top_p=0.8)

    assert client.model_name == "gemma3:4b-it-qat"
    assert client.top_p == 0.8
    assert captured["api_key"] == "ollama"
    assert captured["base_url"] == "http://127.0.0.1:11434/v1"


def test_json_vote_preserves_role_alternation_and_original_history(monkeypatch):
    client = object.__new__(OpenAICompatibleClient)
    messages = [
        {"role": "system", "content": "Private facts."},
        {"role": "user", "content": "Discuss the options."},
        {"role": "assistant", "content": "Consider option A."},
        {"role": "user", "content": "Vote now."},
    ]
    original = [dict(message) for message in messages]

    def fake_completion(request, response_format=None):
        assert [message["role"] for message in request] == [
            "system", "user", "assistant", "user"
        ]
        assert request[-1]["content"].startswith("Vote now.\n\n")
        assert "Return only one valid JSON object." in request[-1]["content"]
        assert response_format == {"type": "json_object"}
        return '{"vote": "A", "rationale": "Supported by the facts."}'

    monkeypatch.setattr(client, "_completion", fake_completion)
    assert client.chat_json(messages, {})["vote"] == "A"
    assert messages == original


def test_json_instruction_starts_user_turn_after_assistant(monkeypatch):
    client = object.__new__(OpenAICompatibleClient)
    messages = [{"role": "assistant", "content": "Ready to vote."}]

    def fake_completion(request, response_format=None):
        assert [message["role"] for message in request] == ["assistant", "user"]
        return '{"vote": "A", "rationale": "Supported by the facts."}'

    monkeypatch.setattr(client, "_completion", fake_completion)
    assert client.chat_json(messages, {})["vote"] == "A"
    assert len(messages) == 1
