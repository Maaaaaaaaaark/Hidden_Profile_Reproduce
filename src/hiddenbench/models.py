from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv


Message = dict[str, str]


class ModelClient(Protocol):
    """Minimal interface needed by HiddenBench."""

    model_name: str

    def chat(self, messages: list[Message]) -> str:
        """Return a free-form assistant message."""

    def chat_json(self, messages: list[Message], schema: dict[str, Any]) -> dict[str, Any]:
        """Return a JSON object matching the requested schema as closely as possible."""


class ModelResponseParseError(ValueError):
    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from model text."""

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("Unbalanced JSON object in model response")


PROVIDER_DEFAULTS = {
    "openai": {"api_key_env": "OPENAI_API_KEY", "base_url": None},
    "google": {
        "api_key_env": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "together": {"api_key_env": "TOGETHER_API_KEY", "base_url": "https://api.together.xyz/v1"},
    "qwen": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    },
    "openrouter": {"api_key_env": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1"},
}


class OpenAICompatibleClient:
    """Adapter for OpenAI and OpenAI-compatible chat-completions APIs."""

    def __init__(
        self,
        model: str,
        *,
        provider: str = "openai",
        api_key: str | None = None,
        api_key_env: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        max_retries: int = 3,
        timeout: float | None = None,
        audit_path: str | None = None,
    ) -> None:
        load_dotenv()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the OpenAI dependency with `pip install hiddenbench[openai]`.") from exc

        defaults = PROVIDER_DEFAULTS.get(provider, {})
        env_name = api_key_env or defaults.get("api_key_env")
        resolved_key = api_key or (os.getenv(env_name) if env_name else None)
        if not resolved_key:
            raise ValueError(f"Missing API key. Set {env_name or 'an API key'} or pass api_key.")

        resolved_base_url = base_url if base_url is not None else defaults.get("base_url")
        # SDK retries are off so that `max_retries` below is the only retry layer.
        kwargs: dict[str, Any] = {
            "api_key": resolved_key,
            "max_retries": 0,
            "timeout": 180.0 if timeout is None else timeout,
        }
        if resolved_base_url:
            kwargs["base_url"] = resolved_base_url

        self.client = OpenAI(**kwargs)
        self.model_name = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body
        self.max_retries = max_retries
        self.run_configuration = {
            "provider": provider, "base_url": resolved_base_url, "model": model,
            "temperature": temperature, "max_tokens": max_tokens,
            "extra_body": extra_body, "max_retries": max_retries,
        }
        # One line per API attempt: status, returned model, finish reason, token usage. No message content.
        self.usage_log_path = Path(audit_path) if audit_path else None
        if self.usage_log_path:
            self.usage_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._usage_lock = threading.Lock()

    def _log_call(self, attempt: int, started: float, **fields: Any) -> None:
        if self.usage_log_path is None:
            return
        event = {
            "requested_model": self.model_name,
            "attempt": attempt + 1,
            "elapsed_s": time.monotonic() - started,
            **fields,
        }
        with self._usage_lock, self.usage_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _completion(self, messages: list[Message], response_format: dict[str, Any] | None = None) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            started = time.monotonic()
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                }
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens
                if self.extra_body is not None:
                    kwargs["extra_body"] = self.extra_body
                if response_format is not None:
                    kwargs["response_format"] = response_format
                response = self.client.chat.completions.create(**kwargs)
            except Exception as exc:  # pragma: no cover - exercised in live runs.
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                self._log_call(attempt, started, status="api_error", error=type(exc).__name__, status_code=status_code)
                if status_code in (400, 401, 402, 403, 404, 422):
                    raise
                time.sleep(min(2**attempt, 8))
                continue
            choice = response.choices[0]
            usage = getattr(response, "usage", None)
            self._log_call(
                attempt,
                started,
                status="ok",
                actual_model=getattr(response, "model", None),
                finish_reason=choice.finish_reason,
                usage=usage.model_dump() if hasattr(usage, "model_dump") else usage,
            )
            return choice.message.content or ""
        assert last_error is not None
        raise last_error

    def chat(self, messages: list[Message]) -> str:
        return self._completion(messages)

    def chat_json(self, messages: list[Message], schema: dict[str, Any]) -> dict[str, Any]:
        json_instruction = {
            "role": "user",
            "content": "Return only one valid JSON object. Do not include markdown or commentary.",
        }
        response = self._completion(messages + [json_instruction], response_format={"type": "json_object"})
        try:
            return extract_json_object(response)
        except ValueError as exc:
            raise ModelResponseParseError(str(exc), response) from exc


def build_model_client(
    *,
    provider: str,
    model: str,
    api_key: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    extra_body: dict | None = None,
    max_retries: int = 3,
    audit_path: str | None = None,
) -> ModelClient:
    if provider == "openai-compatible":
        if not base_url:
            raise ValueError("--base-url is required for provider=openai-compatible")
        return OpenAICompatibleClient(
            model,
            provider=provider,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            max_retries=max_retries,
            audit_path=audit_path,
        )
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"Unsupported provider {provider!r}. Use one of {sorted(PROVIDER_DEFAULTS)} or openai-compatible.")
    return OpenAICompatibleClient(
        model,
        provider=provider,
        api_key=api_key,
        api_key_env=api_key_env,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
        max_retries=max_retries,
        audit_path=audit_path,
    )
