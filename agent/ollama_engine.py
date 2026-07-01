"""Ollama backend: talk to a local Ollama server over HTTP.

Ollama (https://ollama.com) runs quantized GGUF models much faster than raw
PyTorch on ordinary CPUs, and needs no torch/transformers — just a running
Ollama server. Tool calls come back structured from Ollama's API; we re-emit
them as ``<tool_call>{...}</tool_call>`` blocks so the agent loop handles them
exactly like the in-process engine does (same interface, no loop changes).

Only the standard library is used here, so this backend stays dependency-free.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .engine import ToolCall, parse_tool_calls, to_openai_tool

DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"


class OllamaEngine:
    """Drop-in replacement for LocalEngine that runs the model in Ollama."""

    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        host: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")

    def _post_chat(self, messages: list[dict], tools: list[dict]):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "tools": [to_openai_tool(t) for t in tools],
            "stream": True,
            "options": {"temperature": self.temperature, "num_predict": self.max_new_tokens},
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urllib.request.urlopen(request)

    def stream(self, messages: list[dict], tools: list[dict]):
        """Yield the assistant's reply as it streams. Tool calls Ollama reports
        are re-emitted as ``<tool_call>`` blocks so the loop parses them the same
        way it does for the in-process engine."""
        try:
            response = self._post_chat(messages, tools)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is it installed and running? ({exc})"
            ) from exc

        with response:
            for line in response:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise RuntimeError(f"Ollama error: {chunk['error']}")
                message = chunk.get("message") or {}
                content = message.get("content")
                if content:
                    yield content
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):  # some builds stringify the args
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    yield "<tool_call>" + json.dumps(
                        {"name": function.get("name"), "arguments": arguments}
                    ) + "</tool_call>"

    def generate(self, messages: list[dict], tools: list[dict]) -> tuple[str, list[ToolCall]]:
        """Stream to stdout and return ``(full_text, parsed_tool_calls)``."""
        pieces: list[str] = []
        for piece in self.stream(messages, tools):
            print(piece, end="", flush=True)
            pieces.append(piece)
        print()
        text = "".join(pieces)
        return text, parse_tool_calls(text)
