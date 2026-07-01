"""OllamaEngine - verified against a mocked Ollama HTTP stream (no server needed)."""
import json

from agent.engine import ToolCall
from agent.loop import run_turn_stream
from agent.memory import Memory
from agent.ollama_engine import OllamaEngine


class FakeResponse:
    """Stand-in for the urllib streaming response: iterates NDJSON byte lines."""

    def __init__(self, lines):
        self._lines = [ln.encode("utf-8") for ln in lines]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)


def _ndjson(*objs):
    return [json.dumps(o) + "\n" for o in objs]


def test_generate_streams_content_and_extracts_tool_calls(monkeypatch):
    lines = _ndjson(
        {"message": {"role": "assistant", "content": "Let me "}},
        {"message": {"role": "assistant", "content": "compute."}},
        {"message": {"role": "assistant", "content": "",
                     "tool_calls": [{"function": {"name": "calculate",
                                                  "arguments": {"expression": "6*7"}}}]}},
        {"done": True},
    )
    engine = OllamaEngine("qwen2.5:1.5b")
    monkeypatch.setattr(engine, "_post_chat", lambda messages, tools: FakeResponse(lines))

    text, tool_calls = engine.generate([{"role": "user", "content": "6*7?"}], [])
    assert "Let me compute." in text
    assert tool_calls == [ToolCall("calculate", {"expression": "6*7"})]


def test_ollama_drives_the_full_tool_loop(monkeypatch):
    responses = [
        FakeResponse(_ndjson(
            {"message": {"role": "assistant", "content": "",
                         "tool_calls": [{"function": {"name": "calculate",
                                                      "arguments": {"expression": "6*7"}}}]}},
            {"done": True},
        )),
        FakeResponse(_ndjson(
            {"message": {"role": "assistant", "content": "The answer is 42."}},
            {"done": True},
        )),
    ]
    engine = OllamaEngine("qwen2.5:1.5b")
    monkeypatch.setattr(engine, "_post_chat", lambda messages, tools: responses.pop(0))

    mem = Memory()
    final = list(run_turn_stream(engine, mem, "what is 6*7?"))[-1]
    assert "The answer is 42." in final
    assert "42" in final                  # the calculator ran from Ollama's structured tool call
    assert "<tool_call>" not in final     # the re-emitted XML is stripped from the display
    assert [m["role"] for m in mem.messages] == ["user", "assistant", "tool", "assistant"]
    assert mem.messages[2]["content"] == "42"
