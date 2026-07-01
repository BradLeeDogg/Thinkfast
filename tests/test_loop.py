"""The agent loop, driven by stub engines so no model is required.

run_turn only needs an object with ``generate(messages, tools) -> (text, calls)``.
"""
from agent.engine import ToolCall
from agent.loop import MAX_STEPS, run_turn, run_turn_stream
from agent.memory import Memory


class OneToolThenAnswer:
    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return "I'll calculate.", [ToolCall("calculate", {"expression": "6*7"})]
        # by the 2nd round the real tool result must be in the history
        assert any(m.get("role") == "tool" for m in messages)
        return "The answer is 42.", []


def test_run_turn_executes_tool_then_returns_final_answer():
    mem = Memory()
    out = run_turn(OneToolThenAnswer(), mem, "what is 6*7?")
    assert out == "The answer is 42."
    assert [m["role"] for m in mem.messages] == ["user", "assistant", "tool", "assistant"]
    assert mem.messages[2]["content"] == "42"  # calculator actually ran
    assert mem.messages[1]["tool_calls"][0]["function"]["name"] == "calculate"


class NeverStops:
    def generate(self, messages, tools):
        return "again", [ToolCall("calculate", {"expression": "1+1"})]


def test_max_steps_guard_stops_runaway_tool_calling():
    mem = Memory()
    final = run_turn(NeverStops(), mem, "go")
    assert "maximum number of tool-calling steps" in final
    assert sum(1 for m in mem.messages if m["role"] == "tool") == MAX_STEPS


class CallsUnknownTool:
    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return "", [ToolCall("does_not_exist", {})]
        return "done", []


def test_unknown_tool_error_is_fed_back_to_the_model():
    mem = Memory()
    run_turn(CallsUnknownTool(), mem, "x")
    tool_msg = next(m for m in mem.messages if m["role"] == "tool")
    assert "unknown tool" in tool_msg["content"]


class FakeStreamEngine:
    """Stub engine exposing stream() (what run_turn_stream needs)."""

    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.i = 0

    def stream(self, messages, tools):
        text = self.rounds[self.i]
        self.i += 1
        for ch in text:  # emit char by char to mimic token streaming
            yield ch


def test_run_turn_stream_strips_xml_and_runs_tools():
    rounds = [
        'Let me calculate.\n<tool_call>{"name": "calculate", "arguments": {"expression": "6*7"}}</tool_call>',
        "The answer is 42.",
    ]
    mem = Memory()
    chunks = list(run_turn_stream(FakeStreamEngine(rounds), mem, "what is 6*7?"))
    final = chunks[-1]
    assert "The answer is 42." in final
    assert "42" in final                # the calculator result is surfaced
    assert "<tool_call>" not in final   # raw tool-call XML is never shown
    assert [m["role"] for m in mem.messages] == ["user", "assistant", "tool", "assistant"]
    assert mem.messages[2]["content"] == "42"
