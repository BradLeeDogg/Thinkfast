from .engine import parse_tool_calls, strip_tool_calls
from .memory import Memory
from .tools import TOOL_SCHEMAS, TOOLS

# Stop runaway tool-calling: cap how many generate -> run-tools rounds one user
# turn may take before we give up.
MAX_STEPS = 10


def _execute_tool(name: str, tool_input: dict) -> str:
    handler = TOOLS.get(name)
    if handler is None:
        return f"Error: unknown tool '{name}'"
    try:
        return handler(**tool_input)
    except Exception as exc:  # tool implementations are arbitrary; surface any failure to the model
        return f"Error running '{name}': {exc}"


def run_turn(engine, memory: Memory, user_input: str) -> str:
    """Run one user turn through the agent loop until the model stops calling tools.

    ``engine`` only needs a ``generate(messages, tools) -> (text, tool_calls)``
    method, so a stub can stand in for the real model in tests.
    """
    memory.append({"role": "user", "content": user_input})

    for _ in range(MAX_STEPS):
        text, tool_calls = engine.generate(memory.messages, TOOL_SCHEMAS)

        if not tool_calls:
            memory.append({"role": "assistant", "content": text})
            return text

        memory.append(
            {
                "role": "assistant",
                "content": strip_tool_calls(text),
                "tool_calls": [
                    {"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                    for c in tool_calls
                ],
            }
        )
        for call in tool_calls:
            result = _execute_tool(call.name, call.arguments)
            memory.append({"role": "tool", "name": call.name, "content": result})

    final = "(stopped: reached the maximum number of tool-calling steps)"
    memory.append({"role": "assistant", "content": final})
    return final


def run_turn_stream(engine, memory: Memory, user_input: str):
    """Generator form of :func:`run_turn` for UIs.

    Yields the assistant's running reply as markdown — tool-call syntax stripped
    out, each executed tool shown as a step — so a front-end can render it as it
    streams. ``engine`` must provide ``stream(messages, tools)`` yielding text
    pieces (``LocalEngine`` does).
    """
    memory.append({"role": "user", "content": user_input})

    transcript = ""
    for _ in range(MAX_STEPS):
        raw = ""
        for piece in engine.stream(memory.messages, TOOL_SCHEMAS):
            raw += piece
            yield transcript + strip_tool_calls(raw)

        tool_calls = parse_tool_calls(raw)
        message = {"role": "assistant", "content": strip_tool_calls(raw)}
        if tool_calls:
            message["tool_calls"] = [
                {"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in tool_calls
            ]
        memory.append(message)

        if not tool_calls:
            return

        transcript += strip_tool_calls(raw)
        for call in tool_calls:
            result = _execute_tool(call.name, call.arguments)
            memory.append({"role": "tool", "name": call.name, "content": result})
            transcript += f"\n\n🔧 `{call.name}({call.arguments})` → {result}\n\n"
        yield transcript

    transcript += "\n\n_(stopped: reached the maximum number of tool-calling steps)_"
    yield transcript
