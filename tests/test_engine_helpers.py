"""Pure helpers in agent.engine: schema conversion and tool-call parsing.

These need neither a model nor the network, so they make up the bulk of the
deterministic coverage for the engine.
"""
from agent.engine import ToolCall, parse_tool_calls, strip_tool_calls, to_openai_tool
from agent.tools import TOOL_SCHEMAS


def test_to_openai_tool_shape():
    oai = to_openai_tool(TOOL_SCHEMAS[0])
    assert oai["type"] == "function"
    assert oai["function"]["name"] == TOOL_SCHEMAS[0]["name"]
    assert oai["function"]["parameters"] == TOOL_SCHEMAS[0]["input_schema"]


def test_to_openai_tool_converts_every_tool():
    names = [to_openai_tool(t)["function"]["name"] for t in TOOL_SCHEMAS]
    assert names == ["read_file", "write_file", "run_shell", "calculate"]


def test_parse_single_call_surrounded_by_prose():
    text = 'Let me compute.\n<tool_call>\n{"name": "calculate", "arguments": {"expression": "47*19"}}\n</tool_call>'
    assert parse_tool_calls(text) == [ToolCall("calculate", {"expression": "47*19"})]


def test_parse_multiple_calls_and_stringified_arguments():
    text = (
        '<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>'
        '<tool_call>{"name":"b","arguments":"{\\"y\\":2}"}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["a", "b"]
    assert calls[0].arguments == {"x": 1}
    assert calls[1].arguments == {"y": 2}  # stringified JSON gets decoded


def test_parse_ignores_malformed_json():
    assert parse_tool_calls("<tool_call>{not valid json}</tool_call>") == []


def test_parse_returns_empty_without_tool_calls():
    assert parse_tool_calls("just a normal answer, no tools") == []


def test_strip_tool_calls_leaves_only_prose():
    text = 'Here you go.\n<tool_call>{"name":"x","arguments":{}}</tool_call>'
    assert strip_tool_calls(text) == "Here you go."


def test_peft_base_model_detection(tmp_path):
    from agent.engine import _peft_base_model

    assert _peft_base_model(str(tmp_path)) is None  # no adapter_config.json present
    (tmp_path / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"}'
    )
    assert _peft_base_model(str(tmp_path)) == "Qwen/Qwen2.5-7B-Instruct"
