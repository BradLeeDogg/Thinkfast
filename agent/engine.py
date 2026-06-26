"""In-process language model engine.

Runs an open-weights chat model locally with Hugging Face Transformers, so the
agent needs no API key and no external service. Tool-calling is driven by the
model's own chat template (tuned for Qwen2.5-Instruct-style models, which emit
``<tool_call>{...}</tool_call>`` blocks) and parsed back out here.

torch/transformers are imported lazily inside ``LocalEngine`` so the rest of the
package — and the tool loop — can be imported and tested without them.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Matches the <tool_call>{json}</tool_call> blocks emitted by Qwen2.5-style models.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


def to_openai_tool(schema: dict) -> dict:
    """Convert an Anthropic-style tool schema (name/description/input_schema) into
    the OpenAI ``function`` format that Hugging Face chat templates expect."""
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Pull tool calls out of generated text, tolerating stray whitespace and any
    prose surrounding the ``<tool_call>`` blocks."""
    calls: list[ToolCall] = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get("name")
        if not name:
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):  # some models stringify the arguments object
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append(ToolCall(name=name, arguments=args if isinstance(args, dict) else {}))
    return calls


def strip_tool_calls(text: str) -> str:
    """Drop ``<tool_call>`` blocks from text, leaving any surrounding prose."""
    return _TOOL_CALL_RE.sub("", text).strip()


class LocalEngine:
    """Loads a chat model into the current process and generates responses."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> None:
        # Imported here, not at module top, so importing the agent doesn't drag in
        # torch/transformers (heavy, and not needed to test the loop with a stub).
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # `torch_dtype` was renamed to `dtype` in transformers 5.x; pick the keyword
        # the installed version expects so "auto" precision works warning-free on both.
        dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            **{dtype_kw: "auto"},
        )

    def generate(self, messages: list[dict], tools: list[dict]) -> tuple[str, list[ToolCall]]:
        """Generate the assistant's next message, streaming text to stdout as it
        arrives, and return ``(full_text, parsed_tool_calls)``."""
        from transformers import TextIteratorStreamer

        oai_tools = [to_openai_tool(t) for t in tools]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=oai_tools,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs: dict[str, Any] = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=0.8, top_k=20)
        else:
            gen_kwargs.update(do_sample=False)

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        pieces: list[str] = []
        for piece in streamer:
            print(piece, end="", flush=True)
            pieces.append(piece)
        thread.join()
        print()

        text = "".join(pieces)
        return text, parse_tool_calls(text)
