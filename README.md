# agentic-ai

A minimal, generic agent loop that runs entirely on a **local, in-process
language model** — no API keys, no external services. It's a Claude-style
tool-use loop plus conversation memory, with an open-weights model (loaded via
Hugging Face Transformers) generating the responses, so the model is yours to
run, swap, and fine-tune.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

`pip install -e .` pulls `torch`, `transformers`, and `accelerate`. On Linux the
default `torch` wheel is built for CUDA; for a CPU-only install do this first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## Run

```bash
agent                                          # interactive chat loop
agent "what is 47 * 19? use the calculator"    # single instruction
```

The first run downloads the model weights (cached afterwards under
`~/.cache/huggingface`). A GPU is recommended; CPU works but is slow.

## Web chat UI

Prefer a browser over the terminal? Launch a local chat UI backed by the same
agent (tools and all):

```bash
pip install -e .[webui]
agent-web                       # serves http://127.0.0.1:7860
```

`agent-web --help` covers `--host`, `--port`, and `--share` (a temporary public
link). The model still runs locally, so pick `AGENT_MODEL` to match your
hardware (below).

## Choosing a model

The default is `Qwen/Qwen2.5-7B-Instruct` — a capable instruct model with
strong tool-calling. It realistically wants a GPU with ~16 GB of VRAM (in
bf16); on CPU it runs but is slow and memory-hungry.

Point `AGENT_MODEL` at any chat model whose tokenizer advertises tool support
through its chat template — scale up for more capability, or down for lighter
hardware:

```bash
export AGENT_MODEL=Qwen/Qwen2.5-32B-Instruct    # more capable (needs more VRAM)
export AGENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct   # lighter / runs almost anywhere
agent "list the files in the current directory"
```

Only a CUDA (NVIDIA) GPU is used for acceleration; with integrated graphics
(Intel/AMD) the model runs on the CPU, so prefer a 1.5B or 0.5B model there for
a responsive chat.

Tool-calling is tuned for Qwen2.5-style models, which emit
`<tool_call>{...}</tool_call>` blocks; other families with compatible chat
templates (e.g. Llama 3.1 Instruct) generally work too. Generation can also be
tuned with `AGENT_MAX_NEW_TOKENS` and `AGENT_TEMPERATURE`.

## Fine-tuning (make it yours)

Teach the model your own voice, facts, or task with LoRA — a small adapter
trained on top of the base model, no full retrain required. A GPU is required
for training.

```bash
pip install -e .[finetune]
agent-finetune --data examples/finetune_data.jsonl --output ./agent-lora
```

Training data is JSONL, one chat example per line in the agent's own message
format:

    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

Then point the agent at the adapter — `LocalEngine` detects it and loads it on
top of the base model automatically:

```bash
AGENT_MODEL=./agent-lora agent "who built you?"
```

`agent-finetune --help` lists the knobs (epochs, LoRA rank, learning rate, …);
pass `--merge` to also write a standalone merged model.

## How it's structured

- `agent/engine.py` — loads the model in-process and generates responses:
  converts the tool schemas into the chat-template format, streams tokens to
  stdout as they arrive, and parses tool calls back out of the generated text.
- `agent/loop.py` — the agentic loop itself: generate, run any tool calls the
  model makes, feed the results back, repeat until it stops calling tools
  (capped by `MAX_STEPS`).
- `agent/memory.py` — conversation history, kept in process memory and
  optionally persisted to a JSON file between runs.
- `agent/tools/` — tool definitions. Each tool is a small Python function plus
  a JSON schema describing it to the model. Add new tools here and register
  them in `agent/tools/__init__.py`.
- `agent/cli.py` — entry point for interactive and single-shot use.
- `agent/finetune.py` — optional LoRA fine-tuning (`agent-finetune`) that
  produces an adapter the engine loads back in.
- `agent/webui.py` — optional browser chat UI (`agent-web`) built on Gradio.

## Adding a tool

1. Write a function in `agent/tools/` that takes keyword arguments matching
   your JSON schema's `input_schema.properties` and returns a string result.
2. Add the JSON schema (name, description, input_schema) next to it.
3. Register both in `TOOLS` and `TOOL_SCHEMAS` in `agent/tools/__init__.py`.

The loop doesn't need to change — it dispatches on tool name generically.
