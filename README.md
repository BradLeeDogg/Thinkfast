# agentic-ai

A minimal, generic agent loop: an LLM plus tool use and conversation memory,
running entirely on your own machine — no API keys, no external services. Pick
the engine that suits your hardware:

- **Ollama backend** (recommended) — talks to a local
  [Ollama](https://ollama.com) server that runs compressed/quantized models
  quickly, even on a plain CPU. No torch required.
- **In-process backend** — loads the model inside Python with Hugging Face
  Transformers. A CUDA (NVIDIA) GPU helps a lot; on a CPU or integrated GPU it
  works but is slow.

Either way it's the same agent: a tool-calling loop (`read_file`, `write_file`,
`run_shell`, `calculate`), conversation memory, a CLI, and a web chat UI.

## Quick start (Ollama — fast)

1. Install Ollama from https://ollama.com and start it.
2. Then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ollama,webui]"
ollama pull qwen2.5:1.5b          # downloads a small, fast model

export AGENT_BACKEND=ollama
export AGENT_MODEL=qwen2.5:1.5b
agent-web                          # browser chat UI  (or:  agent  for the terminal)
```

**Windows:** just double-click `run_windows.bat` — it installs everything
(Ollama included) and launches the chat for you. See `START_HERE_Windows.txt`.

## In-process backend (Transformers)

```bash
pip install -e ".[local]"          # add ",webui" for the browser UI
agent "what is 47 * 19? use the calculator"
```

The default model is `Qwen/Qwen2.5-7B-Instruct`, which wants a GPU with ~16 GB
of VRAM. On a CPU or integrated GPU, use a smaller model (see below). The first
run downloads the weights to `~/.cache/huggingface`.

## Run

```bash
agent                                          # interactive chat loop
agent "list the files in the current directory"   # single instruction
agent-web                                      # browser chat UI at http://127.0.0.1:7860
```

## Choosing a model

Set `AGENT_MODEL`. The names depend on the backend:

- **Ollama:** `qwen2.5:0.5b`, `qwen2.5:1.5b`, `qwen2.5:7b`, `llama3.1:8b`, …
  (run `ollama pull <name>` first). Smaller = faster.
- **In-process:** Hugging Face ids like `Qwen/Qwen2.5-1.5B-Instruct`,
  `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`.

```bash
export AGENT_BACKEND=ollama
export AGENT_MODEL=qwen2.5:0.5b     # smaller/faster; bump to 7b for better answers
```

Tool-calling works with any tool-capable chat model (Qwen2.5, Llama 3.1, …).
Generation can also be tuned with `AGENT_MAX_NEW_TOKENS` and `AGENT_TEMPERATURE`.

## Web chat UI

```bash
pip install -e ".[ollama,webui]"   # or ".[local,webui]"
agent-web                          # serves http://127.0.0.1:7860
```

`agent-web --help` covers `--host`, `--port`, `--share`, and `--open`.

## Chat with your own books (retrieval)

Point the agent at a folder of PDFs and it becomes an open-book expert: it
searches the real text and answers grounded in it (with citations) instead of
guessing from memory. Uses Ollama for embeddings — no GPU.

```bash
pip install -e ".[ollama,webui,library]"
ollama pull nomic-embed-text
agent-index "/path/to/your/pdfs"     # one-time; slow for a big library
agent-web                            # now ask about your books
```

`agent-index` reads every PDF, splits it into passages, embeds them, and saves a
local index (default `./library_index`; override with `AGENT_LIBRARY_INDEX`).
Re-running skips books already indexed, so a long run resumes. Once indexed, the
`search_library` tool appears automatically, and you can query offline without
the source files. On Windows, run `index_library_windows.bat` once.

Small models don't always decide to search on their own; if it answers from
memory, ask it to "search my library for …", or use a larger `AGENT_MODEL`
(e.g. `qwen2.5:7b`) for more reliable tool use.

## Fine-tuning (make it yours)

Teach the in-process model your own voice or task with LoRA — a small adapter
trained on top of the base model. A GPU is required for training.

```bash
pip install -e ".[finetune]"
agent-finetune --data examples/finetune_data.jsonl --output ./agent-lora
AGENT_BACKEND=local AGENT_MODEL=./agent-lora agent "who built you?"
```

Training data is JSONL, one chat example per line in the agent's own message
format:

    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

`agent-finetune --help` lists the knobs; pass `--merge` for a standalone model.

## How it's structured

- `agent/engine.py` — the in-process (Transformers) engine, plus `build_engine`,
  which picks the backend from `AGENT_BACKEND`.
- `agent/ollama_engine.py` — the Ollama backend (talks to a local Ollama server
  over HTTP; no torch).
- `agent/loop.py` — the agentic loop: generate, run any tool calls, feed the
  results back, repeat until it stops (capped by `MAX_STEPS`). `run_turn_stream`
  is the streaming version used by the web UI.
- `agent/memory.py` — conversation history, optionally persisted to JSON.
- `agent/tools/` — tool definitions (a function + a JSON schema each). Add new
  tools here and register them in `agent/tools/__init__.py`.
- `agent/cli.py` — terminal entry point.
- `agent/webui.py` — browser chat UI (`agent-web`), built on Gradio.
- `agent/library.py` + `agent/index_cli.py` — the PDF library index
  (`agent-index`) and the search behind the `search_library` tool.
- `agent/finetune.py` — optional LoRA fine-tuning (`agent-finetune`).

## Adding a tool

1. Write a function in `agent/tools/` that takes keyword arguments matching your
   JSON schema's `input_schema.properties` and returns a string.
2. Add the JSON schema (name, description, input_schema) next to it.
3. Register both in `TOOLS` and `TOOL_SCHEMAS` in `agent/tools/__init__.py`.

The loop dispatches on tool name generically, so it doesn't need to change.
