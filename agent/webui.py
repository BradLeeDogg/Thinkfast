"""Browser chat UI for the agent, served locally with Gradio.

The same agent (model, tools, memory) as the CLI, in a web page instead of the
terminal. Heavy imports (gradio, plus the model via the engine) happen inside
``main`` so importing this module — and ``agent-web --help`` — stays light.
"""
from __future__ import annotations

import argparse
import os

from .engine import DEFAULT_MODEL, LocalEngine
from .loop import run_turn_stream
from .memory import Memory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-web",
        description="Serve the agent as a local web chat UI.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on (default: 7860).")
    parser.add_argument("--share", action="store_true", help="Expose a temporary public Gradio link.")
    parser.add_argument("--open", action="store_true", help="Open the UI in your browser once it's ready.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    model_name = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
    print(f"Loading {model_name} … (the first run downloads the weights)")
    engine = LocalEngine(
        model_name,
        max_new_tokens=int(os.environ.get("AGENT_MAX_NEW_TOKENS", "1024")),
        temperature=float(os.environ.get("AGENT_TEMPERATURE", "0.7")),
    )
    memory = Memory()  # one conversation for the life of the server

    import gradio as gr

    def respond(message: str, history):
        # `history` is Gradio's display log; the agent's own context lives in
        # `memory`, which run_turn_stream appends to as it goes.
        yield from run_turn_stream(engine, memory, message)

    gr.ChatInterface(
        respond,
        title="agentic-ai",
        description=f"Local agent · {model_name}",
    ).launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=args.open)


if __name__ == "__main__":
    main()
