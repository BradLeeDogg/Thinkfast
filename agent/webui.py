"""Browser chat UI for the agent, served locally with Gradio.

The same agent (model, tools, memory) as the CLI, in a web page instead of the
terminal. The model backend is chosen by ``build_engine`` (AGENT_BACKEND). Heavy
imports (gradio, plus the model via the engine) happen inside ``main`` so
importing this module — and ``agent-web --help`` — stays light.
"""
from __future__ import annotations

import argparse

from .engine import build_engine
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

    engine = build_engine()
    memory = Memory()  # one conversation for the life of the server

    import gradio as gr

    def respond(message: str, history):
        # `history` is Gradio's display log; the agent's own context lives in
        # `memory`, which run_turn_stream appends to as it goes.
        yield from run_turn_stream(engine, memory, message)

    gr.ChatInterface(
        respond,
        title="agentic-ai",
        description=f"Local agent · {engine.model_name}",
    ).launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=args.open)


if __name__ == "__main__":
    main()
