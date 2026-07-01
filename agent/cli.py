import os
import sys

from .engine import DEFAULT_MODEL, LocalEngine
from .loop import run_turn
from .memory import Memory

HISTORY_PATH = ".agent_history.json"


def main() -> None:
    model_name = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
    print(f"Loading {model_name} … (the first run downloads the weights)", file=sys.stderr)
    try:
        engine = LocalEngine(
            model_name,
            max_new_tokens=int(os.environ.get("AGENT_MAX_NEW_TOKENS", "1024")),
            temperature=float(os.environ.get("AGENT_TEMPERATURE", "0.7")),
        )
    except Exception as exc:
        print(f"Could not load model '{model_name}': {exc}", file=sys.stderr)
        sys.exit(1)

    memory = Memory(HISTORY_PATH)

    if len(sys.argv) > 1:
        _run_safely(engine, memory, " ".join(sys.argv[1:]))
        return

    print("agentic-ai — local agent loop. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        _run_safely(engine, memory, user_input)


def _run_safely(engine: LocalEngine, memory: Memory, instruction: str) -> None:
    try:
        run_turn(engine, memory, instruction)
    except Exception as exc:  # keep an interactive session alive across generation errors
        print(f"\nGeneration error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
