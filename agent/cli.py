import sys

from .engine import build_engine
from .loop import run_turn
from .memory import Memory

HISTORY_PATH = ".agent_history.json"


def main() -> None:
    try:
        engine = build_engine()
    except Exception as exc:
        print(f"Could not start the model: {exc}", file=sys.stderr)
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


def _run_safely(engine, memory: Memory, instruction: str) -> None:
    try:
        run_turn(engine, memory, instruction)
    except Exception as exc:  # keep an interactive session alive across generation errors
        print(f"\nGeneration error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
