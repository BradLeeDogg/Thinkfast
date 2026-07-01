"""Conversation memory: in-process by default, JSON-persisted when given a path."""
import json

from agent.memory import Memory


def test_in_memory_mode_does_not_persist():
    mem = Memory()
    mem.append({"role": "user", "content": "hi"})
    assert mem.path is None
    assert mem.messages == [{"role": "user", "content": "hi"}]


def test_persists_and_reloads_from_disk(tmp_path):
    path = tmp_path / "history.json"
    mem = Memory(str(path))
    mem.append({"role": "user", "content": "hi"})
    mem.append({"role": "assistant", "content": "hello"})

    reloaded = Memory(str(path))  # a fresh instance picks up the saved history
    assert reloaded.messages == mem.messages
    assert json.loads(path.read_text())[0]["content"] == "hi"
