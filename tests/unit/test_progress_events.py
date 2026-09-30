"""Every node emits a PROGRESS_UPDATE on entry (and never a terminal event)."""

import pytest

from shared.services.events import EventEmitter, bound_emitter
from shared.services.events.types import EventType
from src.nodes.main_node import MainNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode


class RecordingEmitter(EventEmitter):
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit_event(self, event_type, message="", metadata=None, **kwargs):
        self.events.append({"event_type": event_type, "message": message, "metadata": metadata or {}})


@pytest.mark.parametrize(
    ("node_cls", "stage"),
    [(PreProcessNode, "pre_process"), (MainNode, "main"), (PostProcessNode, "post_process")],
)
def test_node_emits_progress_update_on_entry(base_agent_state, node_cls, stage):
    state = {**base_agent_state, "rendered_prompt": "p", "schema_id": "id", "result": {"answer": "x"}}
    recorder = RecordingEmitter()
    with bound_emitter(recorder):
        node_cls().execute(state)
    assert recorder.events, f"{node_cls.__name__} emitted nothing"
    first = recorder.events[0]
    assert first["event_type"] == EventType.PROGRESS_UPDATE
    assert first["metadata"]["stage"] == stage
    assert all(e["event_type"] == EventType.PROGRESS_UPDATE for e in recorder.events)
