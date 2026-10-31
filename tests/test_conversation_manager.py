"""
Unit tests for the Conversation Manager that don't require a running Ollama
server -- they test stage transitions, off-topic detection, and memory
windowing logic in isolation using a fake LLM engine.
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.conversation_manager import ConversationManager, Turn, is_off_topic
from app.domain import (
    STAGE_CLOSING,
    STAGE_CONFIRMATION,
    STAGE_GREETING,
    STAGE_INFO_GATHERING,
    STAGE_INTENT_ID,
    STAGE_RESOLUTION,
)


class FakeEngine:
    """Deterministic stand-in for OllamaEngine so tests don't need a real model."""

    def __init__(self, response="Sure, happy to help with that."):
        self.response = response
        self.calls = []

    async def stream_chat(self, messages, options=None):
        self.calls.append(messages)
        for word in self.response.split(" "):
            yield word + " ", FakeMetrics()

    async def chat_once(self, messages, options=None):
        self.calls.append(messages)
        return "Customer asked about an order; assistant explained policy."


class FakeMetrics:
    time_to_first_token = 0.05
    total_time = 0.4
    output_tokens_estimate = 5

    @property
    def tokens_per_second(self):
        return 12.5


@pytest.fixture
def manager():
    return ConversationManager(engine=FakeEngine())


def run(coro):
    return asyncio.run(coro)


async def _collect(manager, session, text):
    events = []
    async for ev in manager.handle_user_message(session, text):
        events.append(ev)
    return events


def test_off_topic_detection():
    assert is_off_topic("What's the weather like today?") is True
    assert is_off_topic("Can you write me a poem?") is True
    assert is_off_topic("Where is my order BC-1234?") is False
    assert is_off_topic("Does this blender come in red?") is False


def test_off_topic_short_circuits_without_llm_call(manager):
    session = manager.create_session()
    events = run(_collect(manager, session, "What's the weather in Paris?"))
    types = [e[0] for e in events]
    assert "blocked" in types
    assert "token" not in types
    assert manager.engine.calls == []  # no LLM call made for off-topic


def test_greeting_to_info_gathering_on_clear_intent(manager):
    session = manager.create_session()
    assert session.stage == STAGE_GREETING
    run(_collect(manager, session, "Where is my order?"))
    assert session.stage == STAGE_INFO_GATHERING
    assert session.active_intent == "order_tracking"


def test_greeting_to_intent_id_on_vague_message(manager):
    session = manager.create_session()
    run(_collect(manager, session, "Hi, I need some help."))
    assert session.stage == STAGE_INTENT_ID


def test_topic_switch_mid_conversation_resets_to_info_gathering(manager):
    session = manager.create_session()
    run(_collect(manager, session, "Where is my order?"))
    assert session.active_intent == "order_tracking"
    run(_collect(manager, session, "I want to talk about it"))  # -> resolution
    assert session.stage == STAGE_RESOLUTION

    # Now the user pivots to a completely different supported topic.
    run(_collect(manager, session, "Actually, can I return a blender?"))
    assert session.active_intent == "returns_shipping"
    assert session.stage == STAGE_INFO_GATHERING
    # History from the earlier topic should still be present (context preserved).
    assert any("order" in t.content.lower() for t in session.history)


def test_confirmation_negative_moves_to_closing(manager):
    session = manager.create_session()
    session.stage = STAGE_CONFIRMATION
    session.active_intent = "order_tracking"
    run(_collect(manager, session, "no that's all"))
    assert session.stage == STAGE_CLOSING


def test_confirmation_positive_stays_in_resolution(manager):
    session = manager.create_session()
    session.stage = STAGE_CONFIRMATION
    session.active_intent = "order_tracking"
    run(_collect(manager, session, "yes please explain more"))
    assert session.stage == STAGE_RESOLUTION


def test_history_recorded_for_both_roles(manager):
    session = manager.create_session()
    run(_collect(manager, session, "Where is my order?"))
    roles = [t.role for t in session.history]
    assert roles == ["user", "assistant"]


def test_summarization_triggers_after_window_exceeded(manager):
    session = manager.create_session()
    # Manually seed a long history to force the summarizer to kick in.
    for i in range(10):
        session.history.append(Turn("user", f"message {i} about my order"))
        session.history.append(Turn("assistant", f"reply {i}"))

    run(manager._maybe_summarize(session))
    assert session.rolling_summary != ""
    # History should have been trimmed down to the window size.
    assert len(session.history) <= 12  # MAX_TURNS_IN_WINDOW*2 in default config


def test_reset_session_clears_state(manager):
    session = manager.create_session()
    run(_collect(manager, session, "Where is my order?"))
    assert len(session.history) > 0
    new_session = manager.reset_session(session.session_id)
    assert new_session.history == []
    assert new_session.stage == STAGE_GREETING
