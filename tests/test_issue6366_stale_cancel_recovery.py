"""Tests for #6366: stale-cancel recovery must not append duplicate
rows when the durable transcript has already advanced past the
pending turn.

Bug shape: the recovery path at ``api/models.py::_apply_core_sync_or_error_marker``
checks only ``session.messages[-1]`` against the pending user
checkpoint. When a stale pending turn's user row is older than the
transcript tail, the tail is a *newer* settled assistant row that
can never match the pending **user** checkpoint, and the recovery
path falls through into the append branch. It then appends a
recovered user row, a ``_partial`` clone of the old journal, and a
generic ``No response from provider`` error — all after the valid
final answer. The duplicates survive reload.

Fix: add a transcript-advance detector that scans the durable
messages for a matching user row at any index and, when found,
checks for a later settled assistant row. If one exists, the
recovery path must clear the stale pending fields without
appending any rows. The helper is a pure read so recovery is
idempotent: running it twice produces the same outcome.
"""

import pytest

import api.models as models
from api.models import (
    _transcript_already_advanced_past_pending,
)


class _FakeSession:
    """Minimal stand-in for api.models.Session — only the fields the
    new helper reads."""

    def __init__(self, pending=None, messages=None):
        self.pending_user_message = pending
        self.messages = list(messages or [])


# ── _transcript_already_advanced_past_pending unit cases ───────────────


def test_advances_past_pending_when_user_matched_and_settled_assistant_follows():
    """Canonical bug shape: a stale pending turn's user row sits at
    an older index, and a newer settled assistant row follows.
    Recovery must not append duplicate rows after the assistant.
    Strict match: the existing user row must match the pending
    checkpoint (content + timestamp + source + attachments) so
    that a *new* user turn repeating the same text is not
    misclassified as a stale advance."""
    s = _FakeSession(
        pending="hello world",
        messages=[
            {"role": "user", "content": "hello world", "timestamp": 222, "_source": "webui", "attachments": [{"name": "a.png"}]},
            {"role": "assistant", "content": "answer", "timestamp": 223},
        ],
    )
    # Plumb the full checkpoint identity so the strict match hits.
    s.pending_started_at = 222
    s.pending_user_source = "webui"
    s.pending_attachments = [{"name": "a.png"}]
    assert _transcript_already_advanced_past_pending(s) is True


def test_advances_past_pending_ignores_partial_clone_after_user():
    """A ``_partial`` clone after the matching user row does NOT count
    as a settled advance — the canonical bug shape is the partial
    clone followed by a generic no-response error, both of which
    the previous recovery path appended. Only a non-partial,
    non-error assistant row represents a real advance."""
    s = _FakeSession(
        pending="hello world",
        messages=[
            {"role": "user", "content": "hello world", "timestamp": 1},
            {"role": "assistant", "content": "answer", "timestamp": 2, "_partial": True},
            {"role": "assistant", "content": "no response", "timestamp": 3, "_error": True},
        ],
    )
    assert _transcript_already_advanced_past_pending(s) is False


def test_advances_past_pending_ignores_recovered_assistant_row():
    """A previously-recovered assistant row is not a settled answer
    either; the recovery path may have produced it on an earlier
    pass. Skip it and keep looking forward."""
    s = _FakeSession(
        pending="hello world",
        messages=[
            {"role": "user", "content": "hello world", "timestamp": 1},
            {"role": "assistant", "content": "answer", "timestamp": 2, "_recovered": True},
        ],
    )
    assert _transcript_already_advanced_past_pending(s) is False


def test_does_not_advance_when_only_user_row_matches():
    """The matching user row exists but no later assistant row has
    been settled. Recovery is the legitimate path; the helper
    returns False."""
    s = _FakeSession(
        pending="hello world",
        messages=[
            {"role": "user", "content": "hello world", "timestamp": 1},
        ],
    )
    assert _transcript_already_advanced_past_pending(s) is False


def test_does_not_advance_when_no_user_row_matches():
    """No transcript row carries the pending text. The bug shape
    here is the older non-matching rows + a stale pending. The
    recovery path is legitimate; the helper returns False."""
    s = _FakeSession(
        pending="hello world",
        messages=[
            {"role": "user", "content": "other text", "timestamp": 1},
            {"role": "assistant", "content": "answer", "timestamp": 2},
        ],
    )
    assert _transcript_already_advanced_past_pending(s) is False


def test_does_not_advance_when_no_pending_text():
    """An empty pending text means there is nothing to compare
    against. The helper returns False without scanning the
    transcript."""
    s = _FakeSession(
        pending=None,
        messages=[
            {"role": "assistant", "content": "answer", "timestamp": 2},
        ],
    )
    assert _transcript_already_advanced_past_pending(s) is False


def test_does_not_advance_when_no_messages():
    """An empty transcript is the cold-start case; recovery is the
    legitimate path. The helper returns False."""
    s = _FakeSession(pending="hello world", messages=[])
    assert _transcript_already_advanced_past_pending(s) is False


def test_advances_past_pending_with_unrelated_intervening_rows():
    """A matching user row followed by an unrelated tool row and
    THEN a settled assistant row counts as a real advance. The
    helper scans past non-assistant roles correctly. Strict
    match on the user row."""
    s = _FakeSession(
        pending="hello world",
        messages=[
            {"role": "user", "content": "hello world", "timestamp": 222, "_source": "webui", "attachments": []},
            {"role": "tool", "content": "tool output"},
            {"role": "assistant", "content": "answer", "timestamp": 223},
        ],
    )
    s.pending_started_at = 222
    s.pending_user_source = "webui"
    s.pending_attachments = []
    assert _transcript_already_advanced_past_pending(s) is True


def test_helper_is_pure_read_idempotent():
    """Idempotence: running the helper twice on the same session
    state returns the same result. The helper does not mutate the
    session, so recovery is automatically safe to run twice."""
    s = _FakeSession(
        pending="hello world",
        messages=[
            {"role": "user", "content": "hello world", "timestamp": 222, "_source": "webui", "attachments": []},
            {"role": "assistant", "content": "answer", "timestamp": 223},
        ],
    )
    s.pending_started_at = 222
    s.pending_user_source = "webui"
    s.pending_attachments = []
    first = _transcript_already_advanced_past_pending(s)
    second = _transcript_already_advanced_past_pending(s)
    assert first == second is True
    # And the session state was not mutated.
    assert s.pending_user_message == "hello world"
    assert len(s.messages) == 2



def test_does_not_advance_when_repeated_prompt_has_different_checkpoint():
    """#6366 regression guard for repeated-prompt identity: a *new*
    user turn that happens to repeat the same prompt text as an
    older transcript row is NOT a stale advance. The strict match
    uses timestamp + source + attachments to tell a stale duplicate
    from a legitimate re-send."""
    s = _FakeSession(
        pending="repeat this",
        messages=[
            # Older user row, same text but different identity.
            {
                "role": "user",
                "content": "repeat this",
                "timestamp": 111,
                "_source": "webui",
                "attachments": [{"name": "old.png"}],
            },
            {"role": "assistant", "content": "Earlier answer"},
        ],
    )
    # Pending checkpoint is a fresh turn: different timestamp and
    # different attachments.
    s.pending_started_at = 222
    s.pending_user_source = "webui"
    s.pending_attachments = [{"name": "current.png"}]
    # Strict match fails (timestamp + attachments differ), so the
    # helper returns False and recovery is the legitimate path.
    assert _transcript_already_advanced_past_pending(s) is False


def test_advances_past_pending_with_exact_checkpoint_match():
    """The strict-match path: a transcript user row that exactly
    matches the pending checkpoint (content + timestamp +
    source + attachments) followed by a settled assistant row is
    the canonical #6366 bug shape and must suppress the recovery
    append."""
    s = _FakeSession(
        pending="hello world",
        messages=[
            {
                "role": "user",
                "content": "hello world",
                "timestamp": 222,
                "_source": "webui",
                "attachments": [{"name": "a.png"}],
            },
            {"role": "assistant", "content": "answer", "timestamp": 223},
        ],
    )
    s.pending_started_at = 222
    s.pending_user_source = "webui"
    s.pending_attachments = [{"name": "a.png"}]
    assert _transcript_already_advanced_past_pending(s) is True
