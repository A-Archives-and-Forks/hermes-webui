"""Native-image turns keep model context private from the visible transcript."""

import json
import sqlite3
from types import SimpleNamespace

from api.helpers import public_session_projection
from api.models import get_state_db_session_messages
from api.process_event_utils import build_active_turn_token
from api.streaming import (
    _materialize_active_turn_user,
    _new_turn_context_from_messages,
    _active_turn_authority,
    _find_active_turn_checkpoint_index,
    _sanitize_messages_for_agent,
    _settle_result_messages,
)


IMAGE_A = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
IMAGE_B = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNg+M8AAAICAQB7CYF4AAAAAElFTkSuQmCC"
RECALL_NOTE = "Recall note: the sample is neutral."
PLUGIN_NOTE = "Pre-call note: summarize visible details only."


def _native_user_content(text, image_url, extra_text=()):
    parts = [{"type": "text", "text": f"[Workspace::v1: /fixture]\n{text}"}]
    if image_url:
        parts.append({"type": "image_url", "image_url": {"url": image_url}})
    parts.extend({"type": "text", "text": value} for value in extra_text)
    return parts


def _durable_agent_content(content):
    parts = []
    for part in content:
        parts.append("[screenshot]" if part.get("type") == "image_url" else part["text"])
    return " ".join("\n".join(parts).split())


def _settle_image_turn(
    *,
    session_id="native-image-session",
    text="Describe this image",
    image_url=IMAGE_A,
    timestamp=100.0,
    extra_text=(),
    queued_notifications=(),
    agent_row_id=None,
    empty_display_before_settle=False,
    previous_messages=(),
    previous_context=(),
    attachment_name="sample.png",
):
    stream_id = f"native-image-stream-{timestamp}"
    token = build_active_turn_token(stream_id, timestamp)
    attachments = (
        [{"name": attachment_name, "mime": "image/png", "is_image": True}]
        if attachment_name
        else []
    )
    checkpoint = ({
        "role": "user",
        "content": text,
        "timestamp": timestamp,
        "attachments": attachments,
        "_active_turn_token": token,
    } if text else None)
    session = SimpleNamespace(
        session_id=session_id,
        messages=[*previous_messages, *([checkpoint] if checkpoint else [])],
        context_messages=list(previous_context),
        pending_user_message=text,
        pending_attachments=attachments,
        pending_started_at=timestamp,
        pending_user_source="webui",
    )
    identity = _active_turn_authority(session, stream_id, text)
    agent_input_text = (
        "\n\n".join([*queued_notifications, text]).strip()
        if queued_notifications
        else text
    )
    identity["trusted_agent_input_text"] = agent_input_text
    identity.update({
        "current_turn_user_idx": len(previous_context),
        "turn_id": f"agent-turn-{timestamp}",
        "agent_turn_boundary_resolved": True,
    })
    rich_content = _native_user_content(agent_input_text, image_url, extra_text)
    api_content = json.dumps(rich_content)
    current_user = {
        "role": "user",
        "content": rich_content,
        "timestamp": timestamp,
        "api_content": api_content,
    }
    if agent_row_id is not None:
        current_user["_row_id"] = agent_row_id
    result_messages = [
        *previous_context,
        current_user,
        {"role": "assistant", "content": "A neutral sample image."},
    ]
    if empty_display_before_settle:
        session.messages = []
        assert not any(
            message.get("_active_turn_token") == identity["token"]
            for message in session.messages
        )
    _settle_result_messages(
        session,
        session.messages,
        previous_context,
        result_messages,
        text,
        "webui",
        identity,
    )
    return session, identity, api_content


def test_trusted_notification_prefix_keeps_image_turn_clean_after_reload(
    monkeypatch, tmp_path
):
    import api.models as models

    text = "Describe this image"
    notification = "Queued process update: sample job completed."
    session, identity, _ = _settle_image_turn(
        text=text,
        queued_notifications=(notification,),
        extra_text=(RECALL_NOTE, PLUGIN_NOTE),
        timestamp=900.0,
    )
    expected_context = _native_user_content(
        f"{notification}\n\n{text}", IMAGE_A, (RECALL_NOTE, PLUGIN_NOTE)
    )
    display_users = [
        message for message in session.messages if message.get("role") == "user"
    ]
    assert len(display_users) == 1
    assert display_users[0]["content"] == text
    assert display_users[0]["attachments"] == [
        {"name": "sample.png", "mime": "image/png", "is_image": True}
    ]
    context_users = [
        message for message in session.context_messages
        if message.get("role") == "user"
    ]
    assert len(context_users) == 1
    assert context_users[0]["content"] == expected_context

    session_dir = tmp_path / "webui-sessions"
    session_dir.mkdir()
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    models.Session(
        session_id=session.session_id,
        workspace="/fixture",
        model="fixture-model",
        context_length=128_000,
        messages=session.messages,
        context_messages=session.context_messages,
    ).save(skip_index=True)
    reloaded = models.Session.load(session.session_id)
    assert reloaded is not None
    reloaded_users = [
        message for message in reloaded.messages if message.get("role") == "user"
    ]
    assert len(reloaded_users) == 1
    assert reloaded_users[0]["content"] == text
    assert reloaded_users[0]["attachments"] == display_users[0]["attachments"]
    reloaded_context_users = [
        message for message in reloaded.context_messages
        if message.get("role") == "user"
    ]
    assert len(reloaded_context_users) == 1
    assert reloaded_context_users[0]["content"] == expected_context

    identity["current_turn_user_idx"] = 0
    untrusted = [{
        "role": "user",
        "content": _native_user_content(
            f"Untrusted text prefix\n\n{text}", IMAGE_A, (RECALL_NOTE, PLUGIN_NOTE)
        ),
    }]
    assert _find_active_turn_checkpoint_index(
        untrusted, [], identity, text,
    ) is None


def _write_state_db(path, session_id, rows):
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE messages ("
            "id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, "
            "timestamp REAL, active INTEGER DEFAULT 1, api_content TEXT)"
        )
        conn.executemany(
            "INSERT INTO messages (session_id, role, content, timestamp, api_content) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (session_id, "user", content, timestamp, api_content)
                for content, timestamp, api_content in rows
            ],
        )


def test_settlement_reload_and_next_turn_keep_one_clean_bubble_and_rich_context(
    monkeypatch, tmp_path
):
    import api.config
    import api.models as models
    from api import routes

    text = "Describe this image"
    notification = "Queued process update: sample job completed."
    session, identity, api_content = _settle_image_turn(
        text=text,
        queued_notifications=(notification,),
        extra_text=(RECALL_NOTE, PLUGIN_NOTE),
        agent_row_id=1,
        empty_display_before_settle=True,
    )
    token = identity["token"]
    session_dir = tmp_path / "webui-sessions"
    session_dir.mkdir()
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    persisted = models.Session(
        session_id=session.session_id,
        workspace="/fixture",
        model="fixture-model",
        context_length=128_000,
        messages=session.messages,
        context_messages=session.context_messages,
    )
    persisted.save(skip_index=True)
    session = models.Session.load(session.session_id)
    assert session is not None

    current_rows = [
        message for message in session.messages
        if isinstance(message, dict) and message.get("_active_turn_token") == token
    ]
    assert len(current_rows) == 1
    assert current_rows[0]["content"] == text
    assert "_webui_display_content" not in current_rows[0]
    assert "api_content" not in current_rows[0]
    assert current_rows[0]["attachments"][0]["name"] == "sample.png"
    assert current_rows[0]["_row_id"] == 1
    settled_current = [
        message for message in session.messages
        if message.get("role") == "user" and message.get("timestamp") == 100.0
    ]
    assert len(settled_current) == 1
    assert settled_current[0]["content"] == text

    context_user = next(
        message for message in session.context_messages
        if message.get("_active_turn_token") == token
    )
    expected_context = _native_user_content(
        f"{notification}\n\n{text}", IMAGE_A, (RECALL_NOTE, PLUGIN_NOTE)
    )
    assert context_user["content"] == expected_context
    assert context_user["api_content"] == api_content
    assert context_user["_row_id"] == 1
    assert context_user["_webui_trusted_agent_input_text"] == f"{notification}\n\n{text}"
    assert "_webui_trusted_agent_input_text" not in current_rows[0]

    db_path = tmp_path / "state.db"
    _write_state_db(
        db_path,
        session.session_id,
        [(_durable_agent_content(context_user["content"]), 100.0, None)],
    )
    monkeypatch.setattr(models, "_active_state_db_path", lambda: db_path)
    state_rows = get_state_db_session_messages(session.session_id)
    assert len(state_rows) == 1
    assert state_rows[0].get("api_content") is None
    assert state_rows[0]["_state_db_row_id"] == 1
    display_rows = models.reconciled_state_db_messages_for_session(
        session,
        state_messages=state_rows,
    )
    current_display_rows = [
        message for message in display_rows
        if message.get("_active_turn_token") == token
    ]
    assert len(current_display_rows) == 1
    reconciled_current = [
        message for message in display_rows
        if message.get("role") == "user" and message.get("timestamp") == 100.0
    ]
    assert len(reconciled_current) == 1
    assert reconciled_current[0]["content"] == text
    assert reconciled_current[0]["attachments"][0]["name"] == "sample.png"

    monkeypatch.setattr(api.config, "load_settings", lambda: {"api_redact_enabled": False})
    public = public_session_projection({
        "session_id": session.session_id,
        "messages": display_rows,
    })
    public_current = [
        message for message in public["messages"]
        if message.get("role") == "user" and message.get("timestamp") == 100.0
    ]
    assert len(public_current) == 1
    assert public_current[0]["content"] == text
    assert public_current[0]["attachments"][0]["name"] == "sample.png"
    assert not any(
        key in public_current[0]
        for key in ("_webui_display_content", "_active_turn_token", "api_content")
    )

    route_response = {}
    monkeypatch.setattr(
        routes,
        "get_session",
        lambda sid, metadata_only=False: models.Session.load(sid),
    )
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    monkeypatch.setattr(routes, "_active_stream_ids", lambda: set())
    monkeypatch.setattr(
        routes,
        "j",
        lambda _handler, payload, status=200, **_kwargs: (
            route_response.update(payload=payload, status=status) or payload
        ),
    )
    routes._handle_session_get(
        None,
        SimpleNamespace(
            path="/api/session",
            query=f"session_id={session.session_id}&resolve_model=0",
        ),
    )
    assert route_response["status"] == 200
    route_messages = route_response["payload"]["session"]["messages"]
    route_current = [
        message for message in route_messages
        if message.get("role") == "user" and message.get("timestamp") == 100.0
    ]
    assert len(route_current) == 1
    assert route_current[0]["content"] == text
    assert route_current[0]["attachments"][0]["name"] == "sample.png"
    assert not any(
        key in route_current[0]
        for key in ("_webui_display_content", "_active_turn_token", "api_content")
    )
    projected_context = public_session_projection({
        "context_messages": session.context_messages,
    })["context_messages"]
    assert not any(
        "_webui_trusted_agent_input_text" in message
        for message in projected_context
    )

    recovered_context = models.reconciled_state_db_messages_for_session(
        session,
        prefer_context=True,
        state_messages=state_rows,
    )
    replay = _sanitize_messages_for_agent(recovered_context)
    replay_users = [message for message in replay if message.get("role") == "user"]
    assert len(replay_users) == 1
    assert replay_users[0]["content"] == expected_context
    assert replay_users[0]["api_content"] == api_content

    second_text = "What object is shown?"
    next_turn_history = _new_turn_context_from_messages(
        recovered_context,
        second_text,
    )
    model_history = _sanitize_messages_for_agent(next_turn_history)
    model_history_user = next(
        message for message in model_history if message.get("role") == "user"
    )
    assert model_history_user["content"] == expected_context
    assert model_history_user["api_content"] == api_content

    second_stream_id = "native-image-second-stream"
    session.active_stream_id = second_stream_id
    session.pending_user_message = second_text
    session.pending_attachments = []
    session.pending_started_at = 200.0
    session.pending_user_source = "webui"
    second_identity = _active_turn_authority(session, second_stream_id, second_text)
    second_identity.update({
        "current_turn_user_idx": len(model_history),
        "turn_id": "agent-turn-second",
        "agent_turn_boundary_resolved": True,
    })
    checkpoint = _materialize_active_turn_user(second_identity, second_text, "webui")
    second_identity["checkpoint"] = checkpoint
    previous_display = [*display_rows, checkpoint]
    session.messages = previous_display
    result_messages = [
        *model_history,
        {
            "role": "user",
            "content": f"[Workspace::v1: /fixture]\n{second_text}",
            "timestamp": 200.0,
        },
        {"role": "assistant", "content": "A neutral follow-up.", "timestamp": 200.1},
    ]
    state_rows_before_second_turn = list(state_rows)
    _settle_result_messages(
        session,
        previous_display,
        recovered_context,
        result_messages,
        second_text,
        "webui",
        second_identity,
    )

    settled_users = [
        message for message in session.messages if message.get("role") == "user"
    ]
    assert [message["content"] for message in settled_users] == [text, second_text]
    assert settled_users[0]["attachments"][0]["name"] == "sample.png"
    rich_after_second_turn = next(
        message for message in session.context_messages
        if message.get("_active_turn_token") == token
    )
    assert rich_after_second_turn["content"] == expected_context
    assert rich_after_second_turn["_webui_trusted_agent_input_text"] == (
        f"{notification}\n\n{text}"
    )
    assert get_state_db_session_messages(session.session_id) == state_rows_before_second_turn

    session.active_stream_id = None
    session.pending_user_message = None
    session.pending_attachments = []
    session.pending_started_at = None
    session.pending_user_source = None
    session.save(skip_index=True)
    session = models.Session.load(session.session_id)
    assert session is not None
    reloaded_users = [
        message for message in session.messages if message.get("role") == "user"
    ]
    assert [message["content"] for message in reloaded_users] == [text, second_text]
    assert reloaded_users[0]["attachments"][0]["name"] == "sample.png"
    reloaded_rich_user = next(
        message for message in session.context_messages
        if message.get("_active_turn_token") == token
    )
    assert reloaded_rich_user["content"] == expected_context
    assert reloaded_rich_user["_webui_trusted_agent_input_text"] == (
        f"{notification}\n\n{text}"
    )

    route_response.clear()
    routes._handle_session_get(
        None,
        SimpleNamespace(
            path="/api/session",
            query=f"session_id={session.session_id}&resolve_model=0",
        ),
    )
    api_users = [
        message for message in route_response["payload"]["session"]["messages"]
        if message.get("role") == "user"
    ]
    assert [message["content"] for message in api_users] == [text, second_text]
    assert api_users[0]["attachments"][0]["name"] == "sample.png"


def test_image_only_and_no_context_turns_keep_the_submitted_display_text():
    image_only, image_only_identity, _ = _settle_image_turn(
        text="",
        image_url=IMAGE_A,
        timestamp=200.0,
        attachment_name="image-only.png",
    )
    image_only_row = next(
        message for message in image_only.messages
        if message.get("_active_turn_token") == image_only_identity["token"]
    )
    assert image_only_row["content"] == ""
    assert "_webui_display_content" not in image_only_row
    assert "api_content" not in image_only_row
    assert image_only_row["attachments"][0]["name"] == "image-only.png"

    no_context, no_context_identity, _ = _settle_image_turn(
        text="What is in this image?",
        image_url=IMAGE_B,
        timestamp=300.0,
    )
    no_context_row = next(
        message for message in no_context.messages
        if message.get("_active_turn_token") == no_context_identity["token"]
    )
    assert no_context_row["content"] == "What is in this image?"
    assert "_webui_display_content" not in no_context_row

    no_prefetch, no_prefetch_identity, _ = _settle_image_turn(
        text="This image was not prefetched.",
        image_url=IMAGE_A,
        timestamp=350.0,
        attachment_name=None,
    )
    no_prefetch_row = next(
        message for message in no_prefetch.messages
        if message.get("_active_turn_token") == no_prefetch_identity["token"]
    )
    assert no_prefetch_row["content"] == "This image was not prefetched."
    assert no_prefetch_row.get("attachments") in (None, [])


def test_repeated_literal_marker_prompt_keeps_distinct_image_turns(monkeypatch):
    import api.config
    import api.models as models

    monkeypatch.setattr(api.config, "load_settings", lambda: {"api_redact_enabled": False})
    prompt = "Please keep <memory-context>this literal text</memory-context>."
    first, first_identity, _ = _settle_image_turn(
        text=prompt,
        image_url=IMAGE_A,
        timestamp=400.0,
        attachment_name="first.png",
    )
    second, second_identity, _ = _settle_image_turn(
        text=prompt,
        image_url=IMAGE_B,
        timestamp=500.0,
        previous_messages=first.messages,
        previous_context=first.context_messages,
        attachment_name="second.png",
    )
    tokens = {first_identity["token"], second_identity["token"]}
    for row_id, token in enumerate((first_identity["token"], second_identity["token"]), 1):
        for rows in (second.messages, second.context_messages):
            next(message for message in rows if message.get("_active_turn_token") == token)[
                "_row_id"
            ] = row_id
    settled_turns = [
        message for message in second.messages
        if message.get("role") == "user" and message.get("timestamp") in (400.0, 500.0)
    ]
    assert len(settled_turns) == 2
    assert [message["content"] for message in settled_turns] == [prompt, prompt]
    assert {message.get("_active_turn_token") for message in settled_turns} == tokens
    assert [message["attachments"][0]["name"] for message in settled_turns] == [
        "first.png", "second.png",
    ]

    state_rows = []
    for message in second.context_messages:
        if message.get("_active_turn_token") in tokens:
            state_rows.append({
                "role": "user",
                "content": _durable_agent_content(message["content"]),
                "timestamp": message["timestamp"],
                "api_content": message["api_content"],
                "_state_db_row_id": len(state_rows) + 1,
            })
    display_rows = models.reconciled_state_db_messages_for_session(
        second,
        state_messages=state_rows,
    )
    displayed_turns = [
        message for message in display_rows
        if message.get("role") == "user" and message.get("timestamp") in (400.0, 500.0)
    ]
    assert len(displayed_turns) == 2
    assert [message["content"] for message in displayed_turns] == [prompt, prompt]
    assert [message["attachments"][0]["name"] for message in displayed_turns] == [
        "first.png", "second.png",
    ]

    public = public_session_projection({
        "session_id": second.session_id,
        "messages": display_rows,
    })
    public_turns = [
        message for message in public["messages"]
        if message.get("role") == "user" and message.get("timestamp") in (400.0, 500.0)
    ]
    assert len(public_turns) == 2
    assert [message["content"] for message in public_turns] == [prompt, prompt]
    assert [message["attachments"][0]["name"] for message in public_turns] == [
        "first.png", "second.png",
    ]


def test_state_db_mirror_with_conflicting_identity_is_not_suppressed():
    import api.models as models

    session, identity, api_content = _settle_image_turn(
        text="Describe this image",
        extra_text=(RECALL_NOTE,),
        timestamp=600.0,
    )
    context_user = next(
        message for message in session.context_messages
        if message.get("_active_turn_token") == identity["token"]
    )
    conflicting = {
        "role": "user",
        "content": _durable_agent_content(context_user["content"]),
        "timestamp": 600.0,
        "api_content": api_content + " ",
        "_state_db_row_id": 1,
    }
    reconciled = models.reconciled_state_db_messages_for_session(
        session,
        state_messages=[conflicting],
    )
    assert any(message.get("content") == conflicting["content"] for message in reconciled)


def test_ambiguous_state_db_mirror_identity_is_not_suppressed():
    import api.models as models

    session, identity, api_content = _settle_image_turn(
        text="Describe this image",
        extra_text=(RECALL_NOTE,),
        timestamp=700.0,
    )
    context_user = next(
        message for message in session.context_messages
        if message.get("_active_turn_token") == identity["token"]
    )
    mirror = _durable_agent_content(context_user["content"])
    state_rows = [
        {
            "role": "user",
            "content": mirror,
            "timestamp": 700.0,
            "api_content": api_content,
            "_state_db_row_id": row_id,
        }
        for row_id in (11, 12)
    ]
    reconciled = models.reconciled_state_db_messages_for_session(
        session,
        state_messages=state_rows,
    )
    assert any(message.get("content") == mirror for message in reconciled)


def test_state_db_image_projection_suppresses_only_exact_agent_row_id(
    monkeypatch, tmp_path
):
    import api.models as models

    timestamp = 800.0
    session, identity, _ = _settle_image_turn(
        text="Describe this image",
        queued_notifications=("Queued process update: sample job completed.",),
        extra_text=(RECALL_NOTE, PLUGIN_NOTE),
        timestamp=timestamp,
        agent_row_id=41,
    )
    context_user = next(
        message for message in session.context_messages
        if message.get("_active_turn_token") == identity["token"]
    )
    mirror = _durable_agent_content(context_user["content"])

    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE messages ("
            "id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, "
            "timestamp REAL, active INTEGER DEFAULT 1, api_content TEXT)"
        )
        conn.executemany(
            "INSERT INTO messages (id, session_id, role, content, timestamp, api_content) "
            "VALUES (?, ?, ?, ?, ?, NULL)",
            [
                (41, session.session_id, "user", mirror, timestamp),
                (42, session.session_id, "user", mirror, timestamp),
            ],
        )
    monkeypatch.setattr(models, "_active_state_db_path", lambda: db_path)
    state_rows = models.get_state_db_session_messages(session.session_id)
    assert [message["_state_db_row_id"] for message in state_rows] == [41, 42]
    assert all("api_content" not in message for message in state_rows)

    reconciled = models.reconciled_state_db_messages_for_session(
        session,
        state_messages=state_rows,
    )
    visible_literals = [
        message for message in reconciled
        if message.get("role") == "user" and message.get("content") == mirror
    ]
    assert len(visible_literals) == 1
    assert visible_literals[0]["_state_db_row_id"] == 42
    public_literals = [
        message for message in public_session_projection({"messages": reconciled})["messages"]
        if message.get("role") == "user" and message.get("content") == mirror
    ]
    assert len(public_literals) == 1
    assert "_state_db_row_id" not in public_literals[0]

    reconciled_context = models.reconciled_state_db_messages_for_session(
        session,
        prefer_context=True,
        state_messages=state_rows,
    )
    session.messages = reconciled
    session.context_messages = reconciled_context
    first_api_transcript = public_session_projection(
        {"messages": session.messages}
    )["messages"]
    first_next_replay = _new_turn_context_from_messages(
        session.context_messages,
        "Tell me more",
    )

    reconciled_again = models.reconciled_state_db_messages_for_session(
        session,
        state_messages=state_rows,
    )
    reconciled_context_again = models.reconciled_state_db_messages_for_session(
        session,
        prefer_context=True,
        state_messages=state_rows,
    )
    for messages in (reconciled_again, reconciled_context_again):
        assert sum(
            message.get("_state_db_row_id") == 42
            for message in messages
        ) == 1
    assert public_session_projection(
        {"messages": reconciled_again}
    )["messages"] == first_api_transcript
    assert _new_turn_context_from_messages(
        reconciled_context_again,
        "Tell me more",
    ) == first_next_replay

    for untrusted_identity in (
        {key: value for key, value in state_rows[0].items() if key != "_state_db_row_id"},
        {**state_rows[0], "_row_id": 42},
    ):
        unresolved = models.merge_session_messages_append_only(
            session.messages,
            models._suppress_native_image_display_mirrors(
                session,
                [untrusted_identity],
            ),
            incoming_provenance="state_db",
        )
        assert any(message.get("content") == mirror for message in unresolved)
        assert not any(
            "_webui_unmatched_native_image_mirror" in message
            for message in unresolved
        )


def test_marked_native_image_mirror_repairs_malformed_sidecar_once():
    import api.models as models

    timestamp = 850.0
    session, identity, api_content = _settle_image_turn(
        timestamp=timestamp,
        agent_row_id=41,
    )
    context_user = next(
        message for message in session.context_messages
        if message.get("_active_turn_token") == identity["token"]
    )
    mirror = _durable_agent_content(context_user["content"])
    session.messages.append({
        "role": "user",
        "content": mirror,
        "timestamp": timestamp,
        "_state_db_row_id": 42,
        "api_content": [],
    })
    state_row = {
        "role": "user",
        "content": mirror,
        "timestamp": timestamp,
        "_state_db_row_id": 42,
        "api_content": api_content,
    }

    marked = models._suppress_native_image_display_mirrors(session, [state_row])
    assert len(marked) == 1
    assert marked[0]["_webui_unmatched_native_image_mirror"] is True

    for _ in range(2):
        reconciled = models.reconciled_state_db_messages_for_session(
            session,
            state_messages=[state_row],
        )
        replay_rows = [
            message for message in _sanitize_messages_for_agent(reconciled)
            if message.get("content") == mirror
        ]
        assert len(replay_rows) == 1
        assert replay_rows[0]["api_content"] == api_content


def test_marked_native_image_mirror_conflict_stays_bounded_across_recovery():
    import api.models as models

    timestamp = 860.0
    session, identity, _ = _settle_image_turn(
        timestamp=timestamp,
        agent_row_id=41,
    )
    context_user = next(
        message for message in session.context_messages
        if message.get("_active_turn_token") == identity["token"]
    )
    mirror = _durable_agent_content(context_user["content"])
    session.messages.append({
        "role": "user",
        "content": mirror,
        "timestamp": timestamp,
        "_state_db_row_id": 42,
        "api_content": "OLD-PROVIDER-BYTES",
    })
    state_row = {
        "role": "user",
        "content": mirror,
        "timestamp": timestamp,
        "_state_db_row_id": 42,
        "api_content": "NEW-PROVIDER-BYTES",
    }
    marked = models._suppress_native_image_display_mirrors(session, [state_row])
    assert marked[0]["_webui_unmatched_native_image_mirror"] is True

    first_public = None
    first_replay = None
    for _ in range(4):
        display = models.reconciled_state_db_messages_for_session(
            session,
            state_messages=[state_row],
        )
        context = models.reconciled_state_db_messages_for_session(
            session,
            prefer_context=True,
            state_messages=[state_row],
        )
        display_row_42 = [
            message for message in display
            if message.get("_state_db_row_id") == 42
        ]
        context_row_42 = [
            message for message in context
            if message.get("_state_db_row_id") == 42
        ]
        assert len(display_row_42) == 2
        assert {message["api_content"] for message in display_row_42} == {
            "OLD-PROVIDER-BYTES",
            "NEW-PROVIDER-BYTES",
        }
        assert len(context_row_42) == 1
        assert context_row_42[0]["api_content"] == "NEW-PROVIDER-BYTES"

        public = public_session_projection({"messages": display})["messages"]
        replay = _sanitize_messages_for_agent(context)
        if first_public is None:
            first_public = public
            first_replay = replay
        else:
            assert public == first_public
            assert replay == first_replay
        session.messages = display
        session.context_messages = context


def test_marked_native_image_rows_with_distinct_ids_survive_recovery():
    import api.models as models

    timestamp = 870.0
    session, identity, api_content = _settle_image_turn(
        timestamp=timestamp,
        agent_row_id=41,
    )
    context_user = next(
        message for message in session.context_messages
        if message.get("_active_turn_token") == identity["token"]
    )
    mirror = _durable_agent_content(context_user["content"])
    state_rows = [
        {
            "role": "user",
            "content": mirror,
            "timestamp": timestamp,
            "_state_db_row_id": row_id,
            "api_content": api_content,
        }
        for row_id in (42, 43)
    ]

    first_public = None
    first_replay = None
    for _ in range(4):
        display = models.reconciled_state_db_messages_for_session(
            session,
            state_messages=state_rows,
        )
        context = models.reconciled_state_db_messages_for_session(
            session,
            prefer_context=True,
            state_messages=state_rows,
        )
        for messages in (display, context):
            rows = [
                message for message in messages
                if message.get("_state_db_row_id") in (42, 43)
            ]
            assert [message["_state_db_row_id"] for message in rows] == [42, 43]
            assert all(message["api_content"] == api_content for message in rows)

        public = public_session_projection({"messages": display})["messages"]
        replay = _sanitize_messages_for_agent(context)
        if first_public is None:
            first_public = public
            first_replay = replay
            assert sum(message.get("content") == mirror for message in public) == 2
        else:
            assert public == first_public
            assert replay == first_replay
        session.messages = display
        session.context_messages = context


def test_unlinked_state_db_image_projection_uses_existing_reconciliation():
    import api.models as models

    for prefer_context in (False, True):
        for flushed in (False, True):
            timestamp = 900.0
            session, identity, _ = _settle_image_turn(
                timestamp=timestamp,
                extra_text=(RECALL_NOTE, PLUGIN_NOTE),
                agent_row_id=None,
            )
            context_user = next(
                message for message in session.context_messages
                if message.get("_active_turn_token") == identity["token"]
            )
            assert not any(
                key in context_user
                for key in ("_row_id", "_state_db_row_id", "_db_row_id", "state_db_row_id")
            )
            state_rows = (
                [{
                    "role": "user",
                    "content": _durable_agent_content(context_user["content"]),
                    "timestamp": timestamp,
                    "_state_db_row_id": 1,
                }]
                if flushed
                else []
            )

            recovered = models.reconciled_state_db_messages_for_session(
                session,
                prefer_context=prefer_context,
                state_messages=state_rows,
            )
            users = [
                message for message in recovered
                if message.get("role") == "user" and message.get("timestamp") == timestamp
            ]
            assert len(users) == 1
            display_users = [
                message for message in session.messages
                if message.get("role") == "user" and message.get("timestamp") == timestamp
            ]
            assert len(display_users) == 1
            assert display_users[0]["content"] == "Describe this image"
            assert users[0]["content"] == (
                context_user["content"] if prefer_context else "Describe this image"
            )
            if prefer_context:
                next_turn = _new_turn_context_from_messages(
                    recovered,
                    "Tell me more",
                )
                replay_users = [message for message in next_turn if message.get("role") == "user"]
                assert len(replay_users) == 1
                assert replay_users[0]["content"] == context_user["content"]


def test_agent_index_and_turn_id_do_not_claim_unrelated_user_row():
    _, identity, _ = _settle_image_turn(text="Describe this image")
    unrelated = [{"role": "user", "content": "Different submitted text"}]
    identity["current_turn_user_idx"] = 0
    assert _find_active_turn_checkpoint_index(
        unrelated,
        [],
        identity,
        "Describe this image",
    ) is None

    matching = [{
        "role": "user",
        "content": _native_user_content(
            "Describe this image", IMAGE_A, (RECALL_NOTE, PLUGIN_NOTE)
        ),
    }]
    assert _find_active_turn_checkpoint_index(
        matching,
        [],
        identity,
        "Describe this image",
    ) == 0
