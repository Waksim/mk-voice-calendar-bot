import stat

from tg_voice_transcriber_bot.state import StateStore


def test_state_persists_job_and_advances_offset(tmp_path):
    path = tmp_path / "state.json"
    state = StateStore(path)
    state.save_job(
        42,
        {"account": "personal", "user_message_id": 123, "status": "found"},
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not path.with_suffix(".tmp").exists()

    restored = StateStore(path)
    assert restored.job(42)["user_message_id"] == 123
    assert restored.after_message_id("personal") == 123

    restored.complete(42)
    final = StateStore(path)
    assert final.offset == 43
    assert final.job(42) is None


def test_webhook_queue_is_durable_ordered_and_independent_from_offset(tmp_path):
    path = tmp_path / "state.json"
    state = StateStore(path, completed_update_limit=3)

    assert state.enqueue_update({"update_id": 100, "message": {"text": "first"}})
    assert state.enqueue_update({"update_id": 99, "message": {"text": "second"}})
    assert not state.enqueue_update({"update_id": 100, "message": {"text": "duplicate"}})
    assert state.pending_update_count == 2
    assert state.next_pending_update()["update_id"] == 100

    restored = StateStore(path, completed_update_limit=3)
    assert restored.next_pending_update()["update_id"] == 100
    restored.complete_webhook_update(100)
    assert restored.offset == 0
    assert restored.next_pending_update()["update_id"] == 99
    assert not restored.enqueue_update({"update_id": 100})

    # A lower, previously unseen ID remains valid; webhook deduplication must
    # not use the polling high-water mark.
    assert restored.enqueue_update({"update_id": 3})


def test_completed_webhook_ids_are_bounded_and_jobs_are_removed(tmp_path):
    state = StateStore(tmp_path / "state.json", completed_update_limit=2)
    for update_id in (10, 11, 12):
        assert state.enqueue_update({"update_id": update_id})
        state.save_job(
            update_id,
            {"account": "personal", "user_message_id": update_id, "status": "done"},
        )
        state.complete_webhook_update(update_id)

    assert state.completed_update_ids == (11, 12)
    assert state.pending_update_count == 0
    assert state.job(12) is None
    assert state.offset == 0
