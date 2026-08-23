"""Small atomic state store for Bot API offsets and in-flight jobs."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


class StateStore:
    def __init__(self, path: Path, *, completed_update_limit: int = 2048) -> None:
        if completed_update_limit <= 0:
            raise ValueError("completed update limit must be positive")
        self.path = path
        self.completed_update_limit = completed_update_limit
        self.data: dict[str, Any] = self._load()

    @staticmethod
    def _default() -> dict[str, Any]:
        return {
            "offset": 0,
            "last_user_message_id": {"personal": 0, "work": 0},
            "jobs": {},
            # Webhook updates are persisted before the HTTP request is
            # acknowledged. A list preserves arrival order across restarts.
            "pending_updates": [],
            # Do not use the polling offset for webhook deduplication: webhook
            # deliveries may arrive out of order, and Telegram can randomize
            # update_id after a week without updates.
            "completed_update_ids": [],
        }

    @staticmethod
    def _update_id(update: dict[str, Any]) -> int:
        update_id = update.get("update_id")
        if isinstance(update_id, bool) or not isinstance(update_id, int) or update_id < 0:
            raise ValueError("Telegram update_id must be a non-negative integer")
        return update_id

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default()
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read state file: {type(exc).__name__}") from exc
        default = self._default()
        if not isinstance(loaded, dict):
            raise RuntimeError("State file root must be an object")
        last_ids = loaded.get("last_user_message_id", {})
        jobs = loaded.get("jobs", {})
        offset = loaded.get("offset", 0)
        pending_updates = loaded.get("pending_updates", [])
        completed_update_ids = loaded.get("completed_update_ids", [])
        if not isinstance(last_ids, dict):
            raise RuntimeError("State last_user_message_id must be an object")
        if not isinstance(jobs, dict):
            raise RuntimeError("State jobs must be an object")
        if not isinstance(offset, int) or offset < 0:
            raise RuntimeError("State offset must be a non-negative integer")
        if not isinstance(pending_updates, list):
            raise RuntimeError("State pending_updates must be an array")
        if not isinstance(completed_update_ids, list):
            raise RuntimeError("State completed_update_ids must be an array")
        for account, message_id in last_ids.items():
            if not isinstance(account, str) or not isinstance(message_id, int):
                raise RuntimeError("State message IDs must be integers")
        pending_ids: set[int] = set()
        for update in pending_updates:
            if not isinstance(update, dict):
                raise RuntimeError("State pending update must be an object")
            try:
                update_id = self._update_id(update)
            except ValueError as exc:
                raise RuntimeError("State pending update ID is invalid") from exc
            if update_id in pending_ids:
                raise RuntimeError("State contains duplicate pending update IDs")
            pending_ids.add(update_id)
        completed_ids: list[int] = []
        completed_seen: set[int] = set()
        for update_id in completed_update_ids:
            if (
                isinstance(update_id, bool)
                or not isinstance(update_id, int)
                or update_id < 0
            ):
                raise RuntimeError("State completed update ID is invalid")
            if update_id in pending_ids:
                raise RuntimeError("State update cannot be pending and completed")
            if update_id not in completed_seen:
                completed_ids.append(update_id)
                completed_seen.add(update_id)
        default["offset"] = offset
        default["last_user_message_id"].update(last_ids)
        default["jobs"] = jobs
        default["pending_updates"] = deepcopy(pending_updates)
        default["completed_update_ids"] = completed_ids[
            -self.completed_update_limit :
        ]
        return default

    @property
    def offset(self) -> int:
        return int(self.data.get("offset", 0))

    def after_message_id(self, account: str) -> int:
        return int(self.data["last_user_message_id"].get(account, 0))

    def job(self, update_id: int) -> dict[str, Any] | None:
        value = self.data["jobs"].get(str(update_id))
        return value if isinstance(value, dict) else None

    @property
    def pending_update_count(self) -> int:
        return len(self.data["pending_updates"])

    @property
    def completed_update_ids(self) -> tuple[int, ...]:
        return tuple(self.data["completed_update_ids"])

    def next_pending_update(self) -> dict[str, Any] | None:
        pending = self.data["pending_updates"]
        return deepcopy(pending[0]) if pending else None

    def enqueue_update(self, update: dict[str, Any]) -> bool:
        """Persist a webhook update, returning false for a durable duplicate."""

        if not isinstance(update, dict):
            raise ValueError("Telegram update must be an object")
        update_id = self._update_id(update)
        if update_id in self.data["completed_update_ids"]:
            return False
        if any(
            self._update_id(item) == update_id
            for item in self.data["pending_updates"]
        ):
            return False
        self.data["pending_updates"].append(deepcopy(update))
        try:
            self.save()
        except Exception:
            self.data["pending_updates"].pop()
            raise
        return True

    def complete_webhook_update(self, update_id: int) -> None:
        """Commit webhook completion without consulting or changing offset."""

        if isinstance(update_id, bool) or not isinstance(update_id, int) or update_id < 0:
            raise ValueError("Telegram update_id must be a non-negative integer")
        previous_pending = self.data["pending_updates"]
        previous_completed = list(self.data["completed_update_ids"])
        previous_job = self.data["jobs"].get(str(update_id))
        had_job = str(update_id) in self.data["jobs"]
        self.data["pending_updates"] = [
            item
            for item in previous_pending
            if self._update_id(item) != update_id
        ]
        self.data["jobs"].pop(str(update_id), None)
        completed = self.data["completed_update_ids"]
        if update_id not in completed:
            completed.append(update_id)
            del completed[: max(0, len(completed) - self.completed_update_limit)]
        try:
            self.save()
        except Exception:
            self.data["pending_updates"] = previous_pending
            self.data["completed_update_ids"] = previous_completed
            if had_job:
                self.data["jobs"][str(update_id)] = previous_job
            else:
                self.data["jobs"].pop(str(update_id), None)
            raise

    def save_job(self, update_id: int, job: dict[str, Any]) -> None:
        self.data["jobs"][str(update_id)] = job
        account = str(job["account"])
        message_id = int(job["user_message_id"])
        self.data["last_user_message_id"][account] = max(
            self.after_message_id(account), message_id
        )
        self.save()

    def complete(self, update_id: int) -> None:
        self.data["offset"] = max(self.offset, update_id + 1)
        self.data["jobs"].pop(str(update_id), None)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        serialized = json.dumps(
            self.data, ensure_ascii=False, sort_keys=True, indent=2
        ) + "\n"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = -1
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        os.replace(temporary, self.path)
        os.chmod(self.path, 0o600)
        # Persist the rename itself before acknowledging a webhook. Directory
        # fsync is supported on the Unix platforms where this service runs.
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(self.path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
