import pytest


@pytest.fixture(autouse=True)
def configured_test_owner_ids(monkeypatch):
    """Keep every test independent from the developer's ignored .env file."""

    monkeypatch.setenv("TELEGRAM_PERSONAL_USER_ID", "100000001")
    monkeypatch.setenv("TELEGRAM_WORK_USER_ID", "100000002")
    monkeypatch.delenv("TELEGRAM_PERSONAL_USER_ID_FILE", raising=False)
    monkeypatch.delenv("TELEGRAM_WORK_USER_ID_FILE", raising=False)
