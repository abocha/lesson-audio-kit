import asyncio
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
import pytest

from gradio_frontend.app import app


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": [
            ("authorization", "DUMMY_API_KEY"),
        ],
        "match_on": ["method", "scheme", "host", "port", "path", "query", "body"],
    }


@pytest.fixture
def mock_background_tasks_capture_and_run(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[MagicMock, list[Any]], None, None]:
    captured_task_awaitables = []

    def capture_task_for_manual_run(task_func: Any, *args: Any, **kwargs: Any) -> None:
        async def task_wrapper() -> None:
            if asyncio.iscoroutinefunction(task_func):
                await task_func(*args, **kwargs)
            else:
                task_func(*args, **kwargs)

        captured_task_awaitables.append(task_wrapper)

    mock_add_task = MagicMock(side_effect=capture_task_for_manual_run)
    monkeypatch.setattr(BackgroundTasks, "add_task", mock_add_task)

    yield mock_add_task, captured_task_awaitables


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, Any, None]:
    with TestClient(app) as c:
        yield c
