from collections.abc import Generator
import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from dialogue_tts_core.config_models import (
    QualityTier,
    SpeakerTTSConfig,
    TTSJobStatusCompleted,
    TTSJobStatusFailed,
    TTSJobStatusPending,
    TTSRequestPayload,
)
from gradio_frontend.app import app, job_store, run_tts_orchestration_task

# Add the project root to the sys.path to allow imports from
# dialogue_tts_core and gradio_frontend
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_job_store() -> Generator[None, None, None]:
    """Clears the job_store before each test."""
    job_store.clear()
    yield


@pytest.fixture
def mock_core_services() -> Generator[dict[str, MagicMock | AsyncMock], None, None]:
    """Fixture to mock core services for /api/tts endpoint."""
    with (
        patch("gradio_frontend.app.parse_dialogue_script") as mock_parse_script,
        patch(
            "gradio_frontend.app.get_unique_speakers_from_parsed_script"
        ) as mock_get_speakers,
        patch(
            "gradio_frontend.app.resolve_speaker_configurations"
        ) as mock_resolve_configs,
        patch("gradio_frontend.app.orchestrate_tts_synthesis") as mock_orchestrate_tts,
        patch("gradio_frontend.app.async_openai_client") as mock_openai_client,
    ):
        mock_parse_script.return_value = (
            [{"speaker": "Speaker1", "line": "Hello"}],
            None,
        )
        mock_get_speakers.return_value = ["Speaker1"]
        mock_resolve_configs.return_value = {
            "Speaker1": SpeakerTTSConfig(voice="alloy")
        }
        mock_orchestrate_tts.return_value = (
            "path/to/zip.zip",
            "path/to/merged.mp3",
            "Success",
        )
        mock_openai_client.return_value = (
            AsyncMock()
        )  # Ensure it's an AsyncMock if used in async context
        yield {
            "parse_script": mock_parse_script,
            "get_speakers": mock_get_speakers,
            "resolve_configs": mock_resolve_configs,
            "orchestrate_tts": mock_orchestrate_tts,
            "openai_client": mock_openai_client,
        }


@pytest.fixture
def mock_background_tasks() -> Generator[MagicMock, None, None]:
    """Fixture to mock BackgroundTasks.add_task."""
    with patch("fastapi.BackgroundTasks.add_task") as mock_add_task:
        yield mock_add_task


# --- Tests for POST /api/tts ---


def test_submit_tts_job_valid_payload(
    mock_core_services: Any, mock_background_tasks: MagicMock
) -> None:
    """Test valid TTS job submission."""
    payload = TTSRequestPayload(
        script_text="Speaker1: Hello world.",
        tts_global_model="tts-1",
        speaker_config_method="global",
        global_speaker_config=SpeakerTTSConfig(voice="alloy"),
    )
    response = client.post("/api/tts", json=payload.model_dump())

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert "status_url" in data
    assert data["status_url"].startswith("/api/tts/status/")
    assert len(data["job_id"]) == 32  # UUID hex format

    mock_core_services["parse_script"].assert_called_once_with(payload.script_text)
    mock_core_services["get_speakers"].assert_called_once()
    mock_core_services["resolve_configs"].assert_called_once()
    mock_background_tasks.assert_called_once()
    # Verify the task added is run_tts_orchestration_task
    assert mock_background_tasks.call_args[0][0] == run_tts_orchestration_task
    # Verify initial job status is pending
    assert job_store[data["job_id"]]["status"] == "pending"


def test_submit_tts_job_with_routing_parameters_full(
    mock_core_services: Any,  # noqa: ARG001
    mock_background_tasks: MagicMock,
) -> None:
    """Test TTS job submission with all routing parameters."""
    payload_with_routing = TTSRequestPayload(
        script_text="Speaker1: Hello world with routing.",
        tts_global_model="tts-1",
        speaker_config_method="global",
        global_speaker_config=SpeakerTTSConfig(voice="alloy"),
        user_id="test_user_123",
        desired_quality_tier="HIGH",
        max_total_job_cost_usd=0.5,
        prefer_low_latency=True,
        prefer_emotion_support=True,
        prefer_voice_cloning=True,
        specific_engine_id="engine_xyz",
    )
    response = client.post("/api/tts", json=payload_with_routing.model_dump())

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data

    mock_background_tasks.assert_called_once()
    called_args, called_kwargs = mock_background_tasks.call_args
    assert called_args[0] == run_tts_orchestration_task
    assert called_kwargs.get("user_id") == payload_with_routing.user_id
    assert (
        called_kwargs.get("desired_quality_tier_str")
        == payload_with_routing.desired_quality_tier
    )
    assert (
        called_kwargs.get("max_total_job_cost_usd")
        == payload_with_routing.max_total_job_cost_usd
    )
    assert (
        called_kwargs.get("prefer_low_latency")
        == payload_with_routing.prefer_low_latency
    )
    assert (
        called_kwargs.get("prefer_emotion_support")
        == payload_with_routing.prefer_emotion_support
    )
    assert (
        called_kwargs.get("prefer_voice_cloning")
        == payload_with_routing.prefer_voice_cloning
    )
    assert (
        called_kwargs.get("specific_engine_id")
        == payload_with_routing.specific_engine_id
    )


def test_submit_tts_job_with_routing_parameters_partial(
    mock_core_services: Any,  # noqa: ARG001
    mock_background_tasks: MagicMock,
) -> None:
    """Test TTS job submission with a subset of routing parameters."""
    payload_with_routing = TTSRequestPayload(
        script_text="Speaker1: Hello world with partial routing.",
        tts_global_model="tts-1",
        speaker_config_method="global",
        global_speaker_config=SpeakerTTSConfig(voice="alloy"),
        user_id="partial_user",
        prefer_low_latency=True,
        specific_engine_id="engine_abc",
    )
    response = client.post("/api/tts", json=payload_with_routing.model_dump())

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data

    mock_background_tasks.assert_called_once()
    called_args, called_kwargs = mock_background_tasks.call_args
    assert called_args[0] == run_tts_orchestration_task
    assert called_kwargs.get("user_id") == payload_with_routing.user_id
    assert called_kwargs.get("desired_quality_tier_str") == QualityTier.MID.name
    assert called_kwargs.get("max_total_job_cost_usd") is None  # Not provided
    assert (
        called_kwargs.get("prefer_low_latency")
        == payload_with_routing.prefer_low_latency
    )
    assert called_kwargs.get("prefer_emotion_support") is False  # Default value
    assert called_kwargs.get("prefer_voice_cloning") is False  # Default value
    assert (
        called_kwargs.get("specific_engine_id")
        == payload_with_routing.specific_engine_id
    )


def test_submit_tts_job_with_routing_parameters_defaults(
    mock_core_services: Any,  # noqa: ARG001
    mock_background_tasks: MagicMock,
) -> None:
    """Test TTS job submission with routing parameters relying on defaults."""
    payload_with_routing = TTSRequestPayload(
        script_text="Speaker1: Hello world with default routing.",
        tts_global_model="tts-1",
        speaker_config_method="global",
        global_speaker_config=SpeakerTTSConfig(voice="alloy"),
        # No routing parameters explicitly set
    )
    response = client.post("/api/tts", json=payload_with_routing.model_dump())

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data

    mock_background_tasks.assert_called_once()
    called_args, called_kwargs = mock_background_tasks.call_args
    assert called_args[0] == run_tts_orchestration_task
    assert called_kwargs.get("user_id") is None
    assert called_kwargs.get("desired_quality_tier_str") == QualityTier.MID.name
    assert called_kwargs.get("max_total_job_cost_usd") is None
    assert called_kwargs.get("prefer_low_latency") is False
    assert called_kwargs.get("prefer_emotion_support") is False
    assert called_kwargs.get("prefer_voice_cloning") is False
    assert called_kwargs.get("specific_engine_id") is None


def test_submit_tts_job_invalid_payload_missing_field() -> None:
    """Test invalid TTS job submission with missing required field."""
    payload = {
        "script_text": "Speaker1: Hello world.",
        # "tts_global_model" is missing
        "speaker_config_method": "global",
        "global_speaker_config": {"voice": "alloy"},
    }
    response = client.post("/api/tts", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()
    assert "tts_global_model" in response.json()["detail"][0]["loc"]


def test_submit_tts_job_empty_script(mock_core_services: Any) -> None:
    """Test TTS job submission with an empty script."""
    mock_core_services["parse_script"].return_value = ([], None)
    payload = TTSRequestPayload(
        script_text="",
        tts_global_model="tts-1",
        speaker_config_method="global",
        global_speaker_config=SpeakerTTSConfig(voice="alloy"),
    )
    response = client.post("/api/tts", json=payload.model_dump())
    assert response.status_code == 400
    assert (
        "Script is empty or contains no processable lines." in response.json()["detail"]
    )


def test_submit_tts_job_no_speakers_in_script(mock_core_services: Any) -> None:
    """Test TTS job submission with no speakers found in script."""
    mock_core_services["parse_script"].return_value = ([{"line": "Just a line"}], None)
    mock_core_services["get_speakers"].return_value = []
    payload = TTSRequestPayload(
        script_text="Just a line.",
        tts_global_model="tts-1",
        speaker_config_method="global",
        global_speaker_config=SpeakerTTSConfig(voice="alloy"),
    )
    response = client.post("/api/tts", json=payload.model_dump())
    assert response.status_code == 400
    assert "No speakers found in the script." in response.json()["detail"]


def test_submit_tts_job_config_resolution_error(mock_core_services: Any) -> None:
    """Test TTS job submission with speaker configuration resolution error."""
    mock_core_services["resolve_configs"].side_effect = ValueError("Invalid config")
    payload = TTSRequestPayload(
        script_text="Speaker1: Hello.",
        tts_global_model="tts-1",
        speaker_config_method="global",
        global_speaker_config=SpeakerTTSConfig(voice="alloy"),
    )
    response = client.post("/api/tts", json=payload.model_dump())
    assert response.status_code == 422
    assert (
        "Speaker configuration resolution error: Invalid config"
        in response.json()["detail"]
    )


def test_submit_tts_job_openai_client_not_initialized() -> None:
    """Test TTS job submission when OpenAI client is not initialized."""
    with patch("gradio_frontend.app.async_openai_client", None):
        payload = TTSRequestPayload(
            script_text="Speaker1: Hello world.",
            tts_global_model="tts-1",
            speaker_config_method="global",
            global_speaker_config=SpeakerTTSConfig(voice="alloy"),
        )
        response = client.post("/api/tts", json=payload.model_dump())
        assert response.status_code == 503
        assert (
            "TTS service not available: OpenAI client not initialized."
            in response.json()["detail"]
        )


# --- Tests for GET /api/tts/status/{job_id} ---


def test_get_tts_job_status_pending() -> None:
    """Test getting status for a pending job."""
    job_id = "test_pending_job"
    job_store[job_id] = {"status": "pending", "request_payload": {}}
    response = client.get(f"/api/tts/status/{job_id}")
    assert response.status_code == 200
    status_response = TTSJobStatusPending.model_validate(response.json())
    assert isinstance(status_response, TTSJobStatusPending)
    assert status_response.job_id == job_id
    assert status_response.status == "pending"


def test_get_tts_job_status_processing() -> None:
    """Test getting status for a processing job."""
    job_id = "test_processing_job"
    job_store[job_id] = {"status": "processing", "request_payload": {}}
    response = client.get(f"/api/tts/status/{job_id}")
    assert response.status_code == 200
    status_response = TTSJobStatusPending.model_validate(response.json())
    assert isinstance(
        status_response, TTSJobStatusPending
    )  # Pending is used for processing too
    assert status_response.job_id == job_id
    assert status_response.status == "processing"


def test_get_tts_job_status_completed() -> None:
    """Test getting status for a completed job."""
    job_id = "test_completed_job"
    # Mock os.path.exists for file URL construction
    with patch("os.path.exists", return_value=True):
        job_store[job_id] = {
            "status": "completed",
            "message": "Job done.",
            "outputs": {
                "zip_file_local_path": f".job_outputs/{job_id}/output.zip",
                "merged_mp3_local_path": f".job_outputs/{job_id}/merged.mp3",
            },
            "request_payload": {},
        }
        response = client.get(f"/api/tts/status/{job_id}")
        assert response.status_code == 200
        status_response = TTSJobStatusCompleted.model_validate(response.json())
        assert isinstance(status_response, TTSJobStatusCompleted)
        assert status_response.job_id == job_id
        assert status_response.status == "completed"
        assert status_response.message == "Job done."
        assert status_response.outputs is not None
        assert status_response.outputs.zip_file_url == f"/job_files/{job_id}/output.zip"
        assert (
            status_response.outputs.merged_mp3_url == f"/job_files/{job_id}/merged.mp3"
        )


def test_get_tts_job_status_completed_no_files() -> None:
    """Test getting status for a completed job with no output files."""
    job_id = "test_completed_no_files_job"
    with patch("os.path.exists", return_value=False):  # Simulate files not existing
        job_store[job_id] = {
            "status": "completed",
            "message": "Job done, no files.",
            "outputs": {
                "zip_file_local_path": f".job_outputs/{job_id}/output.zip",
                "merged_mp3_local_path": f".job_outputs/{job_id}/merged.mp3",
            },
            "request_payload": {},
        }
        response = client.get(f"/api/tts/status/{job_id}")
        assert response.status_code == 200
        status_response = TTSJobStatusCompleted.model_validate(response.json())
        assert isinstance(status_response, TTSJobStatusCompleted)
        assert status_response.outputs.zip_file_url is None
        assert status_response.outputs.merged_mp3_url is None


def test_get_tts_job_status_failed() -> None:
    """Test getting status for a failed job."""
    job_id = "test_failed_job"
    job_store[job_id] = {
        "status": "failed",
        "error_message": "Something went wrong.",
        "request_payload": {},
    }
    response = client.get(f"/api/tts/status/{job_id}")
    assert response.status_code == 200
    status_response = TTSJobStatusFailed.model_validate(response.json())
    assert isinstance(status_response, TTSJobStatusFailed)
    assert status_response.job_id == job_id
    assert status_response.status == "failed"
    assert status_response.error_message == "Something went wrong."


def test_get_tts_job_status_unknown_job_id() -> None:
    """Test getting status for an unknown job ID."""
    job_id = "unknown_job_id"
    response = client.get(f"/api/tts/status/{job_id}")
    assert response.status_code == 404
    assert "Job ID not found." in response.json()["detail"]


@pytest.fixture
def mock_background_tasks_capture_and_run(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[MagicMock, list[Any]], None, None]:
    """
    Fixture to mock BackgroundTasks.add_task and capture the awaitable tasks
    for manual execution within tests.
    """
    captured_tasks: list = []
    # The original_add_task line was removed as it was causing Pylance errors due to
    # incorrect key type and is not needed with monkeypatch.

    mock_add_task = MagicMock()

    def _add_task_side_effect(func: Any, *args: Any, **kwargs: Any) -> None:
        async def awaitable_task_runner() -> None:
            if func.__name__ == "run_tts_orchestration_task":
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)

        captured_tasks.append(awaitable_task_runner)

    mock_add_task.side_effect = _add_task_side_effect
    monkeypatch.setattr("fastapi.BackgroundTasks.add_task", mock_add_task)
    yield mock_add_task, captured_tasks
    # Clean up after test
    # monkeypatch handles cleanup automatically, so no need for manual restoration.


async def test_submit_tts_job_async_completion_and_orchestrator_params(
    client: TestClient,
    mock_core_services: Any,
    mock_background_tasks_capture_and_run: tuple[MagicMock, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Test the full asynchronous job submission, background task execution,
    and status updates, including verification of orchestrator parameters.
    """
    # === Phase 1: Setup for Successful Job Completion ===

    # 1.1. UNPACK fixture: mock_add_task_method, captured_tasks_list from
    # mock_background_tasks_capture_and_run.
    mock_add_task_method, captured_tasks_list = mock_background_tasks_capture_and_run

    # 1.2. DEFINE a TTSRequestPayload dictionary (`payload_dict`) with
    # comprehensive fields.
    payload_dict = {
        "script_text": "Speaker S1: Hello world.\nSpeaker S2: This is a test.",
        "tts_global_model": "tts-1",
        "speaker_config_method": "per_speaker_configs",
        "global_speaker_config": {"voice": "alloy", "speed": 1.0},
        "per_speaker_configs": [
            {"speaker_name": "S1", "config": {"voice": "echo"}},
            {"speaker_name": "S2", "config": {"voice": "fable"}},
        ],
        "global_pause_ms": 100,
        "user_id": "test_user_async",
        "desired_quality_tier": "MID",
        "max_total_job_cost_usd": 0.50,
        "prefer_low_latency": False,
        "prefer_emotion_support": True,
        "prefer_voice_cloning": False,
        "specific_engine_id": None,
    }
    payload = TTSRequestPayload(**payload_dict)

    # 1.3. CONFIGURE mocks from `mock_core_services`:
    mock_parse_script = mock_core_services["parse_script"]
    mock_get_speakers = mock_core_services["get_speakers"]
    mock_resolve_configs = mock_core_services["resolve_configs"]
    mock_async_openai_client = mock_core_services["openai_client"]

    mocked_parsed_script_output = (
        [
            {"speaker": "S1", "text": "Hello world."},
            {"speaker": "S2", "text": "This is a test."},
        ],
        2,
    )
    mock_parse_script.return_value = mocked_parsed_script_output
    mocked_speakers_list = ["S1", "S2"]
    mock_get_speakers.return_value = mocked_speakers_list
    mocked_resolved_configs = {
        "S1": SpeakerTTSConfig(voice="echo"),
        "S2": SpeakerTTSConfig(voice="fable"),
    }
    mock_resolve_configs.return_value = mocked_resolved_configs

    # 1.4. CONFIGURE the main `mock_orchestrator =
    # mock_core_services["orchestrate_tts"]`.
    mock_orchestrator = mock_core_services["orchestrate_tts"]
    mock_orchestrator.reset_mock()  # Ensure clean state for this part of the test

    # DEFINE expected output values for a successful run:
    expected_zip_filename = "dialogue_lines_async.zip"
    expected_merged_filename = "merged_dialogue_async.mp3"
    expected_status_msg_success = (
        "TTS Job Summary: All 2 lines processed. Successful: 2, Failed: 0."
    )
    expected_synthesis_details_success = [
        {
            "id": 0,
            "speaker": "S1",
            "status": "success",
            "duration_ms": 1000,
            "error": None,
        },
        {
            "id": 1,
            "speaker": "S2",
            "status": "success",
            "duration_ms": 1200,
            "error": None,
        },
    ]

    # 1.5. MONKEYPATCH environment and OS functions:
    test_job_output_dir = ".job_outputs_test_api_async"
    test_cache_base_dir = ".cache_test_api_async"
    monkeypatch.setattr("os.path.exists", MagicMock(return_value=True))
    monkeypatch.setenv("APP_JOB_OUTPUT_DIR", test_job_output_dir)
    monkeypatch.setenv("APP_CACHE_BASE_DIR", test_cache_base_dir)
    monkeypatch.setattr("os.makedirs", MagicMock())

    # === Phase 2: Action - Submit Job (Success Scenario) ===

    # 2.1. CALL `response = client.post("/api/tts", json=payload.model_dump())`.
    response = client.post("/api/tts", json=payload.model_dump())

    # === Phase 3: Assertions - Initial Response (Success Scenario) ===

    # 3.1. ASSERT `response.status_code == 202`.
    assert response.status_code == 202
    # 3.2. PARSE `job_creation_data = response.json()`.
    job_creation_data = response.json()
    # 3.3. GET `job_id = job_creation_data["job_id"]`.
    # ASSERT `job_id` is not None and is a string.
    job_id = job_creation_data["job_id"]
    assert job_id is not None
    assert isinstance(job_id, str)
    # 3.4. ASSERT `job_creation_data["status_url"] == f"/api/tts/status/{job_id}"`.
    assert job_creation_data["status_url"] == f"/api/tts/status/{job_id}"
    # 3.5. ASSERT `mock_add_task_method.assert_called_once()`.
    mock_add_task_method.assert_called_once()
    # 3.6. ASSERT `len(captured_tasks_list) == 1`.
    assert len(captured_tasks_list) == 1

    # 3.7. UPDATE expected paths for `mock_orchestrator.return_value` using
    # the actual `job_id`:
    actual_expected_zip_path = os.path.join(
        test_job_output_dir, job_id, expected_zip_filename
    )
    actual_expected_merged_path = os.path.join(
        test_job_output_dir, job_id, expected_merged_filename
    )
    mock_orchestrator.return_value = (
        actual_expected_zip_path,
        actual_expected_merged_path,
        expected_status_msg_success,
        expected_synthesis_details_success,
    )

    # === Phase 4: Action - Manually Run Background Task (Success Scenario) ===

    # 4.1. GET the captured task: `awaitable_task_runner = captured_tasks_list[0]`.
    awaitable_task_runner = captured_tasks_list[0]
    # 4.2. AWAIT `await awaitable_task_runner()`.
    await awaitable_task_runner()

    # === Phase 5: Assertions - Orchestrator Call (Success Scenario) ===

    # 5.1. ASSERT `mock_orchestrator.assert_called_once()`.
    mock_orchestrator.assert_called_once()
    # 5.2. GET `call_args = mock_orchestrator.call_args`.
    call_args_tuple, call_kwargs = mock_orchestrator.call_args

    # 5.3. VERIFY arguments passed to `orchestrate_tts_synthesis`:
    assert call_kwargs.get("parsed_script") == mocked_parsed_script_output[0]
    assert call_kwargs.get("resolved_speaker_configs_map") == mocked_resolved_configs
    assert call_kwargs.get("global_pause_ms") == payload.global_pause_ms
    assert call_kwargs.get("output_directory") == os.path.join(
        test_job_output_dir, job_id
    )
    assert call_kwargs.get("cache_base_dir") == test_cache_base_dir
    assert call_kwargs.get("user_id") == payload.user_id
    assert call_kwargs.get("desired_quality_tier_str") == payload.desired_quality_tier
    assert call_kwargs.get("max_total_job_cost_usd") == payload.max_total_job_cost_usd
    assert call_kwargs.get("prefer_low_latency_routing") == payload.prefer_low_latency
    assert (
        call_kwargs.get("prefer_emotion_support_routing")
        == payload.prefer_emotion_support
    )
    assert (
        call_kwargs.get("prefer_voice_cloning_routing") == payload.prefer_voice_cloning
    )
    assert call_kwargs.get("specific_engine_id") == payload.specific_engine_id
    assert call_kwargs.get("openai_client") is mock_async_openai_client

    # === Phase 6: Assertions - Job Status API (Success Scenario) ===

    # 6.1. CALL `status_response_success = client.get(f"/api/tts/status/{job_id}")`.
    status_response_success = client.get(f"/api/tts/status/{job_id}")
    # 6.2. ASSERT `status_response_success.status_code == 200`.
    assert status_response_success.status_code == 200
    # 6.3. PARSE `status_data_success = status_response_success.json()`.
    status_data_success = status_response_success.json()
    # 6.4. ASSERT `status_data_success["job_id"] == job_id`.
    assert status_data_success["job_id"] == job_id
    # 6.5. ASSERT `status_data_success["status"] == "completed"`.
    assert status_data_success["status"] == "completed"
    # 6.6. ASSERT `status_data_success["message"] == expected_status_msg_success`.
    assert status_data_success["message"] == expected_status_msg_success
    # 6.7. CONSTRUCT expected file URLs:
    expected_zip_url = f"/job_files/{job_id}/{expected_zip_filename}"
    expected_merged_url = f"/job_files/{job_id}/{expected_merged_filename}"
    # 6.8. ASSERT `status_data_success["outputs"]["zip_file_url"] == expected_zip_url`.
    assert status_data_success["outputs"]["zip_file_url"] == expected_zip_url
    # 6.9. ASSERT `status_data_success["outputs"]["merged_mp3_url"]
    # == expected_merged_url`.
    assert status_data_success["outputs"]["merged_mp3_url"] == expected_merged_url
    # 6.10. ASSERT `status_data_success["synthesis_details"]
    # == expected_synthesis_details_success`.
    assert (
        status_data_success["synthesis_details"] == expected_synthesis_details_success
    )

    # === Phase 7: Setup for Failed Job Completion ===

    # 7.1. RESET mocks for a clean run: `mock_add_task_method.reset_mock()`,
    # `mock_orchestrator.reset_mock()`.
    mock_add_task_method.reset_mock()
    mock_orchestrator.reset_mock()
    # 7.2. CLEAR `captured_tasks_list.clear()`.
    captured_tasks_list.clear()
    # 7.3. (Optional) `job_store.clear()` if imported and used.
    job_store.clear()
    # 7.4. CONFIGURE `mock_orchestrator.side_effect =
    # Exception("Simulated orchestrator internal failure")`.
    mock_orchestrator.side_effect = Exception("Simulated orchestrator internal failure")
    expected_error_msg_failure = (
        "Orchestration error: Simulated orchestrator internal failure"
    )

    # === Phase 8: Action - Submit Job (Failure Scenario) ===

    # 8.1. CALL `response_fail = client.post("/api/tts", json=payload.model_dump())`
    # (payload can be the same).
    response_fail = client.post("/api/tts", json=payload.model_dump())
    # 8.2. PARSE `job_creation_data_fail = response_fail.json()`.
    job_creation_data_fail = response_fail.json()
    # 8.3. STORE `job_id_fail = job_creation_data_fail["job_id"]`.
    job_id_fail = job_creation_data_fail["job_id"]

    # === Phase 9: Assertions - Initial Response (Failure Scenario) ===
    # 9.1. ASSERT `response_fail.status_code == 202`.
    assert response_fail.status_code == 202
    # 9.2. ASSERT `mock_add_task_method.assert_called_once()`.
    # (Called once for this new submission)
    mock_add_task_method.assert_called_once()
    # 9.3. ASSERT `len(captured_tasks_list) == 1`. (A new task was captured).
    assert len(captured_tasks_list) == 1

    # === Phase 10: Action - Manually Run Background Task (Failure Scenario) ===
    # 10.1. AWAIT `await captured_tasks_list[0]()`. (Task runs and should handle
    # the orchestrator's exception).
    await captured_tasks_list[0]()

    # === Phase 11: Assertions - Job Status API (Failure Scenario) ===
    # 11.1.CALL `status_response_failure = client.get(f"/api/tts/status/{job_id_fail}")`
    status_response_failure = client.get(f"/api/tts/status/{job_id_fail}")
    # 11.2. ASSERT `status_response_failure.status_code == 200`.
    assert status_response_failure.status_code == 200
    # 11.3. PARSE `status_data_failure = status_response_failure.json()`.
    status_data_failure = status_response_failure.json()
    # 11.4. ASSERT `status_data_failure["job_id"] == job_id_fail`.
    assert status_data_failure["job_id"] == job_id_fail
    # 11.5. ASSERT `status_data_failure["status"] == "failed"`.
    assert status_data_failure["status"] == "failed"
    # 11.6. ASSERT `status_data_failure["error_message"] == expected_error_msg_failure`.
    assert status_data_failure["error_message"] == expected_error_msg_failure
    # 11.7. ASSERT `status_data_failure.get("outputs")
    # is None or status_data_failure.get("outputs") == {}`.
    assert (
        status_data_failure.get("outputs") is None
        or status_data_failure.get("outputs") == {}
    )
    # 11.8. ASSERT `status_data_failure.get("synthesis_details") is None or
    # status_data_failure.get("synthesis_details") == []`.
    assert (
        status_data_failure.get("synthesis_details") is None
        or status_data_failure.get("synthesis_details") == []
    )
