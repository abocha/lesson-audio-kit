from collections.abc import Generator
import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from dialogue_tts_core.config_models import (
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
def mock_core_services() -> Any:
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
    _mock_core_services: Any, mock_background_tasks: MagicMock
) -> None:
    """Test TTS job submission with all routing parameters."""
    payload_with_routing = TTSRequestPayload(
        script_text="Speaker1: Hello world with routing.",
        tts_global_model="tts-1",
        speaker_config_method="global",
        global_speaker_config=SpeakerTTSConfig(voice="alloy"),
        user_id="test_user_123",
        desired_quality_tier="premium",
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
    assert called_args[0][0] == run_tts_orchestration_task
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
    _mock_core_services: Any, mock_background_tasks: MagicMock
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
    assert called_args[0][0] == run_tts_orchestration_task
    assert called_kwargs.get("user_id") == payload_with_routing.user_id
    assert called_kwargs.get("desired_quality_tier_str") is None  # Not provided
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
    _mock_core_services: Any, mock_background_tasks: MagicMock
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
    assert called_args[0][0] == run_tts_orchestration_task
    assert called_kwargs.get("user_id") is None
    assert called_kwargs.get("desired_quality_tier_str") is None
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
