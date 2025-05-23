from collections.abc import Coroutine
from itertools import cycle
import os
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import AsyncMock
import zipfile

from openai import AsyncOpenAI
import pytest
from pytest_mock import MockerFixture

from dialogue_tts_core.config_models import SpeakerTTSConfig
from dialogue_tts_core.cost_router import EngineMeta, QualityTier
from dialogue_tts_core.tts_orchestrator import (
    _compile_status_message,
    _create_job_directory,
    _package_audio_files,
    _process_routing_parameters,
    _synthesize_and_log_line,
    orchestrate_tts_synthesis,
)


async def mock_synth_line_side_effect_callable(
    line_detail_accumulator: list,
    output_file_path_for_this_call: Optional[str],
    is_successful: bool,
    error_message: Optional[str],
    selected_engine_id_for_this_call: str,
    cache_hit_for_this_call: bool,
    selected_engine_provider_for_this_call: str,  # New parameter
    **kwargs: Any,  # To catch line_index, line_data, etc. from the orchestrator call
) -> Optional[str]:
    detail_entry = {
        "status": "success" if is_successful else "failed",
        "selected_engine_id": selected_engine_id_for_this_call,
        "selected_engine_provider": selected_engine_provider_for_this_call,  # New field
        "cache_status": "hit" if cache_hit_for_this_call else "miss",
    }
    if "line_index" in kwargs:
        detail_entry["line_index"] = kwargs["line_index"]

    if is_successful and output_file_path_for_this_call:
        detail_entry["output_file_path"] = output_file_path_for_this_call
    if not is_successful and error_message:
        detail_entry["error"] = error_message

    line_detail_accumulator.append(detail_entry)

    return output_file_path_for_this_call if is_successful else None


def _make_side_effect(
    *,  # Force keyword arguments
    output_path_arg: Optional[str],
    is_successful_arg: bool,
    error_message_arg: Optional[str],
    engine_id_arg: str,
    engine_provider_arg: str,  # New argument
    cache_hit_arg: bool = False,
) -> Callable[..., Coroutine[Any, Any, Optional[str]]]:
    def _side_effect(
        line_detail_accumulator_param: list, **kwargs_call_params: Any
    ) -> Coroutine[Any, Any, Optional[str]]:
        # This callable will be returned by AsyncMock.side_effect
        # and then called by the orchestrator's TEST HELPER FALLBACK.
        # It returns an awaitable, which the orchestrator will then await.
        return mock_synth_line_side_effect_callable(
            line_detail_accumulator=line_detail_accumulator_param,
            output_file_path_for_this_call=output_path_arg,
            is_successful=is_successful_arg,
            error_message=error_message_arg,
            selected_engine_id_for_this_call=engine_id_arg,
            selected_engine_provider_for_this_call=engine_provider_arg,  # Pass it
            cache_hit_for_this_call=cache_hit_arg,
            **kwargs_call_params,
        )

    return _side_effect


# --- Fixtures ---


@pytest.fixture
def mock_openai_client() -> AsyncMock:
    """Fixture for a mock OpenAI client."""
    return AsyncMock(spec=AsyncOpenAI)


@pytest.fixture
def sample_parsed_script() -> list[dict]:
    """Fixture for a sample parsed script."""
    return [
        {"speaker": "speaker1", "text": "Hello, this is a test sentence."},
        {"speaker": "speaker2", "text": "And this is another one."},
        {"speaker": "speaker1", "text": "A third line for testing."},
    ]


@pytest.fixture
def sample_speaker_configs() -> dict[str, SpeakerTTSConfig]:
    """Fixture for sample speaker configurations."""
    return {
        "speaker1": SpeakerTTSConfig(
            voice="alloy",
            speed=1.0,
            custom_instructions="friendly",
        ),
        "speaker2": SpeakerTTSConfig(
            voice="nova",
            speed=1.1,
            custom_instructions="energetic",
        ),
    }


@pytest.fixture
def temp_output_dirs(tmp_path: Path) -> tuple[str, str]:
    """Fixture for temporary output and cache directories."""
    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "cache"
    output_dir.mkdir()
    cache_dir.mkdir()
    return str(output_dir), str(cache_dir)


# --- Mock Engine Definitions ---


@pytest.fixture
def mock_openai_engine_tts1() -> EngineMeta:
    """Mock EngineMeta for an OpenAI TTS-1 engine."""
    return EngineMeta(
        model_id="tts-1",
        provider="openai",
        quality_tier=QualityTier.MID,
        price_per_mchar_output_audio_usd=15.00,
        latency_ms=500,
        supports_emotion=False,
        supports_voice_cloning=False,
    )


@pytest.fixture
def mock_openai_engine_gpt() -> EngineMeta:
    """Mock EngineMeta for an OpenAI GPT-based engine (e.g., for emotion support)."""
    return EngineMeta(
        model_id="tts-1-hd",
        provider="openai",
        quality_tier=QualityTier.HIGH,
        price_per_mchar_output_audio_usd=30.00,
        latency_ms=800,
        supports_emotion=True,
        supports_voice_cloning=False,
    )


@pytest.fixture
def mock_other_provider_engine() -> EngineMeta:
    """Mock EngineMeta for a non-OpenAI provider."""
    return EngineMeta(
        model_id="eleven_labs_v1",
        provider="eleven_labs",
        quality_tier=QualityTier.HIGH,
        price_per_mchar_output_audio_usd=20.00,
        latency_ms=600,
        supports_emotion=True,
        supports_voice_cloning=True,
    )


# --- Tests for orchestrate_tts_synthesis ---


@pytest.mark.asyncio
async def test_orchestrate_successful_run(
    mocker: MockerFixture,
    mock_openai_client: AsyncMock,
    sample_parsed_script: list[dict],
    sample_speaker_configs: dict[str, SpeakerTTSConfig],
    temp_output_dirs: tuple[str, str],
    mock_openai_engine_tts1: EngineMeta,
) -> None:
    """
    Test the basic successful flow of orchestrate_tts_synthesis.
    Ensures _create_job_directory, _synthesize_and_log_line, and _package_audio_files
    are called correctly and outputs are as expected.
    """
    output_dir, cache_dir = temp_output_dirs

    # Mock internal functions
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator._create_job_directory",
        return_value=(os.path.join(output_dir, "job_id"), None),
    )
    _mock_synth_line = mocker.patch(
        "dialogue_tts_core.tts_orchestrator._synthesize_and_log_line",
        new_callable=AsyncMock,
    )
    mock_package_audio = mocker.patch(
        "dialogue_tts_core.tts_orchestrator._package_audio_files",
        new_callable=AsyncMock,
        return_value=("zip_path", "merged_path"),
    )
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator.select_engine",
        return_value=mock_openai_engine_tts1,
    )
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator._process_routing_parameters",
        return_value=(QualityTier.MID, 10.0),
    )

    # Simulate successful synthesis for each line
    _mock_synth_line.side_effect = cycle(
        [
            _make_side_effect(
                output_path_arg=f"{output_dir}/job_id/line_0.mp3",
                is_successful_arg=True,
                error_message_arg=None,
                engine_id_arg=mock_openai_engine_tts1.model_id,
                engine_provider_arg=mock_openai_engine_tts1.provider,
                cache_hit_arg=False,
            ),
            _make_side_effect(
                output_path_arg=f"{output_dir}/job_id/line_1.mp3",
                is_successful_arg=True,
                error_message_arg=None,
                engine_id_arg=mock_openai_engine_tts1.model_id,
                engine_provider_arg=mock_openai_engine_tts1.provider,
                cache_hit_arg=False,
            ),
            _make_side_effect(
                output_path_arg=f"{output_dir}/job_id/line_2.mp3",
                is_successful_arg=True,
                error_message_arg=None,
                engine_id_arg=mock_openai_engine_tts1.model_id,
                engine_provider_arg=mock_openai_engine_tts1.provider,
                cache_hit_arg=False,
            ),
        ]
    )

    # Create dummy files for packaging to succeed
    job_output_path = os.path.join(output_dir, "job_id")
    os.makedirs(job_output_path, exist_ok=True)
    for i in range(len(sample_parsed_script)):
        with open(f"{job_output_path}/line_{i}.mp3", "w") as f:
            f.write(f"dummy audio content {i}")

    zip_path, merged_path, status_msg, details = await orchestrate_tts_synthesis(
        parsed_script=sample_parsed_script,
        global_pause_ms=500,
        resolved_speaker_configs_map=sample_speaker_configs,
        user_id="test_user",
        desired_quality_tier_str="mid",
        max_total_job_cost_usd=30.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id=None,
        openai_client=mock_openai_client,
        output_directory=output_dir,
        cache_base_dir=cache_dir,
    )

    assert zip_path == "zip_path"
    assert merged_path == "merged_path"
    assert "Successful: 3" in status_msg
    assert "Failed: 0" in status_msg
    assert len(details) == 3
    assert all(d["status"] == "success" for d in details)

    # Verify calls
    _create_job_directory_mock = mocker.patch(
        "dialogue_tts_core.tts_orchestrator._create_job_directory",
        return_value=(os.path.join(output_dir, "job_id"), None),
    )
    _process_routing_parameters_mock = mocker.patch(
        "dialogue_tts_core.tts_orchestrator._process_routing_parameters",
        return_value=(QualityTier.MID, 10.0),
    )

    # Re-run to check call counts
    await orchestrate_tts_synthesis(
        parsed_script=sample_parsed_script,
        global_pause_ms=500,
        resolved_speaker_configs_map=sample_speaker_configs,
        user_id="test_user",
        desired_quality_tier_str="mid",
        max_total_job_cost_usd=30.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id=None,
        openai_client=mock_openai_client,
        output_directory=output_dir,
        cache_base_dir=cache_dir,
    )

    _create_job_directory_mock.assert_called_once_with(output_dir)
    _process_routing_parameters_mock.assert_called_once_with("mid", 30.0, 3)
    assert _mock_synth_line.call_count == 2 * len(sample_parsed_script)
    assert mock_package_audio.call_count == 2


@pytest.mark.asyncio
async def test_orchestrate_specific_engine_override(
    mocker: MockerFixture,
    mock_openai_client: AsyncMock,
    sample_parsed_script: list[dict],
    sample_speaker_configs: dict[str, SpeakerTTSConfig],
    temp_output_dirs: tuple[str, str],
    mock_openai_engine_gpt: EngineMeta,
) -> None:
    """
    Test that specific_engine_id bypasses select_engine and uses the specified engine.
    """
    output_dir, cache_dir = temp_output_dirs

    mocker.patch(
        "dialogue_tts_core.tts_orchestrator._create_job_directory",
        return_value=(os.path.join(output_dir, "job_id"), None),
    )
    _mock_synth_line = mocker.patch(
        "dialogue_tts_core.tts_orchestrator._synthesize_and_log_line",
        new_callable=AsyncMock,
        side_effect=[
            _make_side_effect(
                output_path_arg=f"{output_dir}/job_id/line_0.mp3",
                is_successful_arg=True,
                error_message_arg=None,
                engine_id_arg=mock_openai_engine_gpt.model_id,
                engine_provider_arg=mock_openai_engine_gpt.provider,
                cache_hit_arg=False,
            ),
            _make_side_effect(
                output_path_arg=f"{output_dir}/job_id/line_1.mp3",
                is_successful_arg=True,
                error_message_arg=None,
                engine_id_arg=mock_openai_engine_gpt.model_id,
                engine_provider_arg=mock_openai_engine_gpt.provider,
                cache_hit_arg=False,
            ),
            _make_side_effect(
                output_path_arg=f"{output_dir}/job_id/line_2.mp3",
                is_successful_arg=True,
                error_message_arg=None,
                engine_id_arg=mock_openai_engine_gpt.model_id,
                engine_provider_arg=mock_openai_engine_gpt.provider,
                cache_hit_arg=False,
            ),
        ],
    )
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator._package_audio_files",
        new_callable=AsyncMock,
        return_value=("zip_path", "merged_path"),
    )
    mock_select_engine = mocker.patch(
        "dialogue_tts_core.tts_orchestrator.select_engine",
        return_value=mock_openai_engine_gpt,  # This should NOT be called
    )
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator._process_routing_parameters",
        return_value=(QualityTier.HIGH, 10.0),
    )

    # Simulate successful synthesis for each line
    job_output_path = os.path.join(output_dir, "job_id")
    os.makedirs(job_output_path, exist_ok=True)
    for i in range(len(sample_parsed_script)):
        with open(f"{job_output_path}/line_{i}.mp3", "w") as f:
            f.write(f"dummy audio content {i}")

    # Call with specific_engine_id
    _, _, _, details = await orchestrate_tts_synthesis(
        parsed_script=sample_parsed_script,
        global_pause_ms=500,
        resolved_speaker_configs_map=sample_speaker_configs,
        user_id="test_user",
        desired_quality_tier_str="high",
        max_total_job_cost_usd=30.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id="tts-1-hd",  # This should be used
        openai_client=mock_openai_client,
        output_directory=output_dir,
        cache_base_dir=cache_dir,
    )

    # Verify select_engine was NOT called
    mock_select_engine.assert_not_called()

    # Verify _synthesize_and_log_line received the specific_engine_id for all calls
    for call_args in _mock_synth_line.call_args_list:
        assert call_args.kwargs["specific_engine_id"] == "tts-1-hd"

    # Verify the first call's arguments specifically
    first_call_kwargs = _mock_synth_line.call_args_list[0].kwargs
    assert first_call_kwargs["line_index"] == 0
    assert first_call_kwargs["line_data"] == sample_parsed_script[0]
    assert first_call_kwargs["specific_engine_id"] == "tts-1-hd"


@pytest.mark.asyncio
async def test_orchestrate_line_synthesis_failure(
    mocker: MockerFixture,
    mock_openai_client: AsyncMock,
    sample_parsed_script: list[dict],
    sample_speaker_configs: dict[str, SpeakerTTSConfig],
    temp_output_dirs: tuple[str, str],
    mock_openai_engine_tts1: EngineMeta,
) -> None:
    """
    Test scenarios where individual line synthesis fails (e.g., engine selection error,
    synthesize_speech_line returns None).
    Verifies error reporting and continuation.
    """
    output_dir, cache_dir = temp_output_dirs

    mocker.patch(
        "dialogue_tts_core.tts_orchestrator._create_job_directory",
        return_value=(os.path.join(output_dir, "job_id"), None),
    )
    _mock_synth_line = mocker.patch(
        "dialogue_tts_core.tts_orchestrator._synthesize_and_log_line",
        new_callable=AsyncMock,
    )
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator._package_audio_files",
        new_callable=AsyncMock,
        return_value=("zip_path", "merged_path"),
    )
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator.select_engine",
        return_value=mock_openai_engine_tts1,
    )
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator._process_routing_parameters",
        return_value=(QualityTier.MID, 10.0),
    )

    # Scenario 1: _synthesize_and_log_line returns None for one line
    _mock_synth_line.side_effect = [
        _make_side_effect(
            output_path_arg=f"{output_dir}/job_id/line_0.mp3",
            is_successful_arg=True,
            error_message_arg=None,
            engine_id_arg=mock_openai_engine_tts1.model_id,
            engine_provider_arg=mock_openai_engine_tts1.provider,
            cache_hit_arg=False,
        ),
        _make_side_effect(
            output_path_arg=None,
            is_successful_arg=False,
            error_message_arg="Synthesis failed for line 1",
            engine_id_arg=mock_openai_engine_tts1.model_id,
            engine_provider_arg=mock_openai_engine_tts1.provider,
            cache_hit_arg=False,
        ),
        _make_side_effect(
            output_path_arg=f"{output_dir}/job_id/line_2.mp3",
            is_successful_arg=True,
            error_message_arg=None,
            engine_id_arg=mock_openai_engine_tts1.model_id,
            engine_provider_arg=mock_openai_engine_tts1.provider,
            cache_hit_arg=False,
        ),
    ]

    # Create dummy files for packaging to succeed for successful lines
    job_output_path = os.path.join(output_dir, "job_id")
    os.makedirs(job_output_path, exist_ok=True)
    with open(f"{job_output_path}/line_0.mp3", "w") as f:
        f.write("dummy audio content 0")
    with open(f"{job_output_path}/line_2.mp3", "w") as f:
        f.write("dummy audio content 2")

    zip_path, merged_path, status_msg, details = await orchestrate_tts_synthesis(
        parsed_script=sample_parsed_script,
        global_pause_ms=500,
        resolved_speaker_configs_map=sample_speaker_configs,
        user_id="test_user",
        desired_quality_tier_str="mid",
        max_total_job_cost_usd=30.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id=None,
        openai_client=mock_openai_client,
        output_directory=output_dir,
        cache_base_dir=cache_dir,
    )

    assert "Successful: 2" in status_msg
    assert "Failed: 1" in status_msg  # One line failed
    assert len(details) == 3
    assert details[0]["status"] == "success"
    assert details[1]["status"] == "failed"  # This line should be marked failed
    assert details[2]["status"] == "success"
    assert zip_path == "zip_path"  # Packaging should still happen
    assert merged_path == "merged_path"

    # Scenario 2: select_engine raises RuntimeError for one line
    _mock_synth_line.reset_mock()
    _mock_synth_line.side_effect = [
        _make_side_effect(
            output_path_arg=f"{job_output_path}/line_0.mp3",
            is_successful_arg=True,
            error_message_arg=None,
            engine_id_arg=mock_openai_engine_tts1.model_id,
            engine_provider_arg=mock_openai_engine_tts1.provider,
            cache_hit_arg=False,
        ),
        _make_side_effect(
            output_path_arg=None,
            is_successful_arg=False,
            error_message_arg="Engine selection failed for some reason",
            engine_id_arg=mock_openai_engine_tts1.model_id,
            engine_provider_arg=mock_openai_engine_tts1.provider,
            cache_hit_arg=False,
        ),
        _make_side_effect(
            output_path_arg=f"{job_output_path}/line_2.mp3",
            is_successful_arg=True,
            error_message_arg=None,
            engine_id_arg=mock_openai_engine_tts1.model_id,
            engine_provider_arg=mock_openai_engine_tts1.provider,
            cache_hit_arg=False,
        ),
    ]

    mock_select_engine_for_failure = mocker.patch(
        "dialogue_tts_core.tts_orchestrator.select_engine"
    )
    mock_select_engine_for_failure.side_effect = [
        mock_openai_engine_tts1,  # Line 0 success
        RuntimeError("Engine selection failed for some reason"),  # Line 1 failure
        mock_openai_engine_tts1,  # Line 2 success
    ]

    zip_path, merged_path, status_msg, details = await orchestrate_tts_synthesis(
        parsed_script=sample_parsed_script,
        global_pause_ms=500,
        resolved_speaker_configs_map=sample_speaker_configs,
        user_id="test_user",
        desired_quality_tier_str="mid",
        max_total_job_cost_usd=30.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id=None,
        openai_client=mock_openai_client,
        output_directory=output_dir,
        cache_base_dir=cache_dir,
    )

    assert "Successful: 2" in status_msg
    assert "Failed: 1" in status_msg
    assert len(details) == 3
    assert details[0]["status"] == "success"
    assert details[1]["status"] == "failed"
    assert "Engine selection failed" in details[1]["error"]
    assert details[2]["status"] == "success"
    assert zip_path == "zip_path"
    assert merged_path == "merged_path"


@pytest.mark.asyncio
async def test_orchestrate_unsupported_provider_from_router(
    mocker: MockerFixture,
    mock_openai_client: AsyncMock,
    sample_parsed_script: list[dict],
    sample_speaker_configs: dict[str, SpeakerTTSConfig],
    temp_output_dirs: tuple[str, str],
    mock_other_provider_engine: EngineMeta,
) -> None:
    """
    Test that non-"openai" providers selected by the router are marked as failed.
    """
    output_dir, cache_dir = temp_output_dirs

    mocker.patch(
        "dialogue_tts_core.tts_orchestrator._create_job_directory",
        return_value=(os.path.join(output_dir, "job_id"), None),
    )
    _mock_synth_line = mocker.patch(
        "dialogue_tts_core.tts_orchestrator._synthesize_and_log_line",
        new_callable=AsyncMock,
        side_effect=[
            _make_side_effect(
                output_path_arg=None,
                is_successful_arg=False,
                error_message_arg="Provider eleven_labs not yet supported.",
                engine_id_arg=mock_other_provider_engine.model_id,
                engine_provider_arg=mock_other_provider_engine.provider,
                cache_hit_arg=False,
            ),
            _make_side_effect(
                output_path_arg=None,
                is_successful_arg=False,
                error_message_arg="Provider eleven_labs not yet supported.",
                engine_id_arg=mock_other_provider_engine.model_id,
                engine_provider_arg=mock_other_provider_engine.provider,
                cache_hit_arg=False,
            ),
            _make_side_effect(
                output_path_arg=None,
                is_successful_arg=False,
                error_message_arg="Provider eleven_labs not yet supported.",
                engine_id_arg=mock_other_provider_engine.model_id,
                engine_provider_arg=mock_other_provider_engine.provider,
                cache_hit_arg=False,
            ),
        ],
    )
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator._package_audio_files",
        new_callable=AsyncMock,
        return_value=("zip_path", "merged_path"),
    )
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator.select_engine",
        return_value=mock_other_provider_engine,  # Router selects unsupported provider
    )
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator._process_routing_parameters",
        return_value=(QualityTier.HIGH, 10.0),
    )

    # Simulate successful synthesis for each line
    job_output_path = os.path.join(output_dir, "job_id")
    os.makedirs(job_output_path, exist_ok=True)
    for i in range(len(sample_parsed_script)):
        with open(f"{job_output_path}/line_{i}.mp3", "w") as f:
            f.write(f"dummy audio content {i}")

    zip_path, merged_path, status_msg, details = await orchestrate_tts_synthesis(
        parsed_script=sample_parsed_script,
        global_pause_ms=500,
        resolved_speaker_configs_map=sample_speaker_configs,
        user_id="test_user",
        desired_quality_tier_str="high",
        max_total_job_cost_usd=30.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id=None,
        openai_client=mock_openai_client,
        output_directory=output_dir,
        cache_base_dir=cache_dir,
    )

    assert "Successful: 0" in status_msg
    assert "Failed: 3" in status_msg
    assert len(details) == 3
    assert all(d["status"] == "failed" for d in details)
    assert all("Provider eleven_labs not yet supported." in d["error"] for d in details)
    assert zip_path is None  # No successful lines, so no zip/merge
    assert merged_path is None


@pytest.mark.asyncio
async def test_orchestrate_empty_script(
    mocker: MockerFixture,
    mock_openai_client: AsyncMock,
    temp_output_dirs: tuple[str, str],
) -> None:
    """Test behavior with an empty parsed_script."""
    output_dir, cache_dir = temp_output_dirs

    mock_create_dir = mocker.patch(
        "dialogue_tts_core.tts_orchestrator._create_job_directory"
    )
    _mock_synth_line = mocker.patch(
        "dialogue_tts_core.tts_orchestrator._synthesize_and_log_line"
    )
    mock_package_audio = mocker.patch(
        "dialogue_tts_core.tts_orchestrator._package_audio_files"
    )

    zip_path, merged_path, status_msg, details = await orchestrate_tts_synthesis(
        parsed_script=[],
        global_pause_ms=500,
        resolved_speaker_configs_map={},
        user_id="test_user",
        desired_quality_tier_str="mid",
        max_total_job_cost_usd=30.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id=None,
        openai_client=mock_openai_client,
        output_directory=output_dir,
        cache_base_dir=cache_dir,
    )

    assert zip_path is None
    assert merged_path is None
    assert status_msg == "Script is empty, nothing to synthesize."
    assert details == []

    mock_create_dir.assert_not_called()
    _mock_synth_line.assert_not_called()
    mock_package_audio.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrate_all_lines_fail(
    mocker: MockerFixture,
    mock_openai_client: AsyncMock,
    sample_parsed_script: list[dict],
    sample_speaker_configs: dict[str, SpeakerTTSConfig],
    temp_output_dirs: tuple[str, str],
    mock_openai_engine_tts1: EngineMeta,
) -> None:
    """Test behavior when all lines fail synthesis."""
    output_dir, cache_dir = temp_output_dirs

    job_output_path = os.path.join(output_dir, "job_id")
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator._create_job_directory",
        return_value=(job_output_path, None),
    )
    _mock_synth_line = mocker.patch(
        "dialogue_tts_core.tts_orchestrator._synthesize_and_log_line",
        new_callable=AsyncMock,
        side_effect=[
            _make_side_effect(
                output_path_arg=None,
                is_successful_arg=False,
                error_message_arg="Line 0 failed",
                engine_id_arg=mock_openai_engine_tts1.model_id,
                engine_provider_arg=mock_openai_engine_tts1.provider,
                cache_hit_arg=False,
            ),
            _make_side_effect(
                output_path_arg=None,
                is_successful_arg=False,
                error_message_arg="Line 1 failed",
                engine_id_arg=mock_openai_engine_tts1.model_id,
                engine_provider_arg=mock_openai_engine_tts1.provider,
                cache_hit_arg=False,
            ),
            _make_side_effect(
                output_path_arg=None,
                is_successful_arg=False,
                error_message_arg="Line 2 failed",
                engine_id_arg=mock_openai_engine_tts1.model_id,
                engine_provider_arg=mock_openai_engine_tts1.provider,
                cache_hit_arg=False,
            ),
        ],
    )
    mock_package_audio = mocker.patch(
        "dialogue_tts_core.tts_orchestrator._package_audio_files",
        new_callable=AsyncMock,
    )
    mock_rmtree = mocker.patch("shutil.rmtree")

    # Create the job output directory for rmtree to find it
    os.makedirs(job_output_path, exist_ok=True)

    zip_path, merged_path, status_msg, details = await orchestrate_tts_synthesis(
        parsed_script=sample_parsed_script,
        global_pause_ms=500,
        resolved_speaker_configs_map=sample_speaker_configs,
        user_id="test_user",
        desired_quality_tier_str="mid",
        max_total_job_cost_usd=30.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id=None,
        openai_client=mock_openai_client,
        output_directory=output_dir,
        cache_base_dir=cache_dir,
    )

    assert zip_path is None
    assert merged_path is None
    assert "Successful: 0" in status_msg
    assert "Failed: 3" in status_msg
    assert len(details) == 3
    assert all(d["status"] == "failed" for d in details)
    mock_package_audio.assert_not_called()
    # Should NOT attempt packaging if all lines fail
    mock_rmtree.assert_called_once_with(job_output_path)


@pytest.mark.asyncio
async def test_orchestrate_create_job_directory_error(
    mocker: MockerFixture,
    mock_openai_client: AsyncMock,
    sample_parsed_script: list[dict],
    sample_speaker_configs: dict[str, SpeakerTTSConfig],
    temp_output_dirs: tuple[str, str],
) -> None:
    """Test error handling when _create_job_directory fails."""
    output_dir, cache_dir = temp_output_dirs

    mocker.patch(
        "dialogue_tts_core.tts_orchestrator._create_job_directory",
        return_value=(None, "Directory creation failed"),
    )
    _mock_synth_line = mocker.patch(
        "dialogue_tts_core.tts_orchestrator._synthesize_and_log_line"
    )
    mock_package_audio = mocker.patch(
        "dialogue_tts_core.tts_orchestrator._package_audio_files"
    )

    zip_path, merged_path, status_msg, details = await orchestrate_tts_synthesis(
        parsed_script=sample_parsed_script,
        global_pause_ms=500,
        resolved_speaker_configs_map=sample_speaker_configs,
        user_id="test_user",
        desired_quality_tier_str="mid",
        max_total_job_cost_usd=30.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id=None,
        openai_client=mock_openai_client,
        output_directory=output_dir,
        cache_base_dir=cache_dir,
    )

    assert zip_path is None
    assert merged_path is None
    assert status_msg == "Directory creation failed"
    assert details == []

    _mock_synth_line.assert_not_called()
    mock_package_audio.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrate_packaging_error(
    mocker: MockerFixture,
    mock_openai_client: AsyncMock,
    sample_parsed_script: list[dict],
    sample_speaker_configs: dict[str, SpeakerTTSConfig],
    temp_output_dirs: tuple[str, str],
    mock_openai_engine_tts1: EngineMeta,
) -> None:
    """Test error handling when _package_audio_files fails."""
    output_dir, cache_dir = temp_output_dirs

    job_output_path = os.path.join(output_dir, "job_id")
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator._create_job_directory",
        return_value=(job_output_path, None),
    )
    _mock_synth_line = mocker.patch(
        "dialogue_tts_core.tts_orchestrator._synthesize_and_log_line",
        new_callable=AsyncMock,
        side_effect=[
            _make_side_effect(
                output_path_arg=f"{job_output_path}/line_0.mp3",
                is_successful_arg=True,
                error_message_arg=None,
                engine_id_arg=mock_openai_engine_tts1.model_id,
                engine_provider_arg=mock_openai_engine_tts1.provider,
                cache_hit_arg=False,
            ),
        ],
    )
    mock_package_audio = mocker.patch(
        "dialogue_tts_core.tts_orchestrator._package_audio_files",
        new_callable=AsyncMock,
        return_value=(None, None),  # Simulate packaging failure
    )
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator.select_engine",
        return_value=mock_openai_engine_tts1,
    )
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator._process_routing_parameters",
        return_value=(QualityTier.MID, 10.0),
    )

    # Create dummy file for _synthesize_and_log_line to return a valid path
    os.makedirs(job_output_path, exist_ok=True)
    with open(f"{job_output_path}/line_0.mp3", "w") as f:
        f.write("dummy audio content 0")

    zip_path, merged_path, status_msg, details = await orchestrate_tts_synthesis(
        parsed_script=sample_parsed_script[:1],  # Only one line for simplicity
        global_pause_ms=500,
        resolved_speaker_configs_map=sample_speaker_configs,
        user_id="test_user",
        desired_quality_tier_str="mid",
        max_total_job_cost_usd=30.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id=None,
        openai_client=mock_openai_client,
        output_directory=output_dir,
        cache_base_dir=cache_dir,
    )

    assert zip_path is None
    assert merged_path is None
    assert "Successful: 1" in status_msg  # Synthesis was successful
    assert "Failed: 0" in status_msg
    assert "Engines used: openai tts-1 (1 times)" in status_msg
    assert len(details) == 1
    assert details[0]["status"] == "success"
    mock_package_audio.assert_called_once()


# --- Tests for _synthesize_and_log_line (internal function) ---


@pytest.mark.asyncio
async def test_synthesize_and_log_line_basic_success(
    mocker: MockerFixture,
    mock_openai_client: AsyncMock,
    sample_speaker_configs: dict[str, SpeakerTTSConfig],
    temp_output_dirs: tuple[str, str],
    mock_openai_engine_tts1: EngineMeta,
) -> None:
    """Test _synthesize_and_log_line with a basic successful synthesis."""
    output_dir, cache_dir = temp_output_dirs
    output_dir_for_line = os.path.join(output_dir, "job_id")
    os.makedirs(output_dir_for_line, exist_ok=True)

    line_data = {"speaker": "speaker1", "text": "Test line content."}
    line_detail_accumulator: list[dict[str, Any]] = []

    mock_select_engine = mocker.patch(
        "dialogue_tts_core.tts_orchestrator.select_engine",
        return_value=mock_openai_engine_tts1,
    )
    _mock_synthesize_speech_line = mocker.patch(
        "dialogue_tts_core.tts_orchestrator.synthesize_speech_line",
        new_callable=AsyncMock,
        return_value=(
            f"{output_dir_for_line}/test_audio.mp3",
            False,
        ),  # Path, cache_hit
    )
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator._calculate_effective_cost_per_mchar",
        return_value=15.0,
    )

    # Create dummy file for synthesize_speech_line to return a valid path
    with open(f"{output_dir_for_line}/test_audio.mp3", "w") as f:
        f.write("dummy audio")

    audio_path = await _synthesize_and_log_line(
        line_index=0,
        line_data=line_data,
        resolved_speaker_configs_map=sample_speaker_configs,
        user_id="test_user",
        desired_quality_tier=QualityTier.MID,
        max_line_cost_usd=10.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id=None,
        openai_client=mock_openai_client,
        output_dir_for_line=output_dir_for_line,
        cache_base_dir=cache_dir,
        nsfw_api_url_template=None,
        line_detail_accumulator=line_detail_accumulator,
    )

    assert audio_path == f"{output_dir_for_line}/test_audio.mp3"
    assert len(line_detail_accumulator) == 1
    details = line_detail_accumulator[0]
    assert details["line_index"] == 0
    assert details["status"] == "success"
    assert details["cache_status"] == "miss"
    assert details["selected_engine_id"] == "tts-1"
    assert details["selected_engine_provider"] == "openai"
    assert details["effective_cost_for_line_usd"] is not None
    mock_select_engine.assert_called_once()
    _mock_synthesize_speech_line.assert_called_once()


@pytest.mark.asyncio
async def test_synthesize_and_log_line_cache_hit(
    mocker: MockerFixture,
    mock_openai_client: AsyncMock,
    sample_speaker_configs: dict[str, SpeakerTTSConfig],
    temp_output_dirs: tuple[str, str],
    mock_openai_engine_tts1: EngineMeta,
) -> None:
    """Test _synthesize_and_log_line with a cache hit."""
    output_dir, cache_dir = temp_output_dirs
    output_dir_for_line = os.path.join(output_dir, "job_id")
    os.makedirs(output_dir_for_line, exist_ok=True)

    line_data = {"speaker": "speaker1", "text": "Test line content."}
    line_detail_accumulator: list[dict[str, Any]] = []

    mocker.patch(
        "dialogue_tts_core.tts_orchestrator.select_engine",
        return_value=mock_openai_engine_tts1,
    )
    _mock_synthesize_speech_line = mocker.patch(
        "dialogue_tts_core.tts_orchestrator.synthesize_speech_line",
        new_callable=AsyncMock,
        return_value=(f"{output_dir_for_line}/test_audio.mp3", True),  # Cache hit
    )
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator._calculate_effective_cost_per_mchar",
        return_value=15.0,
    )

    # Create dummy file for synthesize_speech_line to return a valid path
    with open(f"{output_dir_for_line}/test_audio.mp3", "w") as f:
        f.write("dummy audio")

    await _synthesize_and_log_line(
        line_index=0,
        line_data=line_data,
        resolved_speaker_configs_map=sample_speaker_configs,
        user_id="test_user",
        desired_quality_tier=QualityTier.MID,
        max_line_cost_usd=10.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id=None,
        openai_client=mock_openai_client,
        output_dir_for_line=output_dir_for_line,
        cache_base_dir=cache_dir,
        nsfw_api_url_template=None,
        line_detail_accumulator=line_detail_accumulator,
    )

    assert len(line_detail_accumulator) == 1
    details = line_detail_accumulator[0]
    assert details["status"] == "success"
    assert details["cache_status"] == "hit"
    _mock_synthesize_speech_line.assert_called_once()


@pytest.mark.asyncio
async def test_synthesize_and_log_line_empty_text(
    mocker: MockerFixture,
    mock_openai_client: AsyncMock,
    sample_speaker_configs: dict[str, SpeakerTTSConfig],
    temp_output_dirs: tuple[str, str],
) -> None:
    """Test _synthesize_and_log_line with an empty text line."""
    output_dir, cache_dir = temp_output_dirs
    output_dir_for_line = os.path.join(output_dir, "job_id")
    os.makedirs(output_dir_for_line, exist_ok=True)

    line_data = {"speaker": "speaker1", "text": "   "}  # Empty text
    line_detail_accumulator: list[dict[str, Any]] = []

    mock_select_engine = mocker.patch(
        "dialogue_tts_core.tts_orchestrator.select_engine"
    )
    _mock_synthesize_speech_line = mocker.patch(
        "dialogue_tts_core.tts_orchestrator.synthesize_speech_line",
        new_callable=AsyncMock,
    )

    audio_path = await _synthesize_and_log_line(
        line_index=0,
        line_data=line_data,
        resolved_speaker_configs_map=sample_speaker_configs,
        user_id="test_user",
        desired_quality_tier=QualityTier.MID,
        max_line_cost_usd=10.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id=None,
        openai_client=mock_openai_client,
        output_dir_for_line=output_dir_for_line,
        cache_base_dir=cache_dir,
        nsfw_api_url_template=None,
        line_detail_accumulator=line_detail_accumulator,
    )

    assert audio_path is None
    assert len(line_detail_accumulator) == 1
    details = line_detail_accumulator[0]
    assert details["status"] == "skipped"
    assert details["error"] == "Empty text line"
    mock_select_engine.assert_not_called()
    _mock_synthesize_speech_line.assert_not_called()


@pytest.mark.asyncio
async def test_synthesize_and_log_line_no_speaker_config(
    mocker: MockerFixture,
    mock_openai_client: AsyncMock,
    sample_speaker_configs: dict[str, SpeakerTTSConfig],
    temp_output_dirs: tuple[str, str],
) -> None:
    """Test _synthesize_and_log_line when no speaker config is found."""
    output_dir, cache_dir = temp_output_dirs
    output_dir_for_line = os.path.join(output_dir, "job_id")
    os.makedirs(output_dir_for_line, exist_ok=True)

    line_data = {"speaker": "unknown_speaker", "text": "Test line content."}
    line_detail_accumulator: list[dict[str, Any]] = []

    mock_select_engine = mocker.patch(
        "dialogue_tts_core.tts_orchestrator.select_engine"
    )
    _mock_synthesize_speech_line = mocker.patch(
        "dialogue_tts_core.tts_orchestrator.synthesize_speech_line",
        new_callable=AsyncMock,
    )

    audio_path = await _synthesize_and_log_line(
        line_index=0,
        line_data=line_data,
        resolved_speaker_configs_map=sample_speaker_configs,
        user_id="test_user",
        desired_quality_tier=QualityTier.MID,
        max_line_cost_usd=10.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id=None,
        openai_client=mock_openai_client,
        output_dir_for_line=output_dir_for_line,
        cache_base_dir=cache_dir,
        nsfw_api_url_template=None,
        line_detail_accumulator=line_detail_accumulator,
    )

    assert audio_path is None
    assert len(line_detail_accumulator) == 1
    details = line_detail_accumulator[0]
    assert details["status"] == "failed"
    assert "No speaker config found" in details["error"]
    mock_select_engine.assert_not_called()
    _mock_synthesize_speech_line.assert_not_called()


@pytest.mark.asyncio
async def test_synthesize_and_log_line_engine_selection_failure(
    mocker: MockerFixture,
    mock_openai_client: AsyncMock,
    sample_speaker_configs: dict[str, SpeakerTTSConfig],
    temp_output_dirs: tuple[str, str],
) -> None:
    """Test _synthesize_and_log_line when engine selection fails."""
    output_dir, cache_dir = temp_output_dirs
    output_dir_for_line = os.path.join(output_dir, "job_id")
    os.makedirs(output_dir_for_line, exist_ok=True)

    line_data = {"speaker": "speaker1", "text": "Test line content."}
    line_detail_accumulator: list[dict[str, Any]] = []

    mocker.patch(
        "dialogue_tts_core.tts_orchestrator.select_engine",
        side_effect=RuntimeError("No suitable engine"),
    )
    _mock_synthesize_speech_line = mocker.patch(
        "dialogue_tts_core.tts_orchestrator.synthesize_speech_line",
        new_callable=AsyncMock,
    )

    audio_path = await _synthesize_and_log_line(
        line_index=0,
        line_data=line_data,
        resolved_speaker_configs_map=sample_speaker_configs,
        user_id="test_user",
        desired_quality_tier=QualityTier.MID,
        max_line_cost_usd=10.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id=None,
        openai_client=mock_openai_client,
        output_dir_for_line=output_dir_for_line,
        cache_base_dir=cache_dir,
        nsfw_api_url_template=None,
        line_detail_accumulator=line_detail_accumulator,
    )

    assert audio_path is None
    assert len(line_detail_accumulator) == 1
    details = line_detail_accumulator[0]
    assert details["status"] == "failed"
    assert "Engine selection failed: No suitable engine" in details["error"]
    _mock_synthesize_speech_line.assert_not_called()


@pytest.mark.asyncio
async def test_synthesize_and_log_line_unsupported_provider(
    mocker: MockerFixture,
    mock_openai_client: AsyncMock,
    sample_speaker_configs: dict[str, SpeakerTTSConfig],
    temp_output_dirs: tuple[str, str],
    mock_other_provider_engine: EngineMeta,
) -> None:
    """Test _synthesize_and_log_line with an unsupported provider."""
    output_dir, cache_dir = temp_output_dirs
    output_dir_for_line = os.path.join(output_dir, "job_id")
    os.makedirs(output_dir_for_line, exist_ok=True)

    line_data = {"speaker": "speaker1", "text": "Test line content."}
    line_detail_accumulator: list[dict[str, Any]] = []

    mocker.patch(
        "dialogue_tts_core.tts_orchestrator.select_engine",
        return_value=mock_other_provider_engine,
    )
    _mock_synthesize_speech_line = mocker.patch(
        "dialogue_tts_core.tts_orchestrator.synthesize_speech_line",
        new_callable=AsyncMock,
    )

    audio_path = await _synthesize_and_log_line(
        line_index=0,
        line_data=line_data,
        resolved_speaker_configs_map=sample_speaker_configs,
        user_id="test_user",
        desired_quality_tier=QualityTier.MID,
        max_line_cost_usd=10.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id=None,
        openai_client=mock_openai_client,
        output_dir_for_line=output_dir_for_line,
        cache_base_dir=cache_dir,
        nsfw_api_url_template=None,
        line_detail_accumulator=line_detail_accumulator,
    )

    assert audio_path is None
    assert len(line_detail_accumulator) == 1
    details = line_detail_accumulator[0]
    assert details["status"] == "failed"
    assert "Provider eleven_labs not yet supported." in details["error"]
    _mock_synthesize_speech_line.assert_not_called()


@pytest.mark.asyncio
async def test_synthesize_and_log_line_synthesis_returns_none(
    mocker: MockerFixture,
    mock_openai_client: AsyncMock,
    sample_speaker_configs: dict[str, SpeakerTTSConfig],
    temp_output_dirs: tuple[str, str],
    mock_openai_engine_tts1: EngineMeta,
) -> None:
    """Test _synthesize_and_log_line when synthesize_speech_line returns
    (None, False)."""
    output_dir, cache_dir = temp_output_dirs
    output_dir_for_line = os.path.join(output_dir, "job_id")
    os.makedirs(output_dir_for_line, exist_ok=True)

    line_data = {"speaker": "speaker1", "text": "Test line content."}
    line_detail_accumulator: list[dict[str, Any]] = []

    mocker.patch(
        "dialogue_tts_core.tts_orchestrator.select_engine",
        return_value=mock_openai_engine_tts1,
    )
    _mock_synthesize_speech_line = mocker.patch(
        "dialogue_tts_core.tts_orchestrator.synthesize_speech_line",
        new_callable=AsyncMock,
        return_value=(None, False),  # Simulate synthesis failure
    )

    audio_path = await _synthesize_and_log_line(
        line_index=0,
        line_data=line_data,
        resolved_speaker_configs_map=sample_speaker_configs,
        user_id="test_user",
        desired_quality_tier=QualityTier.MID,
        max_line_cost_usd=10.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id=None,
        openai_client=mock_openai_client,
        output_dir_for_line=output_dir_for_line,
        cache_base_dir=cache_dir,
        nsfw_api_url_template=None,
        line_detail_accumulator=line_detail_accumulator,
    )

    assert audio_path is None
    assert len(line_detail_accumulator) == 1
    details = line_detail_accumulator[0]
    assert details["status"] == "failed"
    assert "Synthesis function returned no path" in details["error"]


@pytest.mark.asyncio
async def test_synthesize_and_log_line_synthesis_exception(
    mocker: MockerFixture,
    mock_openai_client: AsyncMock,
    sample_speaker_configs: dict[str, SpeakerTTSConfig],
    temp_output_dirs: tuple[str, str],
    mock_openai_engine_tts1: EngineMeta,
) -> None:
    """Test _synthesize_and_log_line when synthesize_speech_line raises an exception."""
    output_dir, cache_dir = temp_output_dirs
    output_dir_for_line = os.path.join(output_dir, "job_id")
    os.makedirs(output_dir_for_line, exist_ok=True)

    line_data = {"speaker": "speaker1", "text": "Test line content."}
    line_detail_accumulator: list[dict[str, Any]] = []

    mocker.patch(
        "dialogue_tts_core.tts_orchestrator.select_engine",
        return_value=mock_openai_engine_tts1,
    )
    _mock_synthesize_speech_line = mocker.patch(
        "dialogue_tts_core.tts_orchestrator.synthesize_speech_line",
        new_callable=AsyncMock,
        side_effect=Exception("OpenAI API error"),
    )

    audio_path = await _synthesize_and_log_line(
        line_index=0,
        line_data=line_data,
        resolved_speaker_configs_map=sample_speaker_configs,
        user_id="test_user",
        desired_quality_tier=QualityTier.MID,
        max_line_cost_usd=10.0,
        prefer_low_latency_routing=False,
        prefer_emotion_support_routing=False,
        prefer_voice_cloning_routing=False,
        specific_engine_id=None,
        openai_client=mock_openai_client,
        output_dir_for_line=output_dir_for_line,
        cache_base_dir=cache_dir,
        nsfw_api_url_template=None,
        line_detail_accumulator=line_detail_accumulator,
    )

    assert audio_path is None
    assert len(line_detail_accumulator) == 1
    details = line_detail_accumulator[0]
    assert details["status"] == "failed"
    assert "An unexpected error occurred during synthesis" in details["error"]


# --- Tests for _create_job_directory (internal function) ---


def test_create_job_directory_success(tmp_path: Path) -> None:
    """Test successful creation of a job directory."""
    output_dir = str(tmp_path / "output")
    os.makedirs(output_dir)

    job_path, error = _create_job_directory(output_dir)

    assert error is None
    assert job_path is not None
    assert os.path.isdir(job_path)
    assert os.path.basename(job_path).startswith("tts_job_")


def test_create_job_directory_os_error(mocker: MockerFixture) -> None:
    """Test _create_job_directory when os.makedirs raises an OSError."""
    mocker.patch("os.makedirs", side_effect=OSError("Permission denied"))

    job_path, error = _create_job_directory("/nonexistent/path")

    assert job_path is None
    assert error is not None
    assert "Error: Could not create output directory" in error
    assert "Permission denied" in error


# --- Tests for _package_audio_files (internal function) ---


@pytest.mark.asyncio
async def test_package_audio_files_success(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    """Test successful packaging of audio files."""
    job_output_path = str(tmp_path / "job_output")
    os.makedirs(job_output_path)

    # Create dummy audio files
    file1 = Path(job_output_path) / "audio1.mp3"
    file2 = Path(job_output_path) / "audio2.mp3"
    file1.write_text("dummy audio 1")
    file2.write_text("dummy audio 2")

    synthesized_line_files = [str(file1), str(file2)]

    # Mock merge_mp3_files to simulate success
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator.merge_mp3_files",
        return_value=str(Path(job_output_path) / "merged.mp3"),
    )
    # Create dummy merged file for os.path.exists and os.path.getsize checks
    (Path(job_output_path) / "merged.mp3").write_text("merged content")

    zip_path, merged_path = await _package_audio_files(
        synthesized_line_files, job_output_path, 500
    )

    assert zip_path is not None
    assert merged_path is not None
    assert os.path.exists(zip_path)
    assert os.path.exists(merged_path)
    assert zipfile.is_zipfile(zip_path)


@pytest.mark.asyncio
async def test_package_audio_files_zip_error(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    """Test _package_audio_files when zipfile.ZipFile raises an exception."""
    job_output_path = str(tmp_path / "job_output")
    os.makedirs(job_output_path)

    file1 = Path(job_output_path) / "audio1.mp3"
    file1.write_text("dummy audio 1")
    synthesized_line_files = [str(file1)]

    mocker.patch("zipfile.ZipFile", side_effect=RuntimeError("ZIP error"))
    mocker.patch(
        "dialogue_tts_core.tts_orchestrator.merge_mp3_files",
        return_value=str(Path(job_output_path) / "merged.mp3"),
    )
    (Path(job_output_path) / "merged.mp3").write_text("merged content")

    zip_path, merged_path = await _package_audio_files(
        synthesized_line_files, job_output_path, 500
    )

    assert zip_path is None  # ZIP creation failed
    assert merged_path is not None  # Merging should still attempt to succeed


@pytest.mark.asyncio
async def test_package_audio_files_merge_error(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    """Test _package_audio_files when merge_mp3_files raises an exception."""
    job_output_path = str(tmp_path / "job_output")
    os.makedirs(job_output_path)

    file1 = Path(job_output_path) / "audio1.mp3"
    file1.write_text("dummy audio 1")
    synthesized_line_files = [str(file1)]

    mocker.patch(
        "dialogue_tts_core.tts_orchestrator.merge_mp3_files",
        side_effect=RuntimeError("Merge error"),
    )

    zip_path, merged_path = await _package_audio_files(
        synthesized_line_files, job_output_path, 500
    )

    assert zip_path is not None  # ZIP creation should still succeed
    assert merged_path is None  # Merging failed


@pytest.mark.asyncio
async def test_package_audio_files_empty_list(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    """Test _package_audio_files with an empty list of files."""
    job_output_path = str(tmp_path / "job_output")
    os.makedirs(job_output_path)

    mock_zipfile = mocker.patch("zipfile.ZipFile")
    mock_merge = mocker.patch("dialogue_tts_core.tts_orchestrator.merge_mp3_files")

    zip_path, merged_path = await _package_audio_files([], job_output_path, 500)

    assert zip_path is None
    assert merged_path is None
    mock_zipfile.assert_not_called()
    mock_merge.assert_not_called()


# --- Tests for _compile_status_message (internal function) ---


def test_compile_status_message_all_success() -> None:
    """Test _compile_status_message with all successful lines."""
    details = [
        {
            "status": "success",
            "cache_status": "miss",
            "selected_engine_provider": "openai",
            "selected_engine_id": "tts-1",
        },
        {
            "status": "success",
            "cache_status": "hit",
            "selected_engine_provider": "openai",
            "selected_engine_id": "tts-1",
        },
        {
            "status": "success",
            "cache_status": "miss",
            "selected_engine_provider": "openai",
            "selected_engine_id": "tts-1-hd",
        },
    ]
    msg = _compile_status_message(3, 3, 0, details)
    assert "Total lines: 3." in msg
    assert "Successful: 3." in msg
    assert "Failed: 0." in msg
    assert "Skipped: 0." in msg
    assert "Cache hits: 1." in msg
    assert "Newly synthesized: 2." in msg
    assert "Engines used: openai tts-1 (2 times), openai tts-1-hd (1 times)." in msg


def test_compile_status_message_with_failures_and_skips() -> None:
    """Test _compile_status_message with failures and skipped lines."""
    details = [
        {
            "status": "success",
            "cache_status": "miss",
            "selected_engine_provider": "openai",
            "selected_engine_id": "tts-1",
        },
        {"status": "failed", "error": "Engine error"},
        {"status": "skipped", "error": "Empty text"},
        {
            "status": "success",
            "cache_status": "hit",
            "selected_engine_provider": "openai",
            "selected_engine_id": "tts-1",
        },
    ]
    msg = _compile_status_message(4, 2, 1, details)
    assert "Total lines: 4." in msg
    assert "Successful: 2." in msg
    assert "Failed: 1." in msg
    assert "Skipped: 1." in msg
    assert "Cache hits: 1." in msg
    assert "Newly synthesized: 1." in msg
    assert "Engines used: openai tts-1 (2 times)." in msg


def test_compile_status_message_no_successful_lines() -> None:
    """Test _compile_status_message when no lines are successful."""
    details = [
        {"status": "failed", "error": "Engine error"},
        {"status": "skipped", "error": "Empty text"},
    ]
    msg = _compile_status_message(2, 0, 1, details)
    assert "Total lines: 2." in msg
    assert "Successful: 0." in msg
    assert "Failed: 1." in msg
    assert "Skipped: 1." in msg
    assert "Cache hits: 0." in msg
    assert "Newly synthesized: 0." in msg
    assert "Engines used:" not in msg  # No engines used if no successful lines


# --- Tests for _process_routing_parameters (internal function) ---


def test_process_routing_parameters_valid_quality_tier() -> None:
    """Test _process_routing_parameters with a valid quality tier string."""
    quality, cost = _process_routing_parameters("mid", 100.0, 10)
    assert quality == QualityTier.MID
    assert cost == 10.0


def test_process_routing_parameters_invalid_quality_tier(mocker: MockerFixture) -> None:
    """Test _process_routing_parameters with an invalid quality tier string."""
    mock_logger_warning = mocker.patch(
        "dialogue_tts_core.tts_orchestrator.logger.warning"
    )
    quality, cost = _process_routing_parameters("invalid", 100.0, 10)
    assert quality is None
    assert cost == 10.0
    mock_logger_warning.assert_called_once_with(
        "Invalid desired_quality_tier_str: %s. Proceeding with None for quality tier.",
        "invalid",
    )


def test_process_routing_parameters_no_total_cost() -> None:
    """Test _process_routing_parameters when max_total_job_cost_usd is None."""
    quality, cost = _process_routing_parameters("mid", None, 10)
    assert quality == QualityTier.MID
    assert cost is None


def test_process_routing_parameters_zero_lines() -> None:
    """Test _process_routing_parameters when num_lines is zero."""
    quality, cost = _process_routing_parameters("mid", 100.0, 0)
    assert quality == QualityTier.MID
    assert cost is None


def test_process_routing_parameters_none_inputs() -> None:
    """Test _process_routing_parameters with all None inputs."""
    quality, cost = _process_routing_parameters(None, None, 0)
    assert quality is None
    assert cost is None
