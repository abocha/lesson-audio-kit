# dialogue_tts_core/tts_orchestrator.py

import contextlib
import datetime  # Added import
from datetime import timezone  # Added for timezone-aware datetime
import os
import shutil
from typing import Any, Optional
import zipfile

# Forward reference for type hint if openai.AsyncOpenAI is not directly imported
from openai import AsyncOpenAI  # Make sure this is appropriate or use TYPE_CHECKING

# OPENAI_VOICES might not be needed directly if voice is in SpeakerTTSConfig
# from .tts_client import OPENAI_VOICES
from .audio_utils import merge_mp3_files

# Assuming SpeakerTTSConfig is in dialogue_tts_core.config_models
from .config_models import SpeakerTTSConfig
from .cost_router import EngineMeta, QualityTier, select_engine
from .tts_client import synthesize_speech_line

# Note: The logic for _resolve_instructions_for_line is simplified here.
# It's assumed that SpeakerTTSConfig.custom_instructions will contain the final,
# resolved instruction string if any is needed, potentially prepared by the calling
# layer (e.g., API or UI backend)
# by interpreting 'vibe' and 'custom_instructions' fields.


def _create_job_directory(
    output_directory: str,
) -> tuple[str | None, str | None]:
    """Creates a unique job directory within the output directory."""
    timestamp = datetime.datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    job_id = f"tts_job_{os.urandom(4).hex()}_{timestamp}"
    current_job_output_path = os.path.join(output_directory, job_id)

    try:
        os.makedirs(current_job_output_path, exist_ok=True)
        return current_job_output_path, None
    except OSError as e:
        error_message = (
            f"Error: Could not create output directory '{current_job_output_path}'. "
            f"Details: {e}"
        )
        return None, error_message


async def _synthesize_and_log_line(
    line_data: dict,
    speaker_specific_config: SpeakerTTSConfig,
    user_id: Optional[str],
    desired_quality_tier: QualityTier,
    max_line_cost_usd: Optional[float],
    prefer_low_latency_routing: bool,
    prefer_emotion_support_routing: bool,
    openai_client: AsyncOpenAI,
    current_job_output_path: str,
    cache_base_dir: str,
    nsfw_api_url_template: Optional[str],
    synthesis_details_list: list[dict],
) -> Optional[str]:
    """Synthesizes a single line and logs details."""
    line_id = line_data.get("id", "unknown_id")
    speaker_name = line_data.get("speaker", "UnknownSpeaker")
    text_to_synthesize = line_data.get("text", "")

    line_detail: dict[str, Any] = {
        "id": line_id,
        "speaker": speaker_name,
        "text": text_to_synthesize,
        "status": "unknown",
        "error": None,
        "path": None,
        "cache_status": None,
        "engine_provider": None,
        "engine_model_id": None,
        "cost_usd": None,
    }

    if not text_to_synthesize.strip():
        line_detail.update({"status": "skipped", "error": "Empty text line"})
        synthesis_details_list.append(line_detail)
        return None

    try:
        selected_engine: EngineMeta = select_engine(
            char_len=len(text_to_synthesize),
            desired_quality=desired_quality_tier,
            max_cost_usd_for_job=max_line_cost_usd,
            prefer_low_latency=prefer_low_latency_routing,
            prefer_emotion_support=prefer_emotion_support_routing,
            user_id=user_id,
        )
        line_detail["engine_provider"] = selected_engine.provider
        line_detail["engine_model_id"] = selected_engine.model_id
    except RuntimeError as e:
        line_detail.update(
            {"status": "failed", "error": f"Engine selection failed: {e}"}
        )
        synthesis_details_list.append(line_detail)
        print(f"Error selecting engine for line ID '{line_id}': {e}")
        return None

    line_voice = speaker_specific_config.voice
    line_speed = (
        speaker_specific_config.speed
        if speaker_specific_config.speed is not None
        else 1.0
    )
    line_instructions = speaker_specific_config.custom_instructions

    tts_model_for_client = selected_engine.model_id

    safe_speaker_name = "".join(
        c if c.isalnum() or c in (" ", "_") else "_" for c in speaker_name
    ).replace(" ", "_")
    line_output_filename = os.path.join(
        current_job_output_path, f"line_{line_id}_{safe_speaker_name}.mp3"
    )

    synthesized_path: Optional[str] = None
    was_cache_hit: bool = False

    try:
        line_id_for_tts_client = -1
        with contextlib.suppress(ValueError):
            line_id_for_tts_client = int(line_id)

        synthesized_path_tuple = await synthesize_speech_line(
            client=openai_client,
            text=text_to_synthesize,
            voice=line_voice,
            output_path=line_output_filename,
            model=tts_model_for_client,
            speed=line_speed,
            instructions=line_instructions,
            cache_base_dir=cache_base_dir,
            nsfw_api_url_template=nsfw_api_url_template,
            line_index=line_id_for_tts_client,
        )

        if synthesized_path_tuple:
            synthesized_path, was_cache_hit = synthesized_path_tuple

        line_detail["cache_status"] = "hit" if was_cache_hit else "miss"

        if (
            synthesized_path
            and os.path.exists(synthesized_path)
            and os.path.getsize(synthesized_path) > 0
        ):
            line_detail.update({"status": "success", "path": synthesized_path})
        else:
            error_msg = "Synthesis failed or produced empty file"
            if not synthesized_path:
                error_msg = "Synthesis function returned no path"
            elif not os.path.exists(synthesized_path):
                error_msg = f"Synthesized file path does not exist: {synthesized_path}"
            elif os.path.getsize(synthesized_path) == 0:
                error_msg = f"Synthesized file is empty: {synthesized_path}"
            line_detail.update({"status": "failed", "error": error_msg})

    except RuntimeError as e:
        line_detail.update({"status": "failed", "error": f"Synthesis exception: {e}"})
        print(f"Error during synthesis for line ID '{line_id}': {e}")
    finally:
        synthesis_details_list.append(line_detail)
    return synthesized_path


def _package_audio_files(
    synthesized_line_files: list[str],
    current_job_output_path: str,
    global_pause_ms: int,
) -> tuple[Optional[str], Optional[str]]:
    """Packages synthesized audio files into a ZIP and a merged MP3."""
    zip_output_path = None
    if synthesized_line_files:
        zip_filename = os.path.join(current_job_output_path, "dialogue_lines.zip")
        try:
            with zipfile.ZipFile(
                zip_filename, "w", zipfile.ZIP_DEFLATED
            ) as zf:  # Added compression
                for file_path in synthesized_line_files:
                    zf.write(file_path, os.path.basename(file_path))
            zip_output_path = zip_filename
        except RuntimeError as e:
            print(f"Error creating ZIP file: {e}")
            zip_output_path = None

    merged_audio_output_path = None
    if synthesized_line_files:
        merged_filename = os.path.join(current_job_output_path, "merged_dialogue.mp3")
        try:
            merged_audio_output_path_attempt = merge_mp3_files(
                file_paths=synthesized_line_files,
                output_filename=merged_filename,
                pause_ms=global_pause_ms,
            )
            if (
                merged_audio_output_path_attempt
                and os.path.exists(merged_audio_output_path_attempt)
                and os.path.getsize(merged_audio_output_path_attempt) > 0
            ):
                merged_audio_output_path = merged_audio_output_path_attempt
            else:
                print(
                    "Error: Failed to merge audio files or merged file is invalid/empty"
                )
                merged_audio_output_path = None
        except RuntimeError as e:
            print(f"Error during merging audio files: {e}")
            merged_audio_output_path = None
    return zip_output_path, merged_audio_output_path


def _compile_status_message(
    synthesis_details: list[dict[str, Any]],
    total_lines: int,
    zip_file_path: Optional[str],
    merged_file_path: Optional[str],
    num_actually_synthesized_files: int,
) -> str:
    """Compiles the final status message."""
    num_successful = sum(1 for d in synthesis_details if d.get("status") == "success")
    num_cached = sum(
        1
        for d in synthesis_details
        if d.get("status") == "success" and d.get("cache_status") == "hit"
    )
    num_failed = sum(1 for d in synthesis_details if d.get("status") == "failed")
    num_skipped = sum(1 for d in synthesis_details if d.get("status") == "skipped")
    num_newly_synthesized = sum(
        1
        for d in synthesis_details
        if d.get("status") == "success" and d.get("cache_status") == "miss"
    )

    message_parts = [f"TTS Job Summary: Total Lines: {total_lines}"]
    if num_successful > 0:
        message_parts.append(
            f"Successful: {num_successful} (Cached: {num_cached}, "
            f"Newly Synthesized: {num_newly_synthesized})"
        )
    if num_failed > 0:
        message_parts.append(f"Failed: {num_failed}")
    if num_skipped > 0:
        message_parts.append(f"Skipped: {num_skipped}")

    if zip_file_path:
        message_parts.append("Individual lines ZIP created.")
    elif num_actually_synthesized_files > 0:
        message_parts.append("Failed to create ZIP of individual lines.")

    if merged_file_path:
        message_parts.append("Merged dialogue MP3 created.")
    elif num_actually_synthesized_files > 0:
        message_parts.append("Failed to merge dialogue audio.")

    return " ".join(message_parts).strip()


async def orchestrate_tts_synthesis(
    parsed_script: list[dict],
    global_pause_ms: int,
    resolved_speaker_configs_map: dict[str, SpeakerTTSConfig],
    user_id: Optional[str],
    desired_quality_tier: QualityTier,
    max_total_job_cost_usd: Optional[float],
    prefer_low_latency_routing: bool,
    prefer_emotion_support_routing: bool,
    openai_client: AsyncOpenAI,
    output_directory: str,
    cache_base_dir: str,
    nsfw_api_url_template: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], str, list[dict[str, Any]]]:
    if not parsed_script:
        return (
            None,
            None,
            "Error: Script is empty or contains no processable lines.",
            [],
        )

    current_job_output_path, error_message = _create_job_directory(output_directory)
    if error_message:
        return None, None, error_message, []
    assert current_job_output_path is not None

    synthesized_line_files: list[str] = []
    all_lines_synthesis_details: list[dict[str, Any]] = []

    max_line_cost_usd: Optional[float] = None
    if max_total_job_cost_usd is not None and len(parsed_script) > 0:
        max_line_cost_usd = max_total_job_cost_usd / len(parsed_script)

    for line_data in parsed_script:
        speaker_name = line_data.get("speaker", "UnknownSpeaker")
        speaker_specific_config = resolved_speaker_configs_map.get(speaker_name)

        if speaker_specific_config is None:
            line_id = line_data.get("id", "unknown_id")
            print(
                f"Warning: No specific TTS config found for speaker '{speaker_name}' "
                f"for line ID '{line_id}'. Skipping line."
            )
            all_lines_synthesis_details.append(
                {
                    "id": line_id,
                    "speaker": speaker_name,
                    "status": "skipped",
                    "error": "No speaker config",
                }
            )
            continue

        synthesized_path = await _synthesize_and_log_line(
            line_data=line_data,
            speaker_specific_config=speaker_specific_config,
            user_id=user_id,
            desired_quality_tier=desired_quality_tier,
            max_line_cost_usd=max_line_cost_usd,
            prefer_low_latency_routing=prefer_low_latency_routing,
            prefer_emotion_support_routing=prefer_emotion_support_routing,
            openai_client=openai_client,
            current_job_output_path=current_job_output_path,
            cache_base_dir=cache_base_dir,
            nsfw_api_url_template=nsfw_api_url_template,
            synthesis_details_list=all_lines_synthesis_details,
        )

        if synthesized_path:
            synthesized_line_files.append(synthesized_path)

    if not synthesized_line_files:
        if os.path.exists(current_job_output_path):
            try:
                shutil.rmtree(current_job_output_path)
            except OSError as e:
                print(
                    f"Warning: Could not clean up empty job directory "
                    f"'{current_job_output_path}'. Details: {e}"
                )
        error_msg = "Error: No audio lines were successfully synthesized."
        return None, None, error_msg, all_lines_synthesis_details

    zip_output_path, merged_audio_output_path = _package_audio_files(
        synthesized_line_files, current_job_output_path, global_pause_ms
    )

    status_message = _compile_status_message(
        all_lines_synthesis_details,
        len(parsed_script),
        zip_output_path,
        merged_audio_output_path,
        len(synthesized_line_files),
    )

    return (
        zip_output_path,
        merged_audio_output_path,
        status_message,
        all_lines_synthesis_details,
    )
