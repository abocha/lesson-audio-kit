# dialogue_tts_core/tts_orchestrator.py

import contextlib
import datetime  # Added import
from datetime import timezone  # Added for timezone-aware datetime
import os
import shutil

# from typing import Optional # Optional is no longer used
import zipfile

# Forward reference for type hint if openai.AsyncOpenAI is not directly imported
from openai import AsyncOpenAI  # Make sure this is appropriate or use TYPE_CHECKING

# OPENAI_VOICES might not be needed directly if voice is in SpeakerTTSConfig
# from .tts_client import OPENAI_VOICES
from .audio_utils import merge_mp3_files

# Assuming SpeakerTTSConfig is in dialogue_tts_core.config_models
from .config_models import SpeakerTTSConfig
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


def _determine_line_synthesis_params(
    speaker_specific_config: SpeakerTTSConfig, tts_global_model: str
) -> tuple[str, float, str | None]:
    """Determines voice, speed, and instructions for a line."""
    line_voice = speaker_specific_config.voice
    line_speed = 1.0
    line_instructions = None

    if tts_global_model in ["tts-1", "tts-1-hd"]:
        line_speed = (
            speaker_specific_config.speed
            if speaker_specific_config.speed is not None
            else 1.0
        )
    if tts_global_model == "gpt-4o-mini-tts":
        line_instructions = speaker_specific_config.custom_instructions
    return line_voice, line_speed, line_instructions


async def _synthesize_and_log_line(
    line_data: dict,
    speaker_specific_config: SpeakerTTSConfig,
    tts_global_model: str,
    openai_client: AsyncOpenAI,
    current_job_output_path: str,
    cache_base_dir: str,  # Added for caching
    nsfw_api_url_template: str | None,
    synthesis_details: list[dict],
) -> str | None:
    """Synthesizes a single line and logs details."""
    line_id = line_data.get("id", "unknown_id")
    speaker_name = line_data.get("speaker", "UnknownSpeaker")
    text_to_synthesize = line_data.get("text", "")

    if not text_to_synthesize.strip():
        synthesis_details.append(
            {
                "id": line_id,
                "speaker": speaker_name,
                "status": "skipped",
                "error": "Empty text line",
            }
        )
        return None

    line_voice, line_speed, line_instructions = _determine_line_synthesis_params(
        speaker_specific_config, tts_global_model
    )

    safe_speaker_name = "".join(
        c if c.isalnum() or c in (" ", "_") else "_" for c in speaker_name
    ).replace(" ", "_")
    line_output_filename = os.path.join(
        current_job_output_path, f"line_{line_id}_{safe_speaker_name}.mp3"
    )

    synthesized_path = None
    try:
        line_id_for_tts_client = -1
        with contextlib.suppress(ValueError):
            line_id_for_tts_client = int(line_id)

        synthesized_path = await synthesize_speech_line(
            client=openai_client,
            text=text_to_synthesize,
            voice=line_voice,
            output_path=line_output_filename,
            model=tts_global_model,
            speed=line_speed,
            instructions=line_instructions,
            cache_base_dir=cache_base_dir,  # Added for caching
            nsfw_api_url_template=nsfw_api_url_template,
            line_index=line_id_for_tts_client,
        )
    except RuntimeError as e:
        print(f"Error during synthesis for line ID '{line_id}': {e}")
        synthesis_details.append(
            {
                "id": line_id,
                "speaker": speaker_name,
                "status": "failed",
                "error": f"Synthesis exception: {e}",
            }
        )
        return None
    return synthesized_path


def _handle_synthesis_result(
    synthesized_path: str | None,
    line_id: str,
    speaker_name: str,
    synthesis_details: list[dict],
    synthesized_line_files: list[str],
) -> None:
    """Handles the result of a single line's synthesis."""
    if (
        synthesized_path
        and os.path.exists(synthesized_path)
        and os.path.getsize(synthesized_path) > 0
    ):
        synthesized_line_files.append(synthesized_path)
        synthesis_details.append(
            {
                "id": line_id,
                "speaker": speaker_name,
                "status": "success",
                "path": synthesized_path,
            }
        )
    else:
        error_msg = "Synthesis failed or produced empty file"
        if not synthesized_path:
            error_msg = "Synthesis function returned no path"
        elif not os.path.exists(synthesized_path):
            error_msg = f"Synthesized file path does not exist: {synthesized_path}"
        elif os.path.getsize(synthesized_path) == 0:
            error_msg = f"Synthesized file is empty: {synthesized_path}"
        synthesis_details.append(
            {
                "id": line_id,
                "speaker": speaker_name,
                "status": "failed",
                "error": error_msg,
            }
        )


def _package_audio_files(
    synthesized_line_files: list[str],
    current_job_output_path: str,
    global_pause_ms: int,
) -> tuple[str | None, str | None]:
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
    synthesis_details: list[dict],
    parsed_script_len: int,
    zip_output_path: str | None,
    merged_audio_output_path: str | None,
    num_successful_files: int,
) -> str:
    """Compiles the final status message."""
    status_message = f"Processed {num_successful_files}/{parsed_script_len} lines. "

    failed_lines = [
        detail for detail in synthesis_details if detail["status"] != "success"
    ]
    if failed_lines:
        status_message += f"{len(failed_lines)} line(s) failed or were skipped. "

    if zip_output_path:
        status_message += "Individual lines ZIP created. "
    elif num_successful_files > 0:
        status_message += "Failed to create ZIP of individual lines. "

    if merged_audio_output_path:
        status_message += "Merged dialogue MP3 created. "
    elif num_successful_files > 0:
        status_message += "Failed to merge dialogue audio. "
    return status_message.strip()


async def orchestrate_tts_synthesis(
    parsed_script: list[dict],
    tts_global_model: str,
    global_pause_ms: int,
    resolved_speaker_configs_map: dict[str, SpeakerTTSConfig],
    openai_client: AsyncOpenAI,
    output_directory: str,
    cache_base_dir: str,  # Added for caching
    nsfw_api_url_template: str | None = None,
) -> tuple[str | None, str | None, str]:
    if not parsed_script:
        return None, None, "Error: Script is empty or contains no processable lines."

    current_job_output_path, error_message = _create_job_directory(output_directory)
    if error_message:
        return None, None, error_message
    assert (
        current_job_output_path is not None
    )  # Ensure path is not None for type checkers

    synthesized_line_files: list[str] = []
    synthesis_details: list[dict] = []

    for line_data in parsed_script:
        line_id = line_data.get("id", "unknown_id")
        speaker_name = line_data.get("speaker", "UnknownSpeaker")
        speaker_specific_config = resolved_speaker_configs_map.get(speaker_name)

        if speaker_specific_config is None:
            print(
                f"Warning: No specific TTS config found for speaker '{speaker_name}' "
                f"for line ID '{line_id}'. Skipping line."
            )
            synthesis_details.append(
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
            tts_global_model=tts_global_model,
            openai_client=openai_client,
            current_job_output_path=current_job_output_path,
            cache_base_dir=cache_base_dir,  # Added for caching
            nsfw_api_url_template=nsfw_api_url_template,
            synthesis_details=synthesis_details,
        )

        _handle_synthesis_result(
            synthesized_path,
            line_id,
            speaker_name,
            synthesis_details,
            synthesized_line_files,
        )

    if not synthesized_line_files:
        if os.path.exists(current_job_output_path):
            try:
                shutil.rmtree(current_job_output_path)
            except OSError as e:
                print(
                    f"Warning: Could not clean up empty job directory "
                    f"'{current_job_output_path}'. Details: {e}"
                )
        return None, None, "Error: No audio lines were successfully synthesized."

    zip_output_path, merged_audio_output_path = _package_audio_files(
        synthesized_line_files, current_job_output_path, global_pause_ms
    )

    status_message = _compile_status_message(
        synthesis_details,
        len(parsed_script),
        zip_output_path,
        merged_audio_output_path,
        len(synthesized_line_files),
    )

    return zip_output_path, merged_audio_output_path, status_message
