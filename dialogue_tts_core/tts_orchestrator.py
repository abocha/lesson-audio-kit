# dialogue_tts_core/tts_orchestrator.py

import contextlib
import datetime  # Added import
from datetime import timezone  # Added for timezone-aware datetime
import logging  # Added for logging
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
from .cost_router import (
    ENGINE_TABLE,
    EngineMeta,
    QualityTier,
    _calculate_effective_cost_per_mchar,
    select_engine,
)
from .tts_client import synthesize_speech_line

logger = logging.getLogger(__name__)

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


async def _synthesize_and_log_line(  # noqa: C901
    line_data: dict,
    speaker_specific_config: SpeakerTTSConfig,
    user_id: Optional[str],
    desired_quality_tier_str: Optional[str],
    max_total_job_cost_usd: Optional[float],
    prefer_low_latency: bool,
    prefer_emotion_support: bool,
    prefer_voice_cloning: bool,
    specific_engine_id: Optional[str],
    openai_client: AsyncOpenAI,
    current_job_output_path: str,
    cache_base_dir: str,
    nsfw_api_url_template: Optional[str],
    synthesis_details_list: list[dict[str, Any]],
) -> Optional[str]:
    """Synthesizes a single line and logs details."""
    line_id = line_data.get("id", "unknown_id")
    speaker_name = line_data.get("speaker", "UnknownSpeaker")
    text_to_synthesize = line_data.get("text", "")
    char_len = len(text_to_synthesize)

    line_detail_entry: dict[str, Any] = {
        "id": line_id,
        "speaker": speaker_name,
        "text": text_to_synthesize,
        "text_length": char_len,
        "status": "pending",
        "error": None,
        "path": None,
        "cache_status": "unknown",
        "selected_engine_id": None,
        "selected_engine_provider": None,
        "selected_engine_quality_tier": None,
        "effective_cost_for_line_usd": None,
    }
    synthesis_details_list.append(line_detail_entry)  # Add entry immediately

    if not text_to_synthesize.strip():
        line_detail_entry.update({"status": "skipped", "error": "Empty text line"})
        return None

    selected_engine: Optional[EngineMeta] = None
    try:
        desired_quality = QualityTier.MID
        if desired_quality_tier_str:
            try:
                desired_quality = QualityTier(desired_quality_tier_str)
            except ValueError:
                logger.warning(
                    "Invalid desired_quality_tier_str: %s. Defaulting to MID.",
                    desired_quality_tier_str,
                )
                desired_quality = QualityTier.MID

        if specific_engine_id:
            selected_engine = ENGINE_TABLE.get(specific_engine_id)
            if not selected_engine:
                logger.error(
                    "Specific engine ID '%s' not found in ENGINE_TABLE. "
                    "Falling back to dynamic selection.",
                    specific_engine_id,
                )

        if (
            not selected_engine
        ):  # Fallback if specific_engine_id not provided or not found
            selected_engine = select_engine(
                char_len=char_len,
                desired_quality=desired_quality,
                max_cost_usd_for_job=max_total_job_cost_usd,
                user_id=user_id,
                prefer_low_latency=prefer_low_latency,
                prefer_emotion_support=prefer_emotion_support,
                prefer_voice_cloning=prefer_voice_cloning,
            )

        if not selected_engine:
            raise RuntimeError("No suitable TTS engine could be selected.")

        line_detail_entry["selected_engine_id"] = (
            selected_engine.model_id
        )  # Use model_id as engine_id
        line_detail_entry["selected_engine_provider"] = selected_engine.provider
        line_detail_entry["selected_engine_quality_tier"] = (
            selected_engine.quality_tier.value
        )

        # Calculate effective cost for the line
        effective_cost = _calculate_effective_cost_per_mchar(
            selected_engine,
            char_len,  # Pass selected_engine object directly
        )
        line_detail_entry["effective_cost_for_line_usd"] = (
            effective_cost / 1000
        ) * char_len

        logger.info(
            "Selected engine for line ID '%s': %s "
            "(Provider: %s, Model: %s, Quality: %s)",
            line_id,
            selected_engine.model_id,
            selected_engine.provider,
            selected_engine.model_id,
            selected_engine.quality_tier.value,
        )

    except RuntimeError as e:
        line_detail_entry.update(
            {"status": "failed", "error": f"Engine selection failed: {e}"}
        )
        logger.error("Error selecting engine for line ID '%s': %s", line_id, e)
        return None

    line_voice = speaker_specific_config.voice
    line_speed = 1.0
    if selected_engine.provider == "openai" and selected_engine.model_id in [
        "tts-1",
        "tts-1-hd",
    ]:
        line_speed = (
            speaker_specific_config.speed
            if speaker_specific_config.speed is not None
            else 1.0
        )

    line_instructions: Optional[str] = None
    if selected_engine.supports_emotion:
        line_instructions = speaker_specific_config.custom_instructions

    # Provider Dispatch (Conceptual for now)
    # This part might need to be expanded if other providers are added
    if selected_engine.provider != "openai":
        error_msg = f"Unsupported TTS provider: {selected_engine.provider}"
        logger.error(error_msg)
        line_detail_entry.update({"status": "failed", "error": error_msg})
        return None

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
            model=tts_model_for_client,  # Corrected parameter name
            speed=line_speed,
            instructions=line_instructions,
            cache_base_dir=cache_base_dir,
            nsfw_api_url_template=nsfw_api_url_template,
            line_index=line_id_for_tts_client,
        )

        if synthesized_path_tuple:
            synthesized_path, was_cache_hit = synthesized_path_tuple

        line_detail_entry["cache_status"] = "hit" if was_cache_hit else "miss"

        if (
            synthesized_path
            and os.path.exists(synthesized_path)
            and os.path.getsize(synthesized_path) > 0
        ):
            line_detail_entry.update({"status": "success", "path": synthesized_path})
        else:
            error_msg = "Synthesis failed or produced empty file"
            if not synthesized_path:
                error_msg = "Synthesis function returned no path"
            elif not os.path.exists(synthesized_path):
                error_msg = f"Synthesized file path does not exist: {synthesized_path}"
            elif os.path.getsize(synthesized_path) == 0:
                error_msg = f"Synthesized file is empty: {synthesized_path}"
            line_detail_entry.update({"status": "failed", "error": error_msg})

    except Exception as e:  # noqa: BLE001  # Catch general Exception for synthesis
        line_detail_entry.update(
            {"status": "failed", "error": f"Synthesis exception: {e}"}
        )
        logger.error("Error during synthesis for line ID '%s': %s", line_id, e)
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


def _compile_status_message(all_lines_synthesis_details: list[dict[str, Any]]) -> str:
    """Compiles the final status message based on detailed synthesis results."""
    total_lines = len(all_lines_synthesis_details)
    num_successful = sum(
        1 for d in all_lines_synthesis_details if d.get("status") == "success"
    )
    num_cached = sum(
        1
        for d in all_lines_synthesis_details
        if d.get("status") == "success" and d.get("cache_status") == "hit"
    )
    num_failed = sum(
        1 for d in all_lines_synthesis_details if d.get("status") == "failed"
    )
    num_skipped = sum(
        1 for d in all_lines_synthesis_details if d.get("status") == "skipped"
    )
    num_newly_synthesized = sum(
        1
        for d in all_lines_synthesis_details
        if d.get("status") == "success" and d.get("cache_status") == "miss"
    )

    engine_usage: dict[str, int] = {}
    for d in all_lines_synthesis_details:
        if d.get("status") == "success":
            provider = d.get("selected_engine_provider", "unknown")
            model_id = d.get("selected_engine_id", "unknown")  # Use selected_engine_id
            engine_key = f"{provider} {model_id}"
            engine_usage[engine_key] = engine_usage.get(engine_key, 0) + 1

    message_parts = [f"TTS Job Summary: Total lines: {total_lines}."]
    message_parts.append(f"Successful: {num_successful}.")
    message_parts.append(f"Failed: {num_failed}.")
    message_parts.append(f"Skipped: {num_skipped}.")
    message_parts.append(f"Cache hits: {num_cached}.")
    message_parts.append(f"Newly synthesized: {num_newly_synthesized}.")

    if engine_usage:
        engine_summary = ", ".join(
            [f"{engine} ({count} times)" for engine, count in engine_usage.items()]
        )
        message_parts.append(f"Engines used: {engine_summary}.")

    return " ".join(message_parts).strip()


async def orchestrate_tts_synthesis(
    parsed_script: list[dict],
    global_pause_ms: int,
    resolved_speaker_configs_map: dict[str, SpeakerTTSConfig],
    openai_client: AsyncOpenAI,  # Moved non-default arguments first
    output_directory: str,
    cache_base_dir: str,
    user_id: Optional[str] = None,
    desired_quality_tier_str: Optional[str] = QualityTier.MID.name,  # Changed to .name
    max_total_job_cost_usd: Optional[float] = None,
    prefer_low_latency: bool = False,
    prefer_emotion_support: bool = False,
    prefer_voice_cloning: bool = False,
    specific_engine_id: Optional[str] = None,
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

    # max_line_cost_usd: Optional[float] = None # Removed as per instructions
    # if max_total_job_cost_usd is not None and len(parsed_script) > 0:
    #     max_line_cost_usd = max_total_job_cost_usd / len(parsed_script)

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
            desired_quality_tier_str=desired_quality_tier_str,
            max_total_job_cost_usd=max_total_job_cost_usd,
            prefer_low_latency=prefer_low_latency,
            prefer_emotion_support=prefer_emotion_support,
            prefer_voice_cloning=prefer_voice_cloning,
            specific_engine_id=specific_engine_id,
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

    status_message = _compile_status_message(all_lines_synthesis_details)

    return (
        zip_output_path,
        merged_audio_output_path,
        status_message,
        all_lines_synthesis_details,
    )
