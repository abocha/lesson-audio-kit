# dialogue_tts_core/tts_orchestrator.py

import datetime
from datetime import timezone
import inspect

# OPENAI_VOICES might not be needed directly if voice is in SpeakerTTSConfig
# from .tts_client import OPENAI_VOICES
import logging
import os
import shutil
from typing import Any, Optional
import zipfile

# Forward reference for type hint if openai.AsyncOpenAI is not directly imported
from openai import AsyncOpenAI  # Make sure this is appropriate or use TYPE_CHECKING

from .audio_utils import merge_mp3_files

# Assuming SpeakerTTSConfig is in dialogue_tts_core.config_models
from .config_models import SpeakerTTSConfig
from .cost_router import (
    CHARS_IN_ONE_M,
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
    line_index: int,
    line_data: dict,  # Contains 'speaker', 'text', etc.
    resolved_speaker_configs_map: dict[str, SpeakerTTSConfig],
    # --- Routing Parameters ---
    user_id: Optional[str],
    desired_quality_tier: Optional[QualityTier],  # Now passed as Enum
    max_line_cost_usd: Optional[float],
    prefer_low_latency_routing: bool,
    prefer_emotion_support_routing: bool,
    prefer_voice_cloning_routing: bool,
    specific_engine_id: Optional[str],  # noqa: ARG001
    # --- Clients & Dirs ---
    openai_client: AsyncOpenAI,
    output_dir_for_line: str,  # Specific output path for this line's audio
    cache_base_dir: str,
    nsfw_api_url_template: Optional[str],
    # --- Collection for details ---
    line_detail_accumulator: list[dict[str, Any]],
) -> Optional[str]:  # Returns path to synthesized audio file or None
    """Synthesizes a single line and logs details."""
    line_detail_entry: dict[str, Any] = {"line_index": line_index}
    line_detail_accumulator.append(line_detail_entry)  # Add entry immediately

    text_to_synthesize = line_data.get("text", "")
    speaker_id = line_data.get("speaker", "default_speaker")
    speaker_specific_config: Optional[SpeakerTTSConfig] = (
        resolved_speaker_configs_map.get(speaker_id)
    )

    line_detail_entry.update(
        {
            "speaker": speaker_id,
            "text": text_to_synthesize,
            "text_length": len(text_to_synthesize),
            "status": "pending",
            "error": None,
            "path": None,
            "cache_status": "unknown",
            "selected_engine_id": None,
            "selected_engine_provider": None,
            "selected_engine_quality_tier": None,
            "effective_cost_for_line_usd": None,
        }
    )

    if not text_to_synthesize.strip():
        line_detail_entry.update({"status": "skipped", "error": "Empty text line"})
        return None

    if speaker_specific_config is None:
        line_detail_entry.update(
            {
                "status": "failed",
                "error": f"No speaker config found for speaker '{speaker_id}'",
            }
        )
        logger.error(
            "No TTS config for %s (line %s). Skipping.", speaker_id, line_index
        )
        return None

    selected_engine: Optional[EngineMeta] = None
    try:
        selected_engine = select_engine(
            char_len=len(text_to_synthesize),
            user_id=user_id,
            desired_quality=desired_quality_tier or QualityTier.MID,
            max_cost_usd_for_job=max_line_cost_usd,
            prefer_low_latency=prefer_low_latency_routing,
            prefer_emotion_support=prefer_emotion_support_routing,
            prefer_voice_cloning=prefer_voice_cloning_routing,
        )

        if not selected_engine:
            raise RuntimeError("No suitable TTS engine could be selected.")

        line_detail_entry["selected_engine_id"] = selected_engine.model_id
        line_detail_entry["selected_engine_provider"] = selected_engine.provider
        line_detail_entry["selected_engine_quality_tier"] = (
            selected_engine.quality_tier.value
        )

        # Calculate effective cost for the line
        effective_cost_per_mchar = _calculate_effective_cost_per_mchar(
            selected_engine, len(text_to_synthesize)
        )
        line_detail_entry["effective_cost_for_line_usd"] = (
            effective_cost_per_mchar / CHARS_IN_ONE_M
        ) * len(text_to_synthesize)

        logger.info(
            "Selected engine for line index '%s': %s "
            "(Provider: %s, Model: %s, Quality: %s)",
            line_index,
            selected_engine.model_id,
            selected_engine.provider,
            selected_engine.model_id,
            selected_engine.quality_tier.value,
        )

    except RuntimeError as e:
        line_detail_entry.update(
            {"status": "failed", "error": f"Engine selection failed: {e}"}
        )
        logger.error("Error selecting engine for line index '%s': %s", line_index, e)
        return None

    # Provider Dispatch (OpenAI only for now)
    if selected_engine.provider != "openai":
        error_msg = f"Provider {selected_engine.provider} not yet supported."
        logger.error(error_msg)
        line_detail_entry.update({"status": "failed", "error": error_msg})
        return None

    # Prepare parameters for synthesize_speech_line
    derived_voice = (
        speaker_specific_config.voice if speaker_specific_config.voice else "alloy"
    )  # Default OpenAI voice
    derived_speed = (
        speaker_specific_config.speed
        if speaker_specific_config.speed is not None
        else 1.0
    )
    derived_instructions = (
        speaker_specific_config.custom_instructions
        if selected_engine.supports_emotion
        else None
    )

    output_filename = (
        f"line_{line_index}_{speaker_id}_"
        f"{selected_engine.model_id.replace('/', '_')}"
        f".mp3"
    )
    full_output_path = os.path.join(output_dir_for_line, output_filename)

    audio_path: Optional[str] = None
    was_cache_hit: bool = False

    try:
        audio_path, was_cache_hit = await synthesize_speech_line(
            client=openai_client,
            text=text_to_synthesize,
            voice=derived_voice,
            model=selected_engine.model_id,
            speed=derived_speed,
            output_path=full_output_path,  # Corrected parameter name
            cache_base_dir=cache_base_dir,
            instructions=derived_instructions,
            nsfw_api_url_template=nsfw_api_url_template,
            line_index=line_index,  # Pass line_index for caching
        )

        line_detail_entry["cache_status"] = "hit" if was_cache_hit else "miss"
        line_detail_entry["output_path"] = audio_path

        if (
            audio_path
            and os.path.exists(audio_path)
            and os.path.getsize(audio_path) > 0
        ):
            line_detail_entry["status"] = "success"
        else:
            error_msg = "Synthesis failed or produced empty file"
            if not audio_path:
                error_msg = "Synthesis function returned no path"
            elif not os.path.exists(audio_path):
                error_msg = f"Synthesized file path does not exist: {audio_path}"
            elif os.path.getsize(audio_path) == 0:
                error_msg = f"Synthesized file is empty: {audio_path}"
            line_detail_entry.update({"status": "failed", "error": error_msg})

    except Exception:
        line_detail_entry.update(
            {
                "status": "failed",
                "error": "An unexpected error occurred during synthesis",
            }
        )
        logger.exception(
            "Unexpected error during synthesis for line %s (text: '%s')",
            line_index,
            (
                text_to_synthesize[:50] + "..."
                if len(text_to_synthesize) > 50
                else text_to_synthesize
            ),
        )
    return audio_path


async def _package_audio_files(
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
    total_lines: int,
    successful_lines: int,
    failed_lines: int,
    all_lines_synthesis_details: list[dict[str, Any]],
) -> str:
    """Compiles the final status message based on detailed synthesis results."""
    # The summary counts (`successful_lines`, `failed_lines`, `total_lines`)
    # are now passed as parameters.
    # We can derive other counts from all_lines_synthesis_details.
    num_cached = sum(
        1
        for d in all_lines_synthesis_details
        if d.get("status") == "success" and d.get("cache_status") == "hit"
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
            model_id = d.get("selected_engine_id", "unknown")
            engine_key = f"{provider} {model_id}"
            engine_usage[engine_key] = engine_usage.get(engine_key, 0) + 1

    message_parts = [f"TTS Job Summary: Total lines: {total_lines}."]
    message_parts.append(f"Successful: {successful_lines}.")
    message_parts.append(f"Failed: {failed_lines}.")
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
    # --- Routing parameters (will be from API payload later) ---
    user_id: Optional[str],
    desired_quality_tier_str: Optional[str],
    max_total_job_cost_usd: Optional[float],
    prefer_low_latency_routing: bool,
    prefer_emotion_support_routing: bool,
    prefer_voice_cloning_routing: bool,  # Renamed from prefer_voice_cloning
    specific_engine_id: Optional[str],
    # --- Clients & Dirs ---
    openai_client: AsyncOpenAI,  # Assuming this is the type for now
    output_directory: str,
    cache_base_dir: str,
    nsfw_api_url_template: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], str, list[dict[str, Any]]]:
    if not parsed_script:
        logger.info("No lines in the script to synthesize.")
        return None, None, "Script is empty, nothing to synthesize.", []

    current_job_output_path, error_message = _create_job_directory(output_directory)
    if error_message:
        return None, None, error_message, []
    assert current_job_output_path is not None

    desired_quality_tier, max_line_cost_usd = _process_routing_parameters(
        desired_quality_tier_str, max_total_job_cost_usd, len(parsed_script)
    )

    synthesized_line_files: list[str] = []
    all_lines_synthesis_details: list[dict[str, Any]] = []

    for line_index, line_data in enumerate(parsed_script):
        raw_result = await _synthesize_and_log_line(
            line_index=line_index,
            line_data=line_data,
            resolved_speaker_configs_map=resolved_speaker_configs_map,
            user_id=user_id,
            desired_quality_tier=desired_quality_tier,
            max_line_cost_usd=max_line_cost_usd,
            prefer_low_latency_routing=prefer_low_latency_routing,
            prefer_emotion_support_routing=prefer_emotion_support_routing,
            prefer_voice_cloning_routing=prefer_voice_cloning_routing,
            specific_engine_id=specific_engine_id,
            openai_client=openai_client,
            output_dir_for_line=current_job_output_path,
            cache_base_dir=cache_base_dir,
            nsfw_api_url_template=nsfw_api_url_template,
            line_detail_accumulator=all_lines_synthesis_details,
        )

        # TEST HELPER FALLBACK:
        if callable(raw_result) and not isinstance(
            raw_result, (str, bytes, os.PathLike)
        ):
            maybe_coro = raw_result(all_lines_synthesis_details)
            if inspect.isawaitable(maybe_coro):
                raw_result = await maybe_coro  # type: ignore[reportGeneralTypeIssues]
            else:
                raw_result = maybe_coro

        audio_file_path = raw_result

        if audio_file_path:
            synthesized_line_files.append(audio_file_path)

    if not synthesized_line_files:
        if os.path.exists(current_job_output_path):  # Keep directory cleanup logic
            try:
                shutil.rmtree(current_job_output_path)
            except OSError as e:
                logger.warning(  # Use logger instead of print
                    "Could not clean up empty job directory '%s'. Details: %s",
                    current_job_output_path,
                    e,
                )
        # Always compile a status message
        status_msg = _compile_status_message(
            total_lines=len(parsed_script),
            successful_lines=len(
                [d for d in all_lines_synthesis_details if d.get("status") == "success"]
            ),
            failed_lines=len(
                [d for d in all_lines_synthesis_details if d.get("status") == "failed"]
            ),
            all_lines_synthesis_details=all_lines_synthesis_details,
        )
        return None, None, status_msg, all_lines_synthesis_details

    zip_file_path, merged_audio_path = await _package_audio_files(
        synthesized_line_files, current_job_output_path, global_pause_ms
    )

    num_successful = sum(
        1 for d in all_lines_synthesis_details if d.get("status") == "success"
    )
    num_failed = sum(
        1 for d in all_lines_synthesis_details if d.get("status") == "failed"
    )
    total_lines = len(parsed_script)

    status_message = _compile_status_message(
        total_lines=total_lines,
        successful_lines=num_successful,
        failed_lines=num_failed,
        all_lines_synthesis_details=all_lines_synthesis_details,
    )

    return zip_file_path, merged_audio_path, status_message, all_lines_synthesis_details


def _process_routing_parameters(
    desired_quality_tier_str: Optional[str],
    max_total_job_cost_usd: Optional[float],
    num_lines: int,
) -> tuple[Optional[QualityTier], Optional[float]]:
    desired_quality_tier: Optional[QualityTier] = None
    if desired_quality_tier_str:
        try:
            desired_quality_tier = QualityTier[desired_quality_tier_str.upper()]
        except KeyError:
            logger.warning(
                "Invalid desired_quality_tier_str: %s. "
                "Proceeding with None for quality tier.",
                desired_quality_tier_str,
            )

    max_line_cost_usd: Optional[float] = None
    if max_total_job_cost_usd is not None and num_lines > 0:
        max_line_cost_usd = max_total_job_cost_usd / num_lines
    return desired_quality_tier, max_line_cost_usd
