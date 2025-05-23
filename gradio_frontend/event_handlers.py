# FILE: esl-dialogue-tts/event_handlers.py
import datetime
import os
import random
import shutil
import tempfile
from typing import Any, Literal, Optional, cast

import gradio as gr
from openai import AsyncOpenAI

from dialogue_tts_core.audio_utils import (
    merge_mp3_files,  # noqa: F401 - merge_mp3_files might be used by orchestrator or implicitly
)

# from dialogue_tts_core.tts_client import synthesize_speech_line # Now by orchestrator
from dialogue_tts_core.config_models import OPENAI_VOICES_TUPLE, SpeakerTTSConfig
from dialogue_tts_core.dialogue_script_parser import (
    calculate_cost,  # calculate_cost is still used elsewhere in this file
    parse_dialogue_script,
    # get_speakers_from_script will be defined locally for now,
    # as per pseudocode it should take parsed_lines
)
from dialogue_tts_core.tts_orchestrator import orchestrate_tts_synthesis

from .ui_layout import (
    APP_AVAILABLE_VOICES,
    DEFAULT_VIBE,
    PREDEFINED_VIBES,
)

# Explicitly define the Literal type for OpenAI voices for robust casting
OpenAIVoiceLiteralType = Literal[
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "sage",
    "nova",
    "shimmer",
    "verse",
]


def get_speakers_from_script(parsed_lines: list[dict]) -> list[str]:
    """Extracts unique, ordered speaker names from parsed script lines."""
    if not parsed_lines:
        return []
    seen_speakers = set()
    ordered_unique_speakers = []
    for line_data in parsed_lines:
        speaker = line_data.get("speaker")
        if speaker and speaker not in seen_speakers:
            ordered_unique_speakers.append(speaker)
            seen_speakers.add(speaker)
    return ordered_unique_speakers


def handle_dynamic_accordion_input_change(
    new_value: Any, current_speaker_configs: dict, speaker_name: str, config_key: str
) -> dict:
    if not isinstance(current_speaker_configs, dict):
        print(
            "Warning: current_speaker_configs was not a dict in "
            "handle_dynamic_accordion_input_change. "
            f"Type: {type(current_speaker_configs)}. Re-initializing."
        )
        current_speaker_configs = {}

    updated_configs = current_speaker_configs.copy()

    if speaker_name not in updated_configs:
        updated_configs[speaker_name] = {}

    updated_configs[speaker_name][config_key] = new_value
    updated_configs["_last_dynamic_update_details"] = (
        f"Speaker: {speaker_name}, Key: {config_key}, "
        f"Val: {str(new_value)[:20]}, "
        f"TS: {datetime.datetime.now(datetime.timezone.utc).isoformat()}"
    )
    print(
        f"DEBUG (dynamic_input_change): Speaker '{speaker_name}' config "
        f"'{config_key}' to '{str(new_value)[:50]}'. New state hint: "
        f"{updated_configs.get('_last_dynamic_update_details')}"
    )
    return updated_configs


async def handle_script_processing(  # noqa: C901
    openai_api_key: str,
    async_openai_client: AsyncOpenAI,
    nsfw_api_url_template: str,  # This should be treated as Optional[str]
    dialogue_script: str,
    tts_model: str,  # This is the tts_global_model for the orchestrator
    pause_ms: int,  # This is the global_pause_ms for the orchestrator
    speaker_config_method: str,  # UI method string
    global_voice_selection: str,  # UI selected global voice
    speaker_configs_state_dict: dict,  # UI state for detailed per-speaker configs
    global_speed: float,  # UI global speed
    global_instructions: str,  # UI global instructions
    progress: Optional[gr.Progress] = None,
) -> tuple[str | None, str | None, str]:
    if progress is None:
        progress = gr.Progress(track_tqdm=True)

    progress(0, desc="Initializing...")

    # 1. Initial Validations
    if not openai_api_key or not async_openai_client:
        return None, None, "Error: OpenAI API Key or client is not configured."
    if not dialogue_script or not dialogue_script.strip():
        return None, None, "Error: Script is empty."

    # 2. Prepare a base output directory for the orchestrator
    base_temp_output_dir = tempfile.mkdtemp(prefix="gradio_tts_job_base_")

    # 3. Parse script
    try:
        parsed_lines, _total_chars = parse_dialogue_script(dialogue_script)
        if not parsed_lines:
            shutil.rmtree(base_temp_output_dir)
            return None, None, "Error: No valid lines found in script."
    except ValueError as e:
        shutil.rmtree(base_temp_output_dir)
        return None, None, f"Script parsing error: {e!s}"

    progress(0.1, desc="Script parsed. Resolving speaker configurations...")

    # 4. Resolve Speaker Configurations to map[str, SpeakerTTSConfig]
    resolved_configs: dict[str, SpeakerTTSConfig] = {}
    unique_speakers_in_script = get_speakers_from_script(parsed_lines)

    if not isinstance(speaker_configs_state_dict, dict):
        speaker_configs_state_dict = {}

    # Determine safe_default_global_voice, ensuring it's a valid Literal
    _default_voice_from_tuple = (
        OPENAI_VOICES_TUPLE[0] if OPENAI_VOICES_TUPLE else "alloy"
    )  # Fallback if tuple is empty

    if global_voice_selection in OPENAI_VOICES_TUPLE:
        safe_default_global_voice = global_voice_selection
    # Check if APP_AVAILABLE_VOICES[0] is a valid literal before assigning
    elif APP_AVAILABLE_VOICES and APP_AVAILABLE_VOICES[0] in OPENAI_VOICES_TUPLE:
        safe_default_global_voice = APP_AVAILABLE_VOICES[0]
    else:
        safe_default_global_voice = _default_voice_from_tuple  # This is a known Literal

    default_global_tts_config = SpeakerTTSConfig(
        voice=safe_default_global_voice,  # This is now guaranteed to be a valid Literal
        speed=global_speed if tts_model in ["tts-1", "tts-1-hd"] else 1.0,
        vibe="None",
        custom_instructions=global_instructions.strip()
        if global_instructions and tts_model == "gpt-4o-mini-tts"
        else None,
    )

    if speaker_config_method == "Single Voice (Global)":
        for speaker_name in unique_speakers_in_script:
            resolved_configs[speaker_name] = default_global_tts_config.copy(deep=True)

    elif speaker_config_method == "Random per Speaker":
        # Ensure the pool contains only valid Literal voice names
        effective_voices_pool = [
            v for v in APP_AVAILABLE_VOICES if v in OPENAI_VOICES_TUPLE
        ]
        if not effective_voices_pool:
            effective_voices_pool = [
                safe_default_global_voice
            ]  # safe_default_global_voice is a Literal

        for speaker_name in unique_speakers_in_script:
            _chosen_voice_str = random.choice(effective_voices_pool)
            # Use Pydantic's model_fields to get the annotation for casting
            chosen_voice_literal = cast(OpenAIVoiceLiteralType, _chosen_voice_str)
            resolved_configs[speaker_name] = SpeakerTTSConfig(
                voice=chosen_voice_literal,
                speed=default_global_tts_config.speed,
                vibe=default_global_tts_config.vibe,
                custom_instructions=default_global_tts_config.custom_instructions,
            )

    elif speaker_config_method == "A/B Round Robin":
        # Ensure the pool contains only valid Literal voice names
        effective_voices_pool_ab = [
            v for v in APP_AVAILABLE_VOICES if v in OPENAI_VOICES_TUPLE
        ]
        if not effective_voices_pool_ab:
            effective_voices_pool_ab = [
                safe_default_global_voice
            ]  # safe_default_global_voice is a Literal

        for i, speaker_name in enumerate(unique_speakers_in_script):
            _chosen_voice_str_ab = effective_voices_pool_ab[
                i % len(effective_voices_pool_ab)
            ]
            # Use Pydantic's model_fields to get the annotation for casting
            chosen_voice_literal_ab = cast(OpenAIVoiceLiteralType, _chosen_voice_str_ab)
            resolved_configs[speaker_name] = SpeakerTTSConfig(
                voice=chosen_voice_literal_ab,
                speed=default_global_tts_config.speed,
                vibe=default_global_tts_config.vibe,
                custom_instructions=default_global_tts_config.custom_instructions,
            )

    elif speaker_config_method == "Detailed Configuration (Per Speaker UI)":
        for speaker_name in unique_speakers_in_script:
            speaker_ui_settings = speaker_configs_state_dict.get(speaker_name, {})
            current_speaker_resolved_config = default_global_tts_config.copy(deep=True)

            # Validate voice from UI settings
            ui_voice_selection = speaker_ui_settings.get(
                "voice", default_global_tts_config.voice
            )
            if (
                ui_voice_selection in OPENAI_VOICES_TUPLE
            ):  # Check against the Literal tuple
                current_speaker_resolved_config.voice = ui_voice_selection
            else:
                # Fallback if UI somehow provided an invalid voice string
                current_speaker_resolved_config.voice = (
                    default_global_tts_config.voice
                )  # which is a known Literal

            final_custom_instructions_for_speaker = (
                default_global_tts_config.custom_instructions
            )

            if tts_model in ["tts-1", "tts-1-hd"]:
                current_speaker_resolved_config.speed = float(
                    speaker_ui_settings.get("speed", default_global_tts_config.speed)
                )

            elif tts_model == "gpt-4o-mini-tts":
                speaker_vibe_selection = speaker_ui_settings.get("vibe", DEFAULT_VIBE)
                speaker_custom_instr_text = speaker_ui_settings.get(
                    "custom_instructions", ""
                ).strip()
                current_speaker_resolved_config.vibe = speaker_vibe_selection
                temp_line_instr = None
                if speaker_vibe_selection == "Custom..." and speaker_custom_instr_text:
                    temp_line_instr = speaker_custom_instr_text
                elif (
                    speaker_vibe_selection != "None"
                    and speaker_vibe_selection != "Custom..."
                    and PREDEFINED_VIBES.get(speaker_vibe_selection)
                ):
                    temp_line_instr = PREDEFINED_VIBES[speaker_vibe_selection]

                if temp_line_instr is not None:
                    final_custom_instructions_for_speaker = temp_line_instr

                current_speaker_resolved_config.custom_instructions = (
                    final_custom_instructions_for_speaker
                )
                current_speaker_resolved_config.speed = 1.0  # Speed not applicable for
                # gpt-4o-mini-tts via API

            resolved_configs[speaker_name] = current_speaker_resolved_config
    else:
        shutil.rmtree(base_temp_output_dir)
        return (
            None,
            None,
            f"Error: Unknown speaker configuration method '{speaker_config_method}'.",
        )

    # Ensure all speakers in the script have a configuration, even if it's the default
    for speaker_name in unique_speakers_in_script:
        if speaker_name not in resolved_configs:
            resolved_configs[speaker_name] = default_global_tts_config.copy(deep=True)

    progress(0.3, desc="Configurations resolved. Starting TTS orchestration...")

    effective_nsfw_template = (
        nsfw_api_url_template
        if nsfw_api_url_template and nsfw_api_url_template.strip()
        else None
    )

    zip_file_path, merged_file_path, status_message = await orchestrate_tts_synthesis(
        parsed_script=parsed_lines,
        tts_global_model=tts_model,
        global_pause_ms=pause_ms,
        resolved_speaker_configs_map=resolved_configs,
        openai_client=async_openai_client,
        output_directory=base_temp_output_dir,  # Orchestrator creates sub-directory
        cache_base_dir=os.getenv("APP_CACHE_BASE_DIR", ".cache/tts_cache"),  # Added
        nsfw_api_url_template=effective_nsfw_template,
        # progress_callback=progress # If orchestrator supports it directly
    )

    # If orchestrator doesn't handle progress updates internally,
    # we might need to adjust this.
    # For now, assume orchestrate_tts_synthesis is a long-running task and update
    # progress after it.
    # The pseudocode implies progress updates within the orchestrator or that it's
    # quick enough.
    # Let's assume the orchestrator handles its own internal progress if it's complex,
    # or we set to 1.0 after it's done.
    progress(1.0, desc="Processing complete!")

    # Cleanup: if orchestrator created a job-specific subfolder and it's now empty
    # (e.g. all failed)
    # or if the base_temp_output_dir itself is empty (e.g. orchestrator failed early)
    # This part might need refinement based on orchestrator's exact behavior
    # with output_directory
    if (
        zip_file_path is None
        and merged_file_path is None
        and os.path.exists(base_temp_output_dir)
        and not os.listdir(base_temp_output_dir)
    ):
        # Check if base_temp_output_dir is empty or has an empty job subfolder
        # This logic is a bit simplified; a more robust check might be needed.
        shutil.rmtree(base_temp_output_dir)
        # If orchestrator creates a sub-dir, e.g. base_temp_output_dir/job_XYZ,
        # and that sub-dir is empty, we might want to clean that too.
        # For now, the orchestrator is expected to return None paths if it cleans up
        # its own failed job dir.

    return zip_file_path, merged_file_path, status_message


# ... (rest of the event_handlers.py file remains the same) ...


def handle_calculate_cost(dialogue_script: str, tts_model: str) -> str:
    if not dialogue_script or not dialogue_script.strip():
        return "Cost: $0.00 (Script is empty)"
    try:
        parsed_lines, total_chars = parse_dialogue_script(dialogue_script)
        if not parsed_lines:
            return "Cost: $0.00 (No valid lines in script)"
        cost = calculate_cost(total_chars, len(parsed_lines), tts_model)
        return (
            f"Estimated Cost for {len(parsed_lines)} lines "
            f"({total_chars} chars): ${cost:.6f}"
        )
    except ValueError as e:
        return f"Cost calculation error: {e!s}"
    except Exception as e:  # noqa: BLE001 # Catch unexpected cost calculation issues
        return f"An unexpected error: {e!s}"


def handle_load_refresh_per_speaker_ui_trigger(
    script_text: str, current_speaker_configs: dict, tts_model: str
) -> dict:
    print(
        f"DEBUG (Load/Refresh Trigger): Script: '{script_text[:30]}...', "
        f"Model: {tts_model},"
    )
    keys_str = (
        str(list(current_speaker_configs.keys()))
        if isinstance(current_speaker_configs, dict)
        else "'Not a dict'"
    )
    print(f"Current State Keys: {keys_str}")
    if not isinstance(current_speaker_configs, dict):
        current_speaker_configs = {}
    updated_configs = current_speaker_configs.copy()
    updated_configs["_last_action_source"] = "load_refresh_button"
    updated_configs["_last_action_timestamp"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    return updated_configs


def handle_tts_model_change(
    selected_model: str, current_speaker_configs: dict
) -> tuple[dict, dict, dict]:
    print(f"DEBUG (TTS Model Change): Model: {selected_model}, ")
    keys_str_tts_change = (
        str(list(current_speaker_configs.keys()))
        if isinstance(current_speaker_configs, dict)
        else "'Not a dict'"
    )
    print(f"Current State Keys: {keys_str_tts_change}")
    if not isinstance(current_speaker_configs, dict):
        current_speaker_configs = {}
    updated_configs = current_speaker_configs.copy()
    for speaker_name_key in list(updated_configs.keys()):
        if isinstance(updated_configs[speaker_name_key], dict):
            if selected_model == "gpt-4o-mini-tts":
                updated_configs[speaker_name_key].pop("speed", None)
                if "vibe" not in updated_configs[speaker_name_key]:
                    updated_configs[speaker_name_key]["vibe"] = DEFAULT_VIBE
            elif selected_model in ["tts-1", "tts-1-hd"]:
                updated_configs[speaker_name_key].pop("vibe", None)
                updated_configs[speaker_name_key].pop("custom_instructions", None)
                if "speed" not in updated_configs[speaker_name_key]:
                    updated_configs[speaker_name_key]["speed"] = 1.0
    updated_configs["_last_action_source"] = "tts_model_change"
    updated_configs["_last_action_timestamp"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    is_tts1_family = selected_model in ["tts-1", "tts-1-hd"]
    is_gpt_mini_tts = selected_model == "gpt-4o-mini-tts"
    return (
        gr.update(visible=is_tts1_family, interactive=is_tts1_family),
        gr.update(visible=is_gpt_mini_tts, interactive=is_gpt_mini_tts),
        updated_configs,
    )


def handle_speaker_config_method_visibility_change(
    method: str,
) -> tuple[dict, dict]:
    print(f"DEBUG (Config Method Change): Method: {method}")
    is_single_voice_visible = method == "Single Voice (Global)"
    is_detailed_per_speaker_container_visible = (
        method == "Detailed Configuration (Per Speaker UI)"
    )
    return (
        gr.update(visible=is_single_voice_visible),
        gr.update(visible=is_detailed_per_speaker_container_visible),
    )
