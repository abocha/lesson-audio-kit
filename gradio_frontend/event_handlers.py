# FILE: esl-dialogue-tts/event_handlers.py
import datetime
import os
import random
import shutil
import tempfile
from typing import Any, Optional
import zipfile

import gradio as gr
from openai import AsyncOpenAI  # Moved import here
from ui_layout import (
    APP_AVAILABLE_VOICES,
    DEFAULT_VIBE,
    PREDEFINED_VIBES,
)

from dialogue_tts_core.audio_utils import merge_mp3_files
from dialogue_tts_core.parser import calculate_cost, parse_dialogue_script
from dialogue_tts_core.tts_client import synthesize_speech_line


def get_speakers_from_script(script_text: str) -> list:
    """Extracts unique, ordered speaker names from the script."""
    if not script_text or not script_text.strip():
        return []
    try:
        parsed_lines, _ = parse_dialogue_script(script_text)
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
    except ValueError:
        print("ValueError during script parsing in get_speakers_from_script.")
        return []
    except Exception as e:  # noqa: BLE001 - Last-resort catch for unexpected parsing errors
        print(f"Unexpected error in get_speakers_from_script: {e}")
        return []


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
    nsfw_api_url_template: str,
    dialogue_script: str,
    tts_model: str,
    pause_ms: int,
    speaker_config_method: str,
    global_voice_selection: str,
    speaker_configs_state_dict: dict,
    global_speed: float,
    global_instructions: str,
    progress: Optional[gr.Progress] = None,
) -> tuple[str | None, str | None, str]:
    if progress is None:
        progress = gr.Progress(track_tqdm=True)
    if not openai_api_key or not async_openai_client:
        return None, None, "Error: OpenAI API Key or client is not configured."
    if not dialogue_script or not dialogue_script.strip():
        return None, None, "Error: Script is empty."

    job_audio_path_prefix = os.path.join(
        tempfile.gettempdir(), f"dialogue_tts_job_{random.randint(10000, 99999)}"
    )
    if os.path.exists(job_audio_path_prefix):
        shutil.rmtree(job_audio_path_prefix)
    os.makedirs(job_audio_path_prefix, exist_ok=True)

    try:
        parsed_lines, _ = parse_dialogue_script(dialogue_script)
        if not parsed_lines:
            shutil.rmtree(job_audio_path_prefix)
            return None, None, "Error: No valid lines found in script."
    except ValueError as e:
        shutil.rmtree(job_audio_path_prefix)
        return None, None, f"Script parsing error: {e!s}"

    if not isinstance(speaker_configs_state_dict, dict):
        print(
            "Warning: speaker_configs_state_dict was not a dict in "
            "handle_script_processing. "
            f"Re-initializing. Type: {type(speaker_configs_state_dict)}"
        )
        speaker_configs_state_dict = {}

    safe_default_global_voice = (
        global_voice_selection
        if global_voice_selection in APP_AVAILABLE_VOICES
        else (APP_AVAILABLE_VOICES[0] if APP_AVAILABLE_VOICES else "alloy")
    )

    speaker_voice_map = {}  # Calculated once if needed
    if speaker_config_method in ["Random per Speaker", "A/B Round Robin"]:
        unique_script_speakers_for_map = get_speakers_from_script(dialogue_script)
        temp_voices_pool = APP_AVAILABLE_VOICES.copy()
        if not temp_voices_pool:
            temp_voices_pool = [safe_default_global_voice]

        if speaker_config_method == "Random per Speaker":
            for spk_name in unique_script_speakers_for_map:
                speaker_voice_map[spk_name] = random.choice(temp_voices_pool)
        elif speaker_config_method == "A/B Round Robin" and temp_voices_pool:
            for i, spk_name in enumerate(unique_script_speakers_for_map):
                speaker_voice_map[spk_name] = temp_voices_pool[
                    i % len(temp_voices_pool)
                ]

    processed_results_map = {}
    total_lines = len(parsed_lines)
    progress(0, desc="Starting: Preparing for audio synthesis...")

    for i, line_data in enumerate(parsed_lines):
        speaker_name = line_data["speaker"]
        line_text = line_data["text"]
        line_id = line_data["id"]

        # Determine voice, speed, and instructions for the current line
        line_voice = safe_default_global_voice
        line_speed = global_speed
        line_instructions = (
            global_instructions.strip()
            if global_instructions and global_instructions.strip()
            else None
        )

        if speaker_config_method == "Detailed Configuration (Per Speaker UI)":
            spk_cfg = speaker_configs_state_dict.get(speaker_name, {})
            line_voice = spk_cfg.get("voice", safe_default_global_voice)
            if tts_model in ["tts-1", "tts-1-hd"]:
                line_speed = float(spk_cfg.get("speed", global_speed))
            # For gpt-4o-mini-tts, detailed instructions/vibe
            if tts_model == "gpt-4o-mini-tts":
                vibe = spk_cfg.get("vibe", DEFAULT_VIBE)
                custom_instr_raw = spk_cfg.get("custom_instructions", "")
                custom_instr = custom_instr_raw.strip() if custom_instr_raw else ""
                current_line_specific_instructions = None
                if vibe == "Custom..." and custom_instr:
                    current_line_specific_instructions = custom_instr
                elif (
                    vibe != "None"
                    and vibe != "Custom..."
                    and PREDEFINED_VIBES.get(vibe)
                ):
                    current_line_specific_instructions = PREDEFINED_VIBES[vibe]
                line_instructions = (
                    current_line_specific_instructions
                    if current_line_specific_instructions is not None
                    else line_instructions
                )
        elif speaker_config_method in ["Random per Speaker", "A/B Round Robin"]:
            line_voice = speaker_voice_map.get(speaker_name, safe_default_global_voice)
            # Speed and instructions remain global for these methods

        # Ensure speed is 1.0 if model does not support it explicitly,
        # or handled globally
        if tts_model not in ["tts-1", "tts-1-hd"]:
            line_speed = 1.0

        out_fn = os.path.join(
            job_audio_path_prefix,
            f"line_{line_id}_{speaker_name.replace(' ', '_')}.mp3",
        )

        # Update progress BEFORE awaiting the synthesis for this line
        progress_fraction = (i + 1) / total_lines
        progress(
            progress_fraction,
            desc=f"Synthesizing: Line {i + 1}/{total_lines} ('{speaker_name}')",
        )

        try:
            result_path = await synthesize_speech_line(
                client=async_openai_client,
                text=line_text,
                voice=line_voice,
                output_path=out_fn,
                model=tts_model,
                speed=line_speed,
                instructions=line_instructions,
                nsfw_api_url_template=nsfw_api_url_template,
                line_index=line_id,
            )
            processed_results_map[line_id] = {
                "path": result_path,
                "speaker": speaker_name,
            }
        except Exception as e:  # noqa: BLE001 # Catch any unexpected error during individual line synthesis
            print(f"Error synthesizing line ID {line_id} ({speaker_name}): {e}")
            processed_results_map[line_id] = {
                "path": None,
                "error": str(e),
                "speaker": speaker_name,
            }

    progress(1.0, desc="Finalizing: Assembling audio files...")

    ordered_files_for_merge_and_zip = []
    for p_line in parsed_lines:
        line_id = p_line["id"]
        res = processed_results_map.get(line_id)
        if (
            res
            and res.get("path")
            and os.path.exists(res["path"])
            and os.path.getsize(res["path"]) > 0
        ):
            ordered_files_for_merge_and_zip.append(res["path"])
        else:
            if res:
                print(
                    f"Skipped or failed synthesizing line ID {line_id} "
                    f"({res.get('speaker', 'Unknown')}) for merge/zip. "
                    f"Error: {res.get('error')}"
                )
            else:
                print(
                    f"Result for line ID {line_id} not found in processed_results_map."
                )

    valid_files_for_zip = [f for f in ordered_files_for_merge_and_zip if f]

    if not valid_files_for_zip:
        shutil.rmtree(job_audio_path_prefix)
        return None, None, "Error: No audio was successfully synthesized for any line."

    zip_fn = os.path.join(job_audio_path_prefix, "dialogue_lines.zip")
    with zipfile.ZipFile(zip_fn, "w") as zf:
        for f_path in valid_files_for_zip:
            zf.write(f_path, os.path.basename(f_path))

    files_to_actually_merge = valid_files_for_zip
    merged_fn = os.path.join(job_audio_path_prefix, "merged_dialogue.mp3")
    merged_path = merge_mp3_files(files_to_actually_merge, merged_fn, pause_ms)

    status_msg = (
        f"Successfully processed {len(valid_files_for_zip)} "
        f"out of {len(parsed_lines)} lines. "
    )
    if len(valid_files_for_zip) < len(parsed_lines):
        status_msg += "Some lines may have failed. Check console for details. "
    if not merged_path and len(valid_files_for_zip) > 0:
        status_msg += "Merging audio failed. "
    elif not merged_path:
        status_msg = "No audio to merge (all lines failed or were skipped)."
    else:
        status_msg += "Merged audio generated."

    progress(1.0, desc="Processing complete!")  # Final update

    return (
        zip_fn if os.path.exists(zip_fn) else None,
        merged_path if merged_path and os.path.exists(merged_path) else None,
        status_msg,
    )


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
) -> tuple[gr.update, gr.update, dict]:
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
) -> tuple[gr.update, gr.update]:
    print(f"DEBUG (Config Method Change): Method: {method}")
    is_single_voice_visible = method == "Single Voice (Global)"
    is_detailed_per_speaker_container_visible = (
        method == "Detailed Configuration (Per Speaker UI)"
    )
    return (
        gr.update(visible=is_single_voice_visible),
        gr.update(visible=is_detailed_per_speaker_container_visible),
    )
