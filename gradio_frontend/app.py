# FILE: app.py
import asyncio
from functools import partial
import os
import sys
from typing import Any, Optional
import uuid

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import gradio as gr
from openai import AsyncOpenAI

from dialogue_tts_core.cache_manager import (
    get_cache_stats,
    get_current_cache_size_bytes,
)
from dialogue_tts_core.config_models import (
    CacheStatsResponse,
    SpeakerTTSConfig,
    TTSJobCreationResponse,
    TTSJobOutputs,
    TTSJobStatusCompleted,
    TTSJobStatusFailed,
    TTSJobStatusPending,
    TTSJobStatusResponse,
    TTSRequestPayload,
)
from dialogue_tts_core.cost_router import QualityTier
from dialogue_tts_core.dialogue_script_parser import parse_dialogue_script
from dialogue_tts_core.speaker_config_resolver import (
    get_unique_speakers_from_parsed_script,
    resolve_speaker_configurations,
)
from dialogue_tts_core.tts_orchestrator import orchestrate_tts_synthesis

from .event_handlers import (
    get_speakers_from_script,
    handle_calculate_cost,
    handle_dynamic_accordion_input_change,
    handle_load_refresh_per_speaker_ui_trigger,
    handle_script_processing,
    handle_speaker_config_method_visibility_change,
    handle_tts_model_change,
)
from .ui_layout import (
    APP_AVAILABLE_VOICES,
    DEFAULT_GLOBAL_VOICE,
    DEFAULT_VIBE,
    MODEL_DEFAULT_ENV,
    TTS_MODELS_AVAILABLE,
    VIBE_CHOICES,
    create_action_and_output_components,
    create_main_input_components,
    create_speaker_config_components,
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()

# --- In-memory store for job statuses and results (for simplicity in v0) ---
job_store: dict[str, dict[str, Any]] = {}

# --- Secrets and Client Setup (Same as before) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NSFW_API_URL_TEMPLATE = os.getenv("NSFW_API_URL_TEMPLATE")
MODEL_DEFAULT_FROM_ENV = os.getenv("MODEL_DEFAULT", MODEL_DEFAULT_ENV)
EFFECTIVE_MODEL_DEFAULT = (
    MODEL_DEFAULT_FROM_ENV
    if MODEL_DEFAULT_FROM_ENV in TTS_MODELS_AVAILABLE
    else MODEL_DEFAULT_ENV
)
async_openai_client = None
if not OPENAI_API_KEY:
    # ... (secret loading logic) ...
    pass
if OPENAI_API_KEY:
    async_openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
else:
    print("CRITICAL ERROR: OPENAI_API_KEY secret is not set.")

app = FastAPI(
    title="Lesson Audio Kit API",
    version="v0.1.0",
    description="API for TTS synthesis and application utilities.",
)


@app.get(
    "/api/cache/stats", response_model=CacheStatsResponse, tags=["Cache Utilities"]
)
async def get_cache_statistics_endpoint() -> "CacheStatsResponse":
    cache_base_dir_env = os.getenv("APP_CACHE_BASE_DIR")
    if not cache_base_dir_env:
        print(
            "Warning: APP_CACHE_BASE_DIR environment variable not set. "
            "Using default '.cache/tts_cache'"
        )
        cache_base_dir_env = ".cache/tts_cache"

    max_cache_size_gb_env = os.getenv("APP_MAX_CACHE_SIZE_GB", "2.0")
    max_cache_size_gb_config = 2.0
    try:
        max_cache_size_gb_config = float(max_cache_size_gb_env)
    except ValueError:
        max_cache_size_gb_config = 2.0

    operational_stats = get_cache_stats()

    try:
        current_size_bytes = get_current_cache_size_bytes(cache_base_dir_env)
    except (FileNotFoundError, PermissionError, OSError) as e:
        raise HTTPException(
            status_code=500, detail=f"Error calculating cache size: {e!s}"
        ) from e

    current_size_gb = round(current_size_bytes / (1024**3), 3)

    return CacheStatsResponse(
        cache_size_bytes=current_size_bytes,
        cache_size_gb=current_size_gb,
        max_cache_size_gb=max_cache_size_gb_config,
        hits=operational_stats["hits"],
        misses=operational_stats["misses"],
        errors=operational_stats["errors"],
        total_lookups=operational_stats["total_lookups"],
    )


# --- Helper function to run the orchestration in background ---
async def run_tts_orchestration_task(
    job_id: str,
    parsed_script: list[dict],
    global_pause_ms: int,
    resolved_configs: dict[str, SpeakerTTSConfig],
    output_base_dir: str,
    cache_base_dir: str,
    nsfw_template: Optional[str],
) -> None:
    global job_store, async_openai_client
    if async_openai_client is None:
        job_store[job_id].update(
            {"status": "failed", "error_message": "OpenAI client not initialized."}
        )
        return

    job_store[job_id]["status"] = "processing"
    try:
        (
            zip_path,
            merged_path,
            status_msg,
            all_lines_synthesis_details,
        ) = await orchestrate_tts_synthesis(
            parsed_script=parsed_script,
            global_pause_ms=global_pause_ms,
            resolved_speaker_configs_map=resolved_configs,
            user_id=None,
            desired_quality_tier=QualityTier.MID,
            max_total_job_cost_usd=None,
            prefer_low_latency_routing=False,
            prefer_emotion_support_routing=False,
            openai_client=async_openai_client,
            output_directory=output_base_dir,
            cache_base_dir=cache_base_dir,
            nsfw_api_url_template=nsfw_template,
        )

        job_store[job_id].update(
            {
                "status": "completed",
                "message": status_msg,
                "outputs": {
                    "zip_file_local_path": str(zip_path) if zip_path else None,
                    "merged_mp3_local_path": str(merged_path) if merged_path else None,
                },
            }
        )
    except Exception as e:  # noqa: BLE001
        print(f"Error in TTS orchestration task for job {job_id}: {e!s}")
        job_store[job_id].update(
            {"status": "failed", "error_message": f"Orchestration error: {e!s}"}
        )


@app.post(
    "/api/tts", response_model=TTSJobCreationResponse, status_code=202, tags=["TTS"]
)
async def submit_tts_job_endpoint(
    payload: TTSRequestPayload, background_tasks: BackgroundTasks
) -> TTSJobCreationResponse:
    global job_store, async_openai_client

    if async_openai_client is None:
        raise HTTPException(
            status_code=503,
            detail="TTS service not available: OpenAI client not initialized.",
        )

    try:
        parsed_script_lines, _ = parse_dialogue_script(payload.script_text)
        if not parsed_script_lines:
            raise HTTPException(
                status_code=400,
                detail="Script is empty or contains no processable lines.",
            )
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail=f"Script parsing error: {e!s}"
        ) from e

    unique_speakers = get_unique_speakers_from_parsed_script(parsed_script_lines)
    if not unique_speakers:
        raise HTTPException(status_code=400, detail="No speakers found in the script.")

    try:
        resolved_configs = resolve_speaker_configurations(payload, unique_speakers)
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail=f"Speaker configuration resolution error: {e!s}"
        ) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during config resolution: {e!s}",
        ) from e

    job_id = uuid.uuid4().hex

    job_output_base_dir = os.getenv("APP_JOB_OUTPUT_DIR", ".job_outputs")
    os.makedirs(job_output_base_dir, exist_ok=True)

    cache_base_dir = os.getenv("APP_CACHE_BASE_DIR", ".cache/tts_cache")
    os.makedirs(cache_base_dir, exist_ok=True)

    nsfw_template = (
        payload.nsfw_check_options.api_url_template
        if payload.nsfw_check_options and payload.nsfw_check_options.enabled
        else None
    )
    effective_pause_ms = (
        payload.global_pause_ms if payload.global_pause_ms is not None else 500
    )

    job_store[job_id] = {"status": "pending", "request_payload": payload.model_dump()}

    background_tasks.add_task(
        run_tts_orchestration_task,
        job_id,
        parsed_script_lines,
        effective_pause_ms,
        resolved_configs,
        job_output_base_dir,
        cache_base_dir,
        nsfw_template,
    )

    status_url = f"/api/tts/status/{job_id}"

    return TTSJobCreationResponse(job_id=job_id, status_url=status_url)


# --- Endpoint for GET /api/tts/status/{job_id} ---
@app.get("/api/tts/status/{job_id}", response_model=TTSJobStatusResponse, tags=["TTS"])
async def get_tts_job_status_endpoint(job_id: str) -> TTSJobStatusResponse:
    global job_store
    job_info = job_store.get(job_id)

    if not job_info:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    status = job_info.get("status")

    def make_file_url(local_path: Optional[str], job_id_for_url: str) -> Optional[str]:
        if local_path and os.path.exists(local_path):
            file_name = os.path.basename(local_path)
            return f"/job_files/{job_id_for_url}/{file_name}"
        return None

    if status == "completed":
        outputs_data = job_info.get("outputs", {})

        return TTSJobStatusCompleted(
            job_id=job_id,
            status="completed",
            message=job_info.get("message", "Job completed successfully."),
            outputs=TTSJobOutputs(
                zip_file_url=make_file_url(
                    outputs_data.get("zip_file_local_path"),
                    job_id,
                ),
                merged_mp3_url=make_file_url(
                    outputs_data.get("merged_mp3_local_path"),
                    job_id,
                ),
            ),
        )
    if status == "failed":
        return TTSJobStatusFailed(
            job_id=job_id,
            status="failed",
            error_message=job_info.get("error_message", "Unknown error."),
        )
    # pending or processing
    return TTSJobStatusPending(
        job_id=job_id,
        status=job_info.get("status", "pending"),
    )


# Static file serving

job_output_dir_static = os.getenv("APP_JOB_OUTPUT_DIR", ".job_outputs")
os.makedirs(job_output_dir_static, exist_ok=True)
app.mount("/job_files", StaticFiles(directory=job_output_dir_static), name="job_files")


# --- Main Blocks UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(), elem_id="main_blocks_ui") as demo:  # type: ignore
    gr.Markdown("# Dialogue Script to Speech (OpenAI TTS)")
    if not OPENAI_API_KEY or not async_openai_client:
        gr.Markdown(
            "<h3 style='color:red;'>⚠️ Warning: OPENAI_API_KEY not set or invalid. "
            "Audio generation will fail.</h3>"
        )

    speaker_configs_state = gr.State({})

    # --- Create Main UI Components ---
    (
        script_input,
        tts_model_dropdown,
        pause_input,
        global_speed_input,
        global_instructions_input,
    ) = create_main_input_components(EFFECTIVE_MODEL_DEFAULT)

    (
        speaker_config_method_dropdown,
        single_voice_group,
        global_voice_dropdown,
        detailed_per_speaker_ui_group_container,
        load_per_speaker_ui_button,
    ) = create_speaker_config_components()

    (
        calculate_cost_button,
        generate_button,
        cost_output,
        individual_lines_zip_output,
        merged_dialogue_mp3_output,
        status_output,
    ) = create_action_and_output_components()

    # --- Dynamic UI (@gr.render) Definition (Same as before) ---
    with detailed_per_speaker_ui_group_container:

        @gr.render(
            inputs=[script_input, speaker_configs_state, tts_model_dropdown],
            triggers=[load_per_speaker_ui_button.click, tts_model_dropdown.change],
        )
        def render_dynamic_speaker_ui(
            current_script_text: str,
            current_speaker_configs: dict,
            current_tts_model: str,
        ) -> None:
            # ... (Full @gr.render implementation from previous correct step) ...
            print(f"DEBUG: @gr.render CALLED. Model: {current_tts_model}.")
            print(f"Script: '{current_script_text[:30]}...'.")
            keys_str = (
                str(list(current_speaker_configs.keys()))
                if isinstance(current_speaker_configs, dict)
                else "'Not a dict'"
            )
            print(f"State Keys: {keys_str}")
            parsed_script_lines, _ = parse_dialogue_script(current_script_text)
            unique_speakers = get_speakers_from_script(parsed_script_lines)
            if not unique_speakers:
                gr.Markdown(
                    "<p style='color: #888; margin-top:10px;'>Enter script & click "
                    "'Load/Refresh' for per-speaker settings.</p>"
                )
                return
            for speaker_idx, speaker_name in enumerate(unique_speakers):
                if not isinstance(current_speaker_configs, dict):
                    current_speaker_configs = {}
                speaker_specific_config = current_speaker_configs.get(speaker_name, {})
                speaker_name_safe = speaker_name.replace(" ", "_").lower()
                accordion_elem_id = f"accordion_spk_{speaker_idx}_{speaker_name_safe}"
                with gr.Accordion(
                    f"Settings for Speaker: {speaker_name}",
                    open=False,
                    elem_id=accordion_elem_id,
                ):
                    gr.Markdown(
                        f"Configure voice for **{speaker_name}** using "
                        f"**{current_tts_model}** model."
                    )
                    default_voice = speaker_specific_config.get(
                        "voice", DEFAULT_GLOBAL_VOICE
                    )
                    voice_dd_elem_id = f"voice_dd_spk_{speaker_idx}"
                    voice_dropdown = gr.Dropdown(
                        APP_AVAILABLE_VOICES,
                        value=default_voice,
                        label="Voice",
                        elem_id=voice_dd_elem_id,
                    )
                    voice_dropdown.change(
                        fn=partial(
                            handle_dynamic_accordion_input_change,
                            speaker_name=speaker_name,
                            config_key="voice",
                        ),
                        inputs=[voice_dropdown, speaker_configs_state],
                        outputs=[speaker_configs_state],
                    )
                    if current_tts_model in ["tts-1", "tts-1-hd"]:
                        default_speed = float(speaker_specific_config.get("speed", 1.0))
                        speed_slider_elem_id = f"speed_slider_spk_{speaker_idx}"
                        speed_slider = gr.Slider(
                            minimum=0.25,
                            maximum=4.0,
                            value=default_speed,
                            step=0.05,
                            label="Speed",
                            elem_id=speed_slider_elem_id,
                        )
                        speed_slider.change(
                            fn=partial(
                                handle_dynamic_accordion_input_change,
                                speaker_name=speaker_name,
                                config_key="speed",
                            ),
                            inputs=[speed_slider, speaker_configs_state],
                            outputs=[speaker_configs_state],
                        )
                    elif current_tts_model == "gpt-4o-mini-tts":
                        default_vibe = speaker_specific_config.get("vibe", DEFAULT_VIBE)
                        vibe_dd_elem_id = f"vibe_dd_spk_{speaker_idx}"
                        vibe_dropdown = gr.Dropdown(
                            VIBE_CHOICES,
                            value=default_vibe,
                            label="Vibe/Emotion",
                            elem_id=vibe_dd_elem_id,
                        )
                        default_custom_instructions = speaker_specific_config.get(
                            "custom_instructions", ""
                        )
                        custom_instr_tb_elem_id = f"custom_instr_tb_spk_{speaker_idx}"
                        custom_instructions_textbox = gr.Textbox(
                            label="Custom Instructions",
                            value=default_custom_instructions,
                            placeholder="e.g., Speak slightly hesitant.",
                            lines=2,
                            visible=(default_vibe == "Custom..."),
                            elem_id=custom_instr_tb_elem_id,
                        )
                        vibe_dropdown.change(
                            fn=partial(
                                handle_dynamic_accordion_input_change,
                                speaker_name=speaker_name,
                                config_key="vibe",
                            ),
                            inputs=[vibe_dropdown, speaker_configs_state],
                            outputs=[speaker_configs_state],
                        ).then(
                            fn=lambda vibe_val: gr.update(
                                visible=(vibe_val == "Custom...")
                            ),
                            inputs=[vibe_dropdown],
                            outputs=[custom_instructions_textbox],
                        )
                        custom_instructions_textbox.change(
                            fn=partial(
                                handle_dynamic_accordion_input_change,
                                speaker_name=speaker_name,
                                config_key="custom_instructions",
                            ),
                            inputs=[custom_instructions_textbox, speaker_configs_state],
                            outputs=[speaker_configs_state],
                        )

    # --- Event Listeners (Same as before) ---
    tts_model_dropdown.change(
        fn=handle_tts_model_change,
        inputs=[tts_model_dropdown, speaker_configs_state],
        outputs=[global_speed_input, global_instructions_input, speaker_configs_state],
    )
    speaker_config_method_dropdown.change(
        fn=handle_speaker_config_method_visibility_change,
        inputs=[speaker_config_method_dropdown],
        outputs=[single_voice_group, detailed_per_speaker_ui_group_container],
    )
    load_per_speaker_ui_button.click(
        fn=handle_load_refresh_per_speaker_ui_trigger,
        inputs=[script_input, speaker_configs_state, tts_model_dropdown],
        outputs=[speaker_configs_state],
    )
    calculate_cost_button.click(
        fn=handle_calculate_cost,
        inputs=[script_input, tts_model_dropdown],
        outputs=[cost_output],
    )

    # Define a wrapper for the click handler to check for None client
    async def checked_handle_script_processing_click(
        # These are the inputs from the Gradio UI for the generate_button
        dialogue_script_ui: str,
        tts_model_ui: str,
        pause_ms_ui: int,
        speaker_config_method_ui: str,
        global_voice_selection_ui: str,
        speaker_configs_state_dict_ui: dict,
        global_speed_ui: float,
        global_instructions_ui: str,
        progress_ui: gr.Progress | None = None,  # Gradio provides this
    ) -> tuple[str | None, str | None, str]:
        if not OPENAI_API_KEY or not async_openai_client:
            print(
                "Error: OpenAI API Key or client is not configured for generate button."
            )
            return (
                None,
                None,
                "Error: OpenAI API Key or client not set. Cannot generate.",
            )

        # NSFW_API_URL_TEMPLATE can be None; handle_script_processing handles it.
        # If None, it's passed as None. If str, passed as str.
        # Type hint in handle_script_processing is str, should be str | None.
        # Assuming internal checks in handle_script_processing.
        # Or, ensure it's a string if the function strictly expects one.
        # For now, ensure it's a string or an empty string.
        nsfw_template_str_for_call = (
            NSFW_API_URL_TEMPLATE if NSFW_API_URL_TEMPLATE is not None else ""
        )

        return await handle_script_processing(
            openai_api_key=OPENAI_API_KEY,  # Now definitely a str
            async_openai_client=async_openai_client,  # Now definitely AsyncOpenAI
            nsfw_api_url_template=nsfw_template_str_for_call,  # Now definitely a str
            dialogue_script=dialogue_script_ui,
            tts_model=tts_model_ui,
            pause_ms=pause_ms_ui,
            speaker_config_method=speaker_config_method_ui,
            global_voice_selection=global_voice_selection_ui,
            speaker_configs_state_dict=speaker_configs_state_dict_ui,
            global_speed=global_speed_ui,
            global_instructions=global_instructions_ui,
            progress=progress_ui,
        )

    generate_button.click(
        fn=checked_handle_script_processing_click,  # Use the new wrapper
        inputs=[
            script_input,
            tts_model_dropdown,
            pause_input,
            speaker_config_method_dropdown,
            global_voice_dropdown,
            speaker_configs_state,
            global_speed_input,
            global_instructions_input,
        ],
        outputs=[
            individual_lines_zip_output,
            merged_dialogue_mp3_output,
            status_output,
        ],
    )

    # --- Examples Section Definition (Moved here) ---
    gr.Markdown("## Example Scripts")  # Keep the header if desired

    # Define the lists needed for Examples right here
    example_inputs_list_comps = [
        script_input,
        tts_model_dropdown,
        pause_input,
        speaker_config_method_dropdown,
        global_voice_dropdown,
        global_speed_input,
        global_instructions_input,
    ]
    example_outputs_list_comps = [
        individual_lines_zip_output,
        merged_dialogue_mp3_output,
        status_output,
    ]

    # Wrapper for example processing
    async def checked_example_handle_script_processing_click(
        # These are the inputs from the Gradio examples data
        dialogue_script_ex: str,
        tts_model_ex: str,
        pause_ms_ex: int,
        speaker_config_method_ex: str,
        global_voice_selection_ex: str,
        global_speed_ex: float,
        global_instructions_ex: str,
        # speaker_configs_state_dict is not provided by examples, default to empty
        # progress is not provided by examples
    ) -> tuple[str | None, str | None, str]:
        if not OPENAI_API_KEY or not async_openai_client:
            print("Error: OpenAI API Key or client is not configured for examples.")
            return (
                None,
                None,
                "Error: OpenAI API Key or client not set. Cannot process example.",
            )

        nsfw_template_str_for_call_ex = (
            NSFW_API_URL_TEMPLATE if NSFW_API_URL_TEMPLATE is not None else ""
        )
        speaker_configs_state_dict_for_call_ex = {}  # Examples don't have complex state

        return await handle_script_processing(
            openai_api_key=OPENAI_API_KEY,
            async_openai_client=async_openai_client,
            nsfw_api_url_template=nsfw_template_str_for_call_ex,
            dialogue_script=dialogue_script_ex,
            tts_model=tts_model_ex,
            pause_ms=pause_ms_ex,
            speaker_config_method=speaker_config_method_ex,
            global_voice_selection=global_voice_selection_ex,
            speaker_configs_state_dict=speaker_configs_state_dict_for_call_ex,
            global_speed=global_speed_ex,
            global_instructions=global_instructions_ex,
            progress=None,  # Examples don't use progress UI element directly
        )

    example_process_fn_actual = (
        checked_example_handle_script_processing_click
        if OPENAI_API_KEY and async_openai_client
        else None
    )
    examples_data = [
        [
            (
                "[Alice] Hello Bob, this is a test using the detailed "
                "configuration method.\n"
                "[Bob] Hi Alice! I'm Bob, and I'll have my own voice "
                "settings.\n"
                "[Alice] Let's see how this sounds."
            ),
            "tts-1-hd",
            300,
            "Random per Speaker",
            DEFAULT_GLOBAL_VOICE,
            1.0,
            "",
        ],
        [
            """[Narrator] Once upon a time, there was a gentle breeze over the hills.
[Narrator] The village below prepared for the annual festival as the sun set.""",
            "gpt-4o-mini-tts",
            200,
            "Detailed Configuration (Per Speaker UI)",
            DEFAULT_GLOBAL_VOICE,
            1.0,
            "Speak with a gentle, storytelling tone.",
        ],
        [
            """[Solo] This is a quick single-voice demo for testing purposes.""",
            "tts-1",
            0,
            "Single Voice (Global)",
            "fable",
            1.2,
            "",
        ],
    ]

    # Validate example data length against input components length
    num_inputs_expected = len(example_inputs_list_comps)
    valid_examples_data_inline = []
    for ex_data in examples_data:
        if len(ex_data) == num_inputs_expected:
            valid_examples_data_inline.append(ex_data)
        else:
            print(
                f"Warning (Inline Examples): Example data mismatch. "
                f"Expected {num_inputs_expected}, got {len(ex_data)}. Skipping."
            )

    # Directly instantiate gr.Examples if valid data exists
    if valid_examples_data_inline:
        if example_process_fn_actual:
            gr.Examples(
                examples=valid_examples_data_inline,
                inputs=example_inputs_list_comps,
                outputs=example_outputs_list_comps,
                fn=example_process_fn_actual,
                cache_examples=False,
                examples_per_page=5,
                label=(
                    "Example Scripts (Click to Load)"
                ),  # Label is optional if header exists
                run_on_click=False,
            )
        else:
            gr.Examples(
                examples=valid_examples_data_inline,
                inputs=example_inputs_list_comps,
                examples_per_page=5,
                label=(
                    "Example Scripts (Click to Load Inputs)"
                ),  # Label is optional if header exists
            )
    else:
        gr.Markdown(
            "<p style='color: orange;'>No valid examples could be loaded due to "
            "configuration mismatch.</p>"
        )

app = gr.mount_gradio_app(app, demo, path="/gradio_ui")

# --- Launch ---
if __name__ == "__main__":
    import uvicorn

    # asyncio is already imported at the top
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Make sure 'app' here refers to the FastAPI instance
    uvicorn.run(app, host="0.0.0.0", port=7860)
    # For development with reload, you might use:
    # uvicorn.run("gradio_frontend.app:app", host="0.0.0.0", port=7860, reload=True)
    # Ensure the string "gradio_frontend.app:app" correctly points to your
    # FastAPI app instance.
