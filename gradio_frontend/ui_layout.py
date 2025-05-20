# FILE: ui_layout.py

import gradio as gr

# --- UI Constants and Configuration ---
TTS_MODELS_AVAILABLE = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
MODEL_DEFAULT_ENV = "gpt-4o-mini-tts"

SPEAKER_CONFIG_METHODS = [
    "Single Voice (Global)",
    "Random per Speaker",
    "A/B Round Robin",
    "Detailed Configuration (Per Speaker UI)",
]
DEFAULT_SPEAKER_CONFIG_METHOD = "Random per Speaker"

# Assuming utils.openai_tts.OPENAI_VOICES exists and provides a list of voice names
# If not, define APP_AVAILABLE_VOICES directly here for now.
try:
    from dialogue_tts_core.tts_client import OPENAI_VOICES as ALL_TTS_VOICES

    APP_AVAILABLE_VOICES = ALL_TTS_VOICES.copy()
    if not APP_AVAILABLE_VOICES:  # Fallback if OPENAI_VOICES is empty
        APP_AVAILABLE_VOICES = [
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
        ]  # Keep fallback
except ImportError:
    print(
        "Warning: Could not import OPENAI_VOICES from dialogue_tts_core.tts_client. "
        "Using default list."
    )
    APP_AVAILABLE_VOICES = [
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
    ]  # Keep fallback

PREDEFINED_VIBES = {
    "None": "",
    "Calm": "Speak in a calm, composed, and relaxed manner.",
    "Serene": "Speak softly and gently, radiating tranquility and warmth.",
    "Excited": "Speak with energetic enthusiasm and lively expression.",
    "Happy": "Speak with a cheerful, bright, and joyful voice.",
    "Sad": "Speak slowly with a soft, melancholic tone.",
    "Whisper": "Speak very softly, as if whispering to the listener.",
    "Angry": "Speak with a firm, tense, and forceful voice.",
    "Fearful": "Speak in a trembling, hesitant voice that conveys anxiety.",
    "Dramatic": "Speak in a low, suspenseful tone with deliberate pauses.",
    "Formal": "Speak clearly and precisely with professional neutrality.",
    "Authoritative": "Speak with commanding confidence and steady emphasis.",
    "Friendly": "Speak warmly and conversationally, sounding approachable.",
    "Playful": "Speak in a light-hearted, lively, and teasing manner.",
    "Sarcastic": "Speak with dry, wry intonation to convey ironic humor.",
    "Narrative": "Speak like a storyteller with engaging, expressive cadence.",
    "Motivational": "Speak with uplifting energy and inspiring conviction.",
    "Mysterious": "Speak slowly and quietly, hinting at hidden intrigue.",
    "Romantic": "Speak softly and tenderly with gentle affection.",
    "ASMR": "Speak very softly and slowly for a soothing, tingle effect.",
    "Corporate": "Speak in a polished, business-like and neutral tone.",
    "News": "Speak in a steady, neutral broadcast style for clear facts.",
    "Custom...": "CUSTOM",
}
VIBE_CHOICES = list(PREDEFINED_VIBES.keys())
DEFAULT_VIBE = "None"
DEFAULT_GLOBAL_VOICE = APP_AVAILABLE_VOICES[0] if APP_AVAILABLE_VOICES else "alloy"


# --- UI Element Creation Functions ---


def create_main_input_components(
    model_default_value: str,
) -> tuple[gr.TextArea, gr.Dropdown, gr.Number, gr.Slider, gr.Textbox]:
    """Creates the main input components for script, model, pause, and global
    settings."""
    with gr.Row(elem_id="main_input_row"):
        with gr.Column(scale=2, elem_id="script_input_column"):
            script_input = gr.TextArea(
                label="Dialogue Script",
                placeholder="[Speaker1] Hello world!\n[Speaker2] How are you today?",
                lines=10,
                elem_id="script_input_area",
            )
        with gr.Column(scale=1, elem_id="model_config_column"):
            tts_model_dropdown = gr.Dropdown(
                TTS_MODELS_AVAILABLE,
                label="TTS Model",
                value=model_default_value,
                elem_id="tts_model_dropdown",
            )
            pause_input = gr.Number(
                label="Pause Between Lines (ms)",
                value=500,
                minimum=0,
                maximum=5000,
                step=50,
                elem_id="pause_input",
            )

            is_tts1_family_default = model_default_value in ["tts-1", "tts-1-hd"]
            is_gpt_mini_tts_default = model_default_value == "gpt-4o-mini-tts"

            global_speed_input = gr.Slider(
                minimum=0.25,
                maximum=4.0,
                value=1.0,
                step=0.05,
                label="Global Speed (for tts-1/hd)",
                visible=is_tts1_family_default,
                interactive=True,
                elem_id="global_speed_input",
            )
            global_instructions_input = gr.Textbox(
                label="Global Instructions (for gpt-4o-mini-tts)",
                placeholder="e.g., Speak with a calm tone.",
                visible=is_gpt_mini_tts_default,
                interactive=True,
                lines=2,
                elem_id="global_instructions_input",
            )
    return (
        script_input,
        tts_model_dropdown,
        pause_input,
        global_speed_input,
        global_instructions_input,
    )


def create_speaker_config_components() -> tuple[
    gr.Dropdown, gr.Group, gr.Dropdown, gr.Column, gr.Button
]:
    """Creates components for speaker configuration method selection and the
    container for detailed UI."""
    gr.Markdown("### Speaker Voice & Style Configuration")
    speaker_config_method_dropdown = gr.Dropdown(
        SPEAKER_CONFIG_METHODS,
        label="Configuration Method",
        value=DEFAULT_SPEAKER_CONFIG_METHOD,
        elem_id="speaker_config_method_dropdown",
    )

    with gr.Group(
        visible=(DEFAULT_SPEAKER_CONFIG_METHOD == "Single Voice (Global)"),
        elem_id="single_voice_config_group",
    ) as single_voice_group:
        global_voice_dropdown = gr.Dropdown(
            APP_AVAILABLE_VOICES,
            label="Global Voice",
            value=DEFAULT_GLOBAL_VOICE,
            interactive=True,
            elem_id="global_voice_dropdown",
        )

    detailed_per_speaker_ui_group_container = gr.Column(
        visible=(
            DEFAULT_SPEAKER_CONFIG_METHOD == "Detailed Configuration (Per Speaker UI)"
        ),
        elem_id="detailed_per_speaker_ui_group_container",
    )
    with detailed_per_speaker_ui_group_container:
        load_per_speaker_ui_button = gr.Button(
            "Load/Refresh Per-Speaker Settings UI (from Script Above)",
            elem_id="load_per_speaker_ui_button",
        )
        gr.Markdown(
            "<small>Click button above to populate settings for each speaker found "
            "in the script. Settings are applied per-speaker. If script changes, "
            "click again to refresh.</small>"
        )

    return (
        speaker_config_method_dropdown,
        single_voice_group,
        global_voice_dropdown,
        detailed_per_speaker_ui_group_container,
        load_per_speaker_ui_button,
    )


def create_action_and_output_components() -> tuple[
    gr.Button, gr.Button, gr.Textbox, gr.File, gr.Audio, gr.Textbox
]:
    """Creates buttons for actions (cost, generate) and output display areas."""
    with gr.Row(elem_id="action_buttons_row"):
        calculate_cost_button = gr.Button(
            "Calculate Estimated Cost", elem_id="calculate_cost_button"
        )
        generate_button = gr.Button(
            "Generate Audio", variant="primary", elem_id="generate_button"
        )

    cost_output = gr.Textbox(
        label="Estimated Cost", interactive=False, elem_id="cost_output"
    )
    with gr.Row(elem_id="output_files_row"):
        individual_lines_zip_output = gr.File(
            label="Download Individual Lines (ZIP)",
            elem_id="individual_lines_zip_output",
        )
        merged_dialogue_mp3_output = gr.Audio(
            label="Play/Download Merged Dialogue (MP3)",
            type="filepath",
            elem_id="merged_dialogue_mp3_output",
        )
    status_output = gr.Textbox(
        label="Status",
        interactive=False,
        lines=2,
        max_lines=5,
        elem_id="status_output",
    )
    return (
        calculate_cost_button,
        generate_button,
        cost_output,
        individual_lines_zip_output,
        merged_dialogue_mp3_output,
        status_output,
    )
