# dialogue_tts_core/config_models.py

from typing import Final, Literal, Optional

from pydantic import BaseModel, Field

# Define available OpenAI voices using Literal for validation
# TODO: In a future refactor, this list should be sourced from
# dialogue_tts_core.tts_client.OPENAI_VOICES
# For now, use the direct definition.
OPENAI_VOICES_TUPLE: Final = (
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
)  # This should be a tuple for Literal

# Define available Vibe choices
# TODO: In a future refactor, consider sourcing from or syncing with
# gradio_frontend.ui_layout.VIBE_CHOICES
# For now, use the direct definition.
VIBE_CHOICES_TUPLE: Final = (
    "None",
    "Calm",
    "Serene",
    "Excited",
    "Happy",
    "Sad",
    "Whisper",
    "Angry",
    "Fearful",
    "Dramatic",
    "Formal",
    "Authoritative",
    "Friendly",
    "Playful",
    "Sarcastic",
    "Narrative",
    "Motivational",
    "Mysterious",
    "Romantic",
    "ASMR",
    "Corporate",
    "News",
    "Custom...",
)  # This should be a tuple for Literal


class SpeakerTTSConfig(BaseModel):
    voice: Literal[
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
    ] = Field(
        default=OPENAI_VOICES_TUPLE[0],
        description=(
            "The specific voice to use for the speaker. Must be one of "
            "the standard OpenAI voices."
        ),
    )

    speed: Optional[float] = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description=(
            "Speech speed. Relevant for tts-1 and tts-1-hd models. Range: 0.25 to 4.0."
        ),
    )

    vibe: Optional[
        Literal[
            "None",
            "Calm",
            "Serene",
            "Excited",
            "Happy",
            "Sad",
            "Whisper",
            "Angry",
            "Fearful",
            "Dramatic",
            "Formal",
            "Authoritative",
            "Friendly",
            "Playful",
            "Sarcastic",
            "Narrative",
            "Motivational",
            "Mysterious",
            "Romantic",
            "ASMR",
            "Corporate",
            "News",
            "Custom...",
        ]
    ] = Field(
        default="None",
        description=(
            "Predefined emotional vibe or style for gpt-4o-mini-tts. "
            "E.g., 'Calm', 'Excited'."
        ),
    )

    custom_instructions: Optional[str] = Field(
        default=None,
        max_length=500,
        description=(
            "Specific text-based instructions for the TTS engine, "
            "e.g., for gpt-4o-mini-tts."
        ),
    )

    # Potential future Pydantic model_validator:
    # IF model_type (passed in context or as another field) is tts-1/tts-1-hd THEN
    #   custom_instructions and vibe should ideally be None or ignored.
    # IF model_type is gpt-4o-mini-tts THEN
    #   speed should ideally be 1.0 or ignored.
    # For now, the orchestrating function will handle applying relevant
    # fields based on the global TTS model.
