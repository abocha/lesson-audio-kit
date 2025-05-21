# dialogue_tts_core/config_models.py

from typing import Final, Literal, Optional, Union

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


# Define available TTS models
# TODO: In a future refactor, consider sourcing from or syncing with
# gradio_frontend.ui_layout.TTS_MODELS_AVAILABLE
TTS_MODELS_AVAILABLE_TUPLE: Final = (
    "tts-1-hd",
    "gpt-4o-mini-tts",
    "tts-1",  # Assuming tts-1 is also a valid option
)

# Define available speaker config methods
SPEAKER_CONFIG_METHOD_TUPLE: Final = (
    "global",
    "per_speaker",
    "random_per_speaker",
    "ab_round_robin",
)


class OutputFormatOptions(BaseModel):
    include_individual_lines_zip: bool = Field(
        default=True,
        description="Whether to include a ZIP file of individual audio lines.",
    )
    include_merged_dialogue_mp3: bool = Field(
        default=True,
        description="Whether to include a single merged MP3 of the dialogue.",
    )


# GlobalSpeakerConfig can reuse SpeakerTTSConfig as the structure is identical
GlobalSpeakerConfig = SpeakerTTSConfig

# SpeakerSpecificConfig can also reuse SpeakerTTSConfig
SpeakerSpecificConfig = SpeakerTTSConfig


class PerSpeakerConfigItem(BaseModel):
    speaker_name: str = Field(description="The name of the speaker.")
    config: SpeakerSpecificConfig = Field(
        description="Speaker-specific TTS configuration."
    )


class NSFWCheckOptions(BaseModel):
    enabled: bool = Field(default=False, description="Whether NSFW check is enabled.")
    api_url_template: Optional[str] = Field(
        default=None, description="URL template for the NSFW check API."
    )


class TTSRequestPayload(BaseModel):
    script_text: str = Field(
        max_length=10000, description="The script text to be synthesized."
    )
    tts_global_model: Literal["tts-1-hd", "gpt-4o-mini-tts", "tts-1"] = Field(
        description="The global TTS model to use for synthesis."
    )  # TODO: Use TTS_MODELS_AVAILABLE_TUPLE
    global_pause_ms: Optional[int] = Field(
        default=500,
        ge=0,
        le=5000,
        description="Global pause in milliseconds to add between dialogue lines.",
    )
    output_format_options: Optional[OutputFormatOptions] = Field(
        default_factory=OutputFormatOptions,
        description="Options for output audio formats.",
    )
    speaker_config_method: Literal[
        "global", "per_speaker", "random_per_speaker", "ab_round_robin"
    ] = Field(
        description="Method to determine speaker configurations."
    )  # TODO: Use SPEAKER_CONFIG_METHOD_TUPLE
    global_speaker_config: Optional[GlobalSpeakerConfig] = Field(
        default_factory=GlobalSpeakerConfig,
        description=(
            "Global configuration for all speakers, used if method is 'global'."
        ),
    )
    per_speaker_configs: Optional[list[PerSpeakerConfigItem]] = Field(
        default=None,
        description=(
            "List of configurations for each speaker, used if method is 'per_speaker'."
        ),
    )
    nsfw_check_options: Optional[NSFWCheckOptions] = Field(
        default_factory=NSFWCheckOptions,
        description="Options for NSFW content checking.",
    )

    # TODO: Add Pydantic model_validator to ensure:
    # 1. If speaker_config_method is "global", global_speaker_config must be provided.
    # 2. If speaker_config_method is "per_speaker",
    #    per_speaker_configs must be provided and not empty.
    # 3. Other methods might have their own validation logic.


# --- API Response Payload Models ---


class TTSJobCreationResponse(BaseModel):
    job_id: str = Field(description="Unique identifier for the synthesis job.")
    status_url: str = Field(
        description="URL to poll for job status, e.g., /api/tts/status/{job_id}."
    )
    estimated_completion_time: Optional[str] = Field(
        default=None, description="Estimated completion time, e.g., '30s-2min'."
    )


class TTSJobStatusPending(BaseModel):
    job_id: str = Field(description="Unique identifier for the synthesis job.")
    status: Literal["pending", "processing"] = Field(
        description="Current status of the job."
    )
    progress: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Progress of the job (0.0-1.0)."
    )


class TTSJobOutputs(BaseModel):
    zip_file_url: Optional[str] = Field(
        default=None,
        description="URL to the ZIP file containing individual audio lines.",
    )
    merged_mp3_url: Optional[str] = Field(
        default=None, description="URL to the merged MP3 dialogue file."
    )


class TTSSynthesisDetail(BaseModel):
    id: int = Field(description="Identifier for the synthesis line/item.")
    speaker: str = Field(description="Speaker for this line.")
    status: str = Field(
        description="Status of synthesis for this line (e.g., 'success', 'failed')."
    )
    duration_ms: Optional[int] = Field(
        default=None,
        description="Duration of the synthesized audio for this line in milliseconds.",
    )
    error: Optional[str] = Field(
        default=None, description="Error message if synthesis failed for this line."
    )


class TTSJobStatusCompleted(BaseModel):
    job_id: str = Field(description="Unique identifier for the synthesis job.")
    status: Literal["completed"] = Field(description="Current status of the job.")
    message: str = Field(
        description=(
            "A message regarding the job completion, from orchestrate_tts_synthesis."
        ),
    )
    outputs: TTSJobOutputs = Field(description="Output file URLs.")
    synthesis_details: Optional[list[TTSSynthesisDetail]] = Field(
        default=None, description="Details of each synthesis line item."
    )


class TTSJobStatusFailed(BaseModel):
    job_id: str = Field(description="Unique identifier for the synthesis job.")
    status: Literal["failed"] = Field(description="Current status of the job.")
    error_message: str = Field(description="Message describing the failure.")


TTSJobStatusResponse = Union[
    TTSJobStatusPending, TTSJobStatusCompleted, TTSJobStatusFailed
]
