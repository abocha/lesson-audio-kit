# dialogue_tts_core/config_models.py

from typing import Final, Literal, Optional, Union
import warnings  # For warning if context is missing

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from dialogue_tts_core.tts_client import OPENAI_VOICES
from gradio_frontend.ui_layout import VIBE_CHOICES

# Define available OpenAI voices using Literal for validation
# Sourced from dialogue_tts_core.tts_client.OPENAI_VOICES
OPENAI_VOICES_TUPLE: Final = tuple(OPENAI_VOICES)

# Define available Vibe choices
# Sourced from gradio_frontend.ui_layout.VIBE_CHOICES
VIBE_CHOICES_TUPLE: Final = tuple(VIBE_CHOICES)


class SpeakerTTSConfig(BaseModel):
    voice: Literal[OPENAI_VOICES_TUPLE] = Field(  # type: ignore[valid-type]
        default=OPENAI_VOICES_TUPLE[0] if OPENAI_VOICES_TUPLE else "alloy",
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

    vibe: Optional[Literal[VIBE_CHOICES_TUPLE]] = Field(  # type: ignore[valid-type]
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

    @model_validator(mode="after")
    def validate_conditional_fields(self, info: ValidationInfo) -> "SpeakerTTSConfig":
        model_type = info.context.get("tts_global_model") if info.context else None

        if not model_type:
            warnings.warn(
                "SpeakerTTSConfig validation for model-specific fields skipped "
                "due to missing 'tts_global_model' in validation context.",
                UserWarning,
                stacklevel=2,
            )
            return self

        if model_type == "gpt-4o-mini-tts":
            if self.speed is not None and self.speed != 1.0:
                raise ValueError(
                    f"For TTS model '{model_type}', speed must be 1.0 or not set. "
                    f"Got speed: {self.speed}."
                )
        elif model_type in ["tts-1", "tts-1-hd"]:
            if self.vibe is not None and self.vibe != "None":
                raise ValueError(
                    f"For TTS model '{model_type}', vibe must be 'None' or not set. "
                    f"Got vibe: '{self.vibe}'."
                )
            if self.custom_instructions is not None:
                raise ValueError(
                    f"For TTS model '{model_type}', "
                    "custom_instructions must not be set. "
                    f"Got: '{self.custom_instructions}'.",
                )
        return self


# Define available TTS models
# TODO: In a future refactor, consider sourcing from or syncing with
# gradio_frontend.ui_layout.TTS_MODELS_AVAILABLE (this is already fairly synced)
TTS_MODELS_AVAILABLE_TUPLE: Final = (
    "tts-1-hd",
    "gpt-4o-mini-tts",
    "tts-1",
)

# Define available speaker config methods for general use
SPEAKER_CONFIG_METHOD_TUPLE: Final = (
    "global",
    "per_speaker",
    "random_per_speaker",
    "ab_round_robin",
)

# Define speaker config methods specifically for TTSRequestPayload validation
TTS_REQUEST_SPEAKER_CONFIG_METHODS_TUPLE: Final = ("global", "per_speaker_configs")


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
    tts_global_model: Literal[TTS_MODELS_AVAILABLE_TUPLE] = Field(  # type: ignore[valid-type]
        description="The global TTS model to use for synthesis."
    )
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
        TTS_REQUEST_SPEAKER_CONFIG_METHODS_TUPLE  # type: ignore[valid-type]
    ] = Field(description="Method to determine speaker configurations.")
    global_speaker_config: Optional[GlobalSpeakerConfig] = Field(
        default=None,  # Changed from default_factory for explicit None checks
        description=(
            "Global configuration for all speakers, used if method is 'global'."
        ),
    )
    per_speaker_configs: Optional[list[PerSpeakerConfigItem]] = Field(
        default=None,
        description=(
            "List of configurations for each speaker, used if method is "
            "'per_speaker_configs'."
        ),
    )
    nsfw_check_options: Optional[NSFWCheckOptions] = Field(
        default_factory=NSFWCheckOptions,
        description="Options for NSFW content checking.",
    )

    @model_validator(mode="after")
    def validate_speaker_config_logic(self) -> "TTSRequestPayload":
        if self.speaker_config_method == "global":
            if self.global_speaker_config is None:
                raise ValueError(
                    "If speaker_config_method is 'global', "
                    "global_speaker_config must be provided."
                )
            if (
                self.per_speaker_configs is not None
                and len(self.per_speaker_configs) > 0
            ):
                raise ValueError(
                    "If speaker_config_method is 'global', "
                    "per_speaker_configs must be None or empty."
                )
        elif self.speaker_config_method == "per_speaker_configs":
            if not self.per_speaker_configs:  # Checks for None or empty list
                raise ValueError(
                    "If speaker_config_method is 'per_speaker_configs', "
                    "per_speaker_configs must be provided and not empty."
                )
            if self.global_speaker_config is not None:
                raise ValueError(
                    "If speaker_config_method is 'per_speaker_configs', "
                    "global_speaker_config must be None."
                )
        return self


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
