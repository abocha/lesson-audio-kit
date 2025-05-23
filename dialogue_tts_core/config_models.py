# dialogue_tts_core/config_models.py

from typing import Any, Final, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from dialogue_tts_core.cost_router import QualityTier
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
            # If context is missing, skip model-specific validation and do not warn.
            # The tests that trigger this are generally focused on other aspects of
            # TTSRequestPayload validation, not the deep validation of SpeakerTTSConfig
            # which is covered by tests that DO provide context.
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

# Define available speaker config methods for the API request payload
ACCEPTED_SPEAKER_CONFIG_METHODS_FOR_API: Final = (
    "global",
    "per_speaker_configs",  # Existing, explicit per-speaker mapping
    "random_per_speaker",  # New
    "ab_round_robin",  # New
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
    tts_global_model: Literal[TTS_MODELS_AVAILABLE_TUPLE] = Field(  # type: ignore[valid-type]
        description="The global TTS model to use for synthesis."
    )
    global_pause_ms: Optional[int] = Field(
        default=None,  # Changed from 500
        ge=0,
        le=5000,
        description="Global pause in milliseconds to add between dialogue lines.",
    )
    output_format_options: Optional[OutputFormatOptions] = Field(
        default=None,  # Changed from default_factory
        description="Options for output audio formats.",
    )
    speaker_config_method: Literal[
        ACCEPTED_SPEAKER_CONFIG_METHODS_FOR_API  # type: ignore
    ] = Field(
        default="global",
        description="Strategy for selecting per-speaker TTS configurations.",
    )
    global_speaker_config: Optional[GlobalSpeakerConfig] = Field(
        default=None,
        description=(
            "Global configuration for all speakers. Used if method is 'global', "
            "or as a base for 'random_per_speaker' and 'ab_round_robin'."
        ),
    )
    per_speaker_configs: Optional[list[PerSpeakerConfigItem]] = Field(
        default=None,
        description=(
            "List of configurations for each speaker. "
            "MUST be provided if method is 'per_speaker_configs'. "
            "MUST be None or empty for 'random_per_speaker' and 'ab_round_robin'."
        ),
    )
    nsfw_check_options: Optional[NSFWCheckOptions] = Field(
        default=None,  # Changed from default_factory
        description="Options for NSFW content checking.",
    )
    user_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description=(
            "An optional identifier for the user making the request, "
            "used for free tier "
            "calculations or user-specific routing rules."
        ),
    )
    desired_quality_tier: Optional[str] = Field(
        default=QualityTier.MID.name,
        description=(
            f"Desired quality tier for TTS. Defaults to MID. Options: "
            f"{[q.name for q in QualityTier]}."
        ),
    )
    max_total_job_cost_usd: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Optional maximum total cost in USD for the entire TTS job. If set, the "
            "router will try to stay within this budget."
        ),
    )
    prefer_low_latency: Optional[bool] = Field(
        default=False,
        description=(
            "Optional preference for lower latency engines. If true, "
            "router prioritizes "
            "engines below a certain latency threshold."
        ),
    )
    prefer_emotion_support: Optional[bool] = Field(
        default=False,
        description=(
            "Optional preference for engines that support "
            "explicit emotion/style controls."
        ),
    )
    prefer_voice_cloning: Optional[bool] = Field(
        default=False,
        description=(
            "Optional preference for engines that support voice cloning features."
        ),
    )
    specific_engine_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description=(
            "Optional specific engine_id (e.g., 'openai_tts_1_hd') to use, "
            "bypassing the "
            "cost router's selection logic. The engine must exist in the ENGINE_TABLE."
        ),
    )

    @field_validator("desired_quality_tier")
    @classmethod
    def validate_quality_tier_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        try:
            QualityTier[v.upper()]
        except KeyError as e:
            raise ValueError(
                f"Invalid quality tier '{v}'. Must be one of "
                f"{[q.name for q in QualityTier]}."
            ) from e
        return v.upper()

    @field_validator("global_speaker_config", mode="before")
    @classmethod
    def pre_validate_global_speaker_config(cls, v: Any, info: ValidationInfo) -> Any:
        if isinstance(v, dict) and info.data and "tts_global_model" in info.data:
            tts_model = info.data["tts_global_model"]
            # Return a validated model instance, Pydantic will use this directly
            return GlobalSpeakerConfig.model_validate(
                v, context={"tts_global_model": tts_model}
            )
        return v  # Return as is if None, or already a model instance

    @field_validator("per_speaker_configs", mode="before")
    @classmethod
    def pre_validate_per_speaker_configs(cls, v: Any, info: ValidationInfo) -> Any:
        if isinstance(v, list) and info.data and "tts_global_model" in info.data:
            tts_model = info.data["tts_global_model"]
            validated_items = []
            for item_data in v:
                if (
                    isinstance(item_data, dict)
                    and "config" in item_data
                    and isinstance(item_data["config"], dict)
                ):
                    # Validate the nested 'config' dict and replace it
                    validated_config = SpeakerSpecificConfig.model_validate(
                        item_data["config"], context={"tts_global_model": tts_model}
                    )
                    # Create a new dict for PerSpeakerConfigItem to ensure it's
                    # processed correctly
                    # Pydantic will then parse this dict into a
                    # PerSpeakerConfigItem instance
                    validated_items.append(
                        {
                            "speaker_name": item_data.get("speaker_name"),
                            "config": validated_config,
                        }
                    )
                else:
                    # If item_data is already a PerSpeakerConfigItem or not in
                    # the expected dict format, pass through
                    validated_items.append(item_data)
            return validated_items
        return v  # Return as is if None, or already a list of model instances

    # Helper methods for validate_speaker_config_logic
    def _run_speaker_config_validation(
        self, config_data_dict: dict, context: dict, config_name: str
    ) -> None:
        """Helper to validate SpeakerTTSConfig-like model data (expected as dict)."""
        try:
            SpeakerTTSConfig.model_validate(config_data_dict, context=context)
        except ValueError as e:
            raise ValueError(f"Validation error in {config_name}: {e}") from e

    def _validate_global_config_present_for_global_method(self) -> None:
        """Rule: For 'global' method, global_speaker_config must be present."""
        if self.global_speaker_config is None:
            raise ValueError(
                "If speaker_config_method is 'global', "
                "global_speaker_config MUST be provided."
            )

    def _validate_per_speaker_configs_absent_for_global_method(self) -> None:
        """Rule: For 'global' method, per_speaker_configs must be absent."""
        if self.per_speaker_configs and len(self.per_speaker_configs) > 0:
            raise ValueError(
                "If speaker_config_method is 'global', "
                "per_speaker_configs MUST be None or empty."
            )

    def _validate_per_speaker_configs_present_and_items_valid_for_method(
        self, validation_context: dict
    ) -> None:
        """Rule: For 'per_speaker_configs', list must be present and items valid."""
        if not self.per_speaker_configs:
            raise ValueError(
                "If speaker_config_method is 'per_speaker_configs', "
                "per_speaker_configs MUST be provided and not empty."
            )
        for item_config in self.per_speaker_configs:
            try:
                # Re-validate the nested config with the main model's context
                SpeakerTTSConfig.model_validate(
                    item_config.config.model_dump(), context=validation_context
                )
            except ValueError as e:
                error_message = (
                    f"Validation error in per_speaker_configs for "
                    f"speaker '{item_config.speaker_name}': {e}"
                )
                raise ValueError(error_message) from e

    def _validate_per_speaker_configs_absent_for_random_or_ab_method(self) -> None:
        """Rule: For 'random_...' or 'ab_...' methods, per_speaker_configs absent."""
        if self.per_speaker_configs and len(self.per_speaker_configs) > 0:
            raise ValueError(
                f"If speaker_config_method is '{self.speaker_config_method}', "
                "per_speaker_configs MUST be None or empty."
            )

    @model_validator(mode="after")
    def validate_speaker_config_logic(self) -> "TTSRequestPayload":
        validation_context = {"tts_global_model": self.tts_global_model}

        # 1. Validate global_speaker_config itself if it's provided
        if self.global_speaker_config:
            self._run_speaker_config_validation(
                self.global_speaker_config.model_dump(),
                validation_context,
                "global_speaker_config",
            )

        # 2. Apply method-specific rules regarding presence/absence of configs
        if self.speaker_config_method == "global":
            self._validate_global_config_present_for_global_method()
            self._validate_per_speaker_configs_absent_for_global_method()

        elif self.speaker_config_method == "per_speaker_configs":
            self._validate_per_speaker_configs_present_and_items_valid_for_method(
                validation_context
            )
            # Note: global_speaker_config can co-exist as a fallback; its own
            # validity was checked in step 1 if it was provided.

        elif self.speaker_config_method in ["random_per_speaker", "ab_round_robin"]:
            self._validate_per_speaker_configs_absent_for_random_or_ab_method()
            # Note: global_speaker_config can co-exist as a base for non-voice params;
            # its own validity was checked in step 1 if it was provided.
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


class CacheStatsResponse(BaseModel):
    cache_size_bytes: int = Field(
        description="Current total size of the cache in bytes."
    )
    cache_size_gb: float = Field(
        description="Current total size of the cache in gigabytes."
    )
    max_cache_size_gb: float = Field(
        description="Configured maximum cache size in gigabytes."
    )
    hits: int = Field(
        description="Number of cache hits since last reset or server start."
    )
    misses: int = Field(
        description="Number of cache misses since last reset or server start."
    )
    errors: int = Field(
        description="Number of errors during cache get/store operations."
    )
    total_lookups: int = Field(
        description="Total number of cache lookups (hits + misses)."
    )
