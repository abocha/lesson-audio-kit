# dialogue_tts_core/speaker_config_resolver.py
import itertools
import random
from typing import Optional

from .config_models import (
    OPENAI_VOICES_TUPLE,
    GlobalSpeakerConfig,  # Assuming this might be used or aliased
    SpeakerTTSConfig,
    TTSRequestPayload,
)


# Helper function (can be part of this module or dialogue_script_parser)
def get_unique_speakers_from_parsed_script(parsed_lines: list[dict]) -> list[str]:
    seen = set()
    ordered_unique = []
    for line in parsed_lines:
        speaker = line.get("speaker")
        if speaker and speaker not in seen:
            ordered_unique.append(speaker)
            seen.add(speaker)
    return ordered_unique


def _get_base_config_for_random_methods(
    global_speaker_config: Optional[GlobalSpeakerConfig],
    tts_global_model: Optional[str],
) -> SpeakerTTSConfig:
    """
    Creates a base SpeakerTTSConfig from the global_speaker_config.
    This is used as a template for 'random_per_speaker' and 'ab_round_robin'.
    """
    if global_speaker_config:
        try:
            return SpeakerTTSConfig.model_validate(
                global_speaker_config.model_dump(),
                context={"tts_global_model": tts_global_model},
            )
        except ValueError as e:
            # This implies an issue not caught by payload validation or a type mismatch.
            raise ValueError(
                "Error validating global_speaker_config as base for random methods: "
                f"{e}"
            ) from e
    return SpeakerTTSConfig()  # Default empty config if no global is provided


def _resolve_global_config(
    payload: TTSRequestPayload, unique_script_speakers: list[str]
) -> dict[str, SpeakerTTSConfig]:
    """Resolves speaker configurations using the 'global' method."""
    resolved_configs_map: dict[str, SpeakerTTSConfig] = {}
    if payload.global_speaker_config is None:
        # This should ideally be caught by Pydantic model validation beforehand.
        raise ValueError(
            "Internal Error: global_speaker_config is required for 'global' method "
            "but was not found post-validation."
        )

    try:
        validated_global_config = SpeakerTTSConfig.model_validate(
            payload.global_speaker_config.model_dump(),
            context={"tts_global_model": payload.tts_global_model},
        )
    except ValueError as e:
        raise ValueError(
            f"Invalid global_speaker_config during 'global' resolution: {e}"
        ) from e

    for speaker_name in unique_script_speakers:
        resolved_configs_map[speaker_name] = validated_global_config.model_copy(
            deep=True
        )
    return resolved_configs_map


def _resolve_per_speaker_configs(
    payload: TTSRequestPayload, unique_script_speakers: list[str]
) -> dict[str, SpeakerTTSConfig]:
    """Resolves speaker configurations using the 'per_speaker_configs' method."""
    resolved_configs_map: dict[str, SpeakerTTSConfig] = {}
    # Import moved here as it's only used in this function
    from .config_models import PerSpeakerConfigItem

    if not payload.per_speaker_configs:
        # This should ideally be caught by Pydantic model validation.
        raise ValueError(
            "Internal Error: per_speaker_configs is required for "
            "'per_speaker_configs' method but was not found post-validation."
        )

    # Ensure per_speaker_configs is not None before list comprehension
    # This check is more for type safety with linters, Pydantic should ensure it.
    current_per_speaker_configs: list[PerSpeakerConfigItem] = (
        payload.per_speaker_configs or []
    )

    temp_configs_from_payload: dict[str, SpeakerTTSConfig] = {
        item.speaker_name: item.config for item in current_per_speaker_configs
    }

    base_for_merging_data: Optional[dict] = None
    if payload.global_speaker_config:
        base_for_merging_data = payload.global_speaker_config.model_dump(
            exclude_unset=True
        )

    for speaker_name in unique_script_speakers:
        specific_config_data_model = temp_configs_from_payload.get(speaker_name)
        merged_data = {}

        if base_for_merging_data:
            merged_data.update(base_for_merging_data)

        if specific_config_data_model:
            merged_data.update(
                specific_config_data_model.model_dump(exclude_unset=True)
            )
        elif not base_for_merging_data:
            # No specific config for this speaker AND no global config to fall back on.
            raise ValueError(
                f"Configuration missing for speaker: {speaker_name} in "
                "'per_speaker_configs' method and no global fallback provided."
            )

        # Ensure merged_data is not empty if we reached here.
        # If specific_config_data_model was None, base_for_merging_data must exist.
        if not merged_data:
            # This case should be covered by the elif above.
            # Line broken for length
            error_msg = (
                f"Critical Error: No configuration data found for speaker: "
                f"{speaker_name} despite checks. This indicates a logic flaw."
            )
            raise ValueError(error_msg)

        try:
            resolved_configs_map[speaker_name] = SpeakerTTSConfig.model_validate(
                merged_data, context={"tts_global_model": payload.tts_global_model}
            )
        except ValueError as e:
            raise ValueError(
                f"Invalid merged config for speaker '{speaker_name}' in "
                f"'per_speaker_configs': {e}"
            ) from e
    return resolved_configs_map


def _resolve_random_per_speaker(
    payload: TTSRequestPayload,
    unique_script_speakers: list[str],
    base_config: SpeakerTTSConfig,
) -> dict[str, SpeakerTTSConfig]:
    """Resolves speaker configurations using the 'random_per_speaker' method."""
    resolved_configs_map: dict[str, SpeakerTTSConfig] = {}
    available_voices = list(OPENAI_VOICES_TUPLE)
    if not available_voices:
        raise RuntimeError(
            "No voices available in OPENAI_VOICES_TUPLE for random assignment."
        )

    for speaker_name in unique_script_speakers:
        chosen_voice = random.choice(available_voices)
        temp_cfg_data = base_config.model_dump(exclude_unset=True)
        temp_cfg_data["voice"] = chosen_voice

        try:
            resolved_configs_map[speaker_name] = SpeakerTTSConfig.model_validate(
                temp_cfg_data,
                context={"tts_global_model": payload.tts_global_model},
            )
        except ValueError as e:
            # Using RuntimeError as this implies an internal issue if validation fails
            # with a randomly chosen valid voice and base config.
            raise RuntimeError(
                f"Internal error resolving random config for {speaker_name}: {e}"
            ) from e
    return resolved_configs_map


def _resolve_ab_round_robin(
    payload: TTSRequestPayload,
    unique_script_speakers: list[str],
    base_config: SpeakerTTSConfig,
) -> dict[str, SpeakerTTSConfig]:
    """Resolves speaker configurations using the 'ab_round_robin' method."""
    resolved_configs_map: dict[str, SpeakerTTSConfig] = {}
    available_voices = list(OPENAI_VOICES_TUPLE)
    if not available_voices:
        raise RuntimeError(
            "No voices available in OPENAI_VOICES_TUPLE for A/B round robin."
        )

    voice_cycle = itertools.cycle(available_voices)
    for speaker_name in unique_script_speakers:
        chosen_voice = next(voice_cycle)
        temp_cfg_data = base_config.model_dump(exclude_unset=True)
        temp_cfg_data["voice"] = chosen_voice

        try:
            resolved_configs_map[speaker_name] = SpeakerTTSConfig.model_validate(
                temp_cfg_data,
                context={"tts_global_model": payload.tts_global_model},
            )
        except ValueError as e:
            raise RuntimeError(
                f"Internal error resolving round-robin config for {speaker_name}: {e}"
            ) from e
    return resolved_configs_map


def resolve_speaker_configurations(
    payload: TTSRequestPayload, unique_script_speakers: list[str]
) -> dict[str, SpeakerTTSConfig]:
    """
    Resolves TTS configurations for each unique speaker in a script based on
    the specified method in the payload.
    """
    method = payload.speaker_config_method

    if method == "global":
        return _resolve_global_config(payload, unique_script_speakers)

    if method == "per_speaker_configs":
        return _resolve_per_speaker_configs(payload, unique_script_speakers)

    # For methods requiring a base config (potentially from global_speaker_config)
    base_config_for_random = _get_base_config_for_random_methods(
        payload.global_speaker_config, payload.tts_global_model
    )

    if method == "random_per_speaker":
        return _resolve_random_per_speaker(
            payload, unique_script_speakers, base_config_for_random
        )

    if method == "ab_round_robin":
        return _resolve_ab_round_robin(
            payload, unique_script_speakers, base_config_for_random
        )

    # Fallback or error for unknown method, though Pydantic should prevent this.
    # However, defensive coding suggests handling it.
    raise ValueError(f"Unsupported speaker_config_method: {method}")
