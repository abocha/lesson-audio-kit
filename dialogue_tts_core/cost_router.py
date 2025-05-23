# dialogue_tts_core/cost_router.py

from __future__ import annotations  # For type hints if needed for older Pythons

from dataclasses import dataclass
from enum import Enum, auto
import logging
import math
from typing import Final

# Module-level constant for TTS duration estimation
CHAR_PER_SEC_EST: Final[int] = 17  # Approximately 75-80 wpm

CHARS_IN_ONE_M: Final[float] = 1_000_000.0
DEFAULT_CHARS_PER_AUDIO_MINUTE: Final[int] = (
    CHAR_PER_SEC_EST * 60
)  # e.g., 17 * 60 = 1020


class QualityTier(Enum):
    """Defines the quality tiers for TTS engines."""

    LOW = auto()  # Suitable for drafts, temporary audio, very low cost focus
    MID = auto()  # Good balance for production default, clear voice
    HIGH = auto()  # Higher fidelity, suitable for important content, marketing
    ULTRA = (
        auto()
    )  # Premium, best available, potentially for client-paid or specific features

    def __lt__(self, other: QualityTier) -> bool:
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __le__(self, other: QualityTier) -> bool:
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other: QualityTier) -> bool:
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other: QualityTier) -> bool:
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented


@dataclass(frozen=True, slots=True)
class EngineMeta:
    """Metadata for a TTS engine."""

    provider: str
    model_id: str
    latency_ms: int
    quality_tier: QualityTier
    price_per_mchar_input_usd: float | None = None
    price_per_mchar_output_audio_usd: float | None = None
    price_per_audio_minute_usd: float | None = None
    supports_emotion: bool = False
    supports_voice_cloning: bool = False


# --- Static Engine Metadata Table ---
ENGINE_TABLE: Final[dict[str, EngineMeta]] = {
    # OpenAI
    "openai_tts_1": EngineMeta(
        provider="openai",
        model_id="tts-1",
        price_per_mchar_output_audio_usd=15.0,
        latency_ms=350,
        quality_tier=QualityTier.MID,
        supports_emotion=False,
    ),
    "openai_tts_1_hd": EngineMeta(
        provider="openai",
        model_id="tts-1-hd",
        price_per_mchar_output_audio_usd=30.0,
        latency_ms=450,
        quality_tier=QualityTier.HIGH,
        supports_emotion=False,
    ),
    "openai_gpt4o_mini_tts": EngineMeta(
        provider="openai",
        model_id="gpt-4o-mini-tts",
        price_per_mchar_input_usd=0.60,
        price_per_mchar_output_audio_usd=12.0,
        latency_ms=300,
        quality_tier=QualityTier.MID,
        supports_emotion=True,
    ),
    # Fal.ai
    "fal_minimax_speech_02_turbo": EngineMeta(
        provider="fal.ai",
        model_id="minimax/speech-02-turbo",
        price_per_mchar_output_audio_usd=30.0,
        latency_ms=320,
        quality_tier=QualityTier.MID,
    ),
    "fal_minimax_speech_02_hd": EngineMeta(
        provider="fal.ai",
        model_id="minimax/speech-02-hd",
        price_per_mchar_output_audio_usd=50.0,
        latency_ms=420,
        quality_tier=QualityTier.HIGH,
    ),
    "fal_playai_tts_v3": EngineMeta(
        provider="fal.ai",
        model_id="fal-ai/playht-v2-turbo",
        price_per_audio_minute_usd=0.03,
        latency_ms=280,
        quality_tier=QualityTier.LOW,
    ),
    "fal_playai_tts_dialog": EngineMeta(
        provider="fal.ai",
        model_id="fal-ai/playht-v2-dialog",
        price_per_audio_minute_usd=0.05,
        latency_ms=350,
        quality_tier=QualityTier.HIGH,
        supports_emotion=True,
    ),
    "fal_dia_tts": EngineMeta(
        provider="fal.ai",
        model_id="dia-tts",  # Placeholder
        price_per_mchar_output_audio_usd=40.0,
        latency_ms=380,
        quality_tier=QualityTier.MID,
    ),
    "fal_orpheus_tts": EngineMeta(
        provider="fal.ai",
        model_id="orpheus-tts",  # Placeholder
        price_per_mchar_output_audio_usd=50.0,
        latency_ms=400,
        quality_tier=QualityTier.HIGH,
    ),
    # ElevenLabs (Creator Plan - overage rate)
    "elevenlabs_creator_overage": EngineMeta(
        provider="elevenlabs",
        model_id="eleven_multilingual_v2",  # Example
        price_per_mchar_output_audio_usd=150.0,
        latency_ms=260,
        quality_tier=QualityTier.ULTRA,
        supports_emotion=True,
        supports_voice_cloning=True,
    ),
    # Cartesia
    "cartesia_sonic_startup": EngineMeta(
        provider="cartesia",
        model_id="sonic-english",
        price_per_mchar_output_audio_usd=39.0,
        latency_ms=90,
        quality_tier=QualityTier.MID,
        supports_emotion=True,
        supports_voice_cloning=True,
    ),
    # Google (Gemini 2.5 Flash/Pro Preview TTS)
    "google_gemini_2_5_flash_tts": EngineMeta(
        provider="google",
        model_id="gemini-2.5-flash-tts-preview",  # Placeholder
        price_per_mchar_output_audio_usd=10.0,
        latency_ms=400,
        quality_tier=QualityTier.MID,
        supports_emotion=True,
    ),
    "google_gemini_2_5_pro_tts": EngineMeta(
        provider="google",
        model_id="gemini-2.5-pro-tts-preview",  # Placeholder
        price_per_mchar_output_audio_usd=20.0,
        latency_ms=600,
        quality_tier=QualityTier.HIGH,
        supports_emotion=True,
    ),
    # AWS Polly Standard
    "aws_polly_standard": EngineMeta(
        provider="aws",
        model_id="polly_standard_joanna",  # Example
        price_per_mchar_output_audio_usd=4.0,
        latency_ms=500,
        quality_tier=QualityTier.LOW,
        supports_emotion=False,
    ),
    # Google Cloud Standard
    "google_cloud_standard_tts": EngineMeta(
        provider="google",
        model_id="cloud_standard_wave_a",  # Example
        price_per_mchar_output_audio_usd=4.0,
        latency_ms=550,
        quality_tier=QualityTier.LOW,
        supports_emotion=False,
    ),
}

# --- Free Tier Information ---
FREE_TIERS: Final[dict[str, int]] = {
    "cartesia_sonic-english": 20_000,
    "aws_polly_standard_joanna": 5_000_000,
    "google_cloud_standard_wave_a": 1_000_000,
    "elevenlabs_eleven_multilingual_v2": 100_000,
}

FREE_POOL: Final[list[str]] = [
    "cartesia_sonic-english",
    "aws_polly_standard_joanna",
    "google_cloud_standard_wave_a",
]

# --- Mock/Stub for User Quota Tracking (Replace with actual implementation later) ---
# This would interact with a database (e.g., the one mentioned in monetization.md)
_MOCK_USER_FREE_QUOTAS: dict[str, dict[str, int]] = {
    "user123": {  # Example user
        "cartesia_sonic-english": 15000,  # Remaining chars for Cartesia
        "aws_polly_standard_joanna": 4000000,
    }
}


def _get_remaining_free_quota(user_id: str | None, engine_id: str) -> int:
    """
    STUB: Returns the remaining free quota for a user and engine.
    In a real system, this would query a usage tracking database.
    """
    if user_id and user_id in _MOCK_USER_FREE_QUOTAS:
        return _MOCK_USER_FREE_QUOTAS[user_id].get(engine_id, 0)

    return FREE_TIERS.get(engine_id, 0)


# --- Helper for Cost Calculation ---
def _calculate_effective_cost_per_mchar(
    engine: EngineMeta, _char_len_for_job: int
) -> float:  # Note: _char_len_for_job
    """
    Calculates an effective cost per 1 Million characters for a given engine.
    Now explicitly handles separate input and output audio costs for applicable models.
    """
    # For models billed per minute of generated audio:
    if engine.price_per_audio_minute_usd is not None:
        if DEFAULT_CHARS_PER_AUDIO_MINUTE <= 0:
            raise ValueError(
                "DEFAULT_CHARS_PER_AUDIO_MINUTE must be positive "
                "for minute-based cost calculation."
            )

        cost_audio_output_per_mchar = (
            engine.price_per_audio_minute_usd * CHARS_IN_ONE_M
        ) / DEFAULT_CHARS_PER_AUDIO_MINUTE

        if engine.price_per_mchar_input_usd is not None:
            return cost_audio_output_per_mchar + engine.price_per_mchar_input_usd
        return cost_audio_output_per_mchar

    # For models billed per character for audio output
    # (potentially with separate input cost)
    if engine.price_per_mchar_output_audio_usd is not None:
        cost_audio_output_per_mchar = engine.price_per_mchar_output_audio_usd

        if engine.price_per_mchar_input_usd is not None:
            return cost_audio_output_per_mchar + engine.price_per_mchar_input_usd
        return cost_audio_output_per_mchar

    # If only input cost is defined
    if engine.price_per_mchar_input_usd is not None:
        logging.warning(
            "Engine '%s/%s' only has input cost. Assuming audio output bundled.",
            engine.provider,
            engine.model_id,
        )
        return engine.price_per_mchar_input_usd

    # Default case if no other pricing model matched (contents of former else block)
    logging.warning(
        "Engine '%s/%s' has no recognized pricing scheme. Cost set to infinity.",
        engine.provider,
        engine.model_id,
    )
    return float("inf")


# --- Router Logic ---
LATENCY_THRESHOLD_MS: Final[int] = 350


def select_engine(  # noqa: C901
    char_len: int,
    desired_quality: QualityTier = QualityTier.MID,
    max_cost_usd_for_job: float | None = None,
    prefer_low_latency: bool = False,
    prefer_emotion_support: bool = False,
    prefer_voice_cloning: bool = False,
    user_id: str | None = None,
) -> EngineMeta:
    """
    Selects the most appropriate TTS engine based on constraints.
    Considers free tiers, quality, latency, budget, and feature preferences.
    """
    all_engines: list[EngineMeta] = list(ENGINE_TABLE.values())

    # Step 1: Filter by desired quality (all engines >= desired_quality)
    quality_candidates = [
        engine
        for engine in all_engines
        if engine.quality_tier.value >= desired_quality.value
    ]
    if not quality_candidates:
        raise RuntimeError(f"No engines match desired quality: {desired_quality.name}")

    # Step 2: Determine preference availability
    # among all quality candidates (>= desired_quality)
    low_latency_possible_among_quality_candidates = any(
        e.latency_ms <= LATENCY_THRESHOLD_MS for e in quality_candidates
    )
    emotion_possible_among_quality_candidates = any(
        e.supports_emotion for e in quality_candidates
    )
    voice_cloning_possible_among_quality_candidates = any(
        e.supports_voice_cloning for e in quality_candidates
    )

    # Determine preference availability specifically within the desired quality tier
    desired_tier_only_candidates = [
        e for e in quality_candidates if e.quality_tier == desired_quality
    ]
    low_latency_possible_in_desired_tier = any(
        e.latency_ms <= LATENCY_THRESHOLD_MS for e in desired_tier_only_candidates
    )
    emotion_possible_in_desired_tier = any(
        e.supports_emotion for e in desired_tier_only_candidates
    )
    voice_cloning_possible_in_desired_tier = any(
        e.supports_voice_cloning for e in desired_tier_only_candidates
    )

    # Step 3: Apply preferences as hard filters IF possible within the desired tier.
    # If not possible within desired tier, log warning and DO NOT filter.
    candidate_engines = list(
        quality_candidates
    )  # Start with all engines >= desired_quality

    if prefer_low_latency:
        if low_latency_possible_in_desired_tier:
            candidate_engines = [
                engine
                for engine in candidate_engines
                if engine.latency_ms <= LATENCY_THRESHOLD_MS
            ]
        else:
            logging.warning("No low-latency engines found in desired quality tier.")

    if prefer_emotion_support:
        if emotion_possible_in_desired_tier:
            candidate_engines = [
                engine for engine in candidate_engines if engine.supports_emotion
            ]
        else:
            logging.warning(
                "No emotion-supporting engines found in desired quality tier."
            )

    if prefer_voice_cloning:
        if voice_cloning_possible_in_desired_tier:
            candidate_engines = [
                engine for engine in candidate_engines if engine.supports_voice_cloning
            ]
        else:
            logging.warning(
                "No voice cloning-supporting engines found in desired quality tier."
            )

    if not candidate_engines:
        raise RuntimeError("No engines meet all specified criteria.")

    # Step 4: Prepare candidates with cost, quality gap, and preference penalties
    priced_candidates: list[tuple[EngineMeta, float, bool, int, int, int, int]] = []

    for engine in candidate_engines:
        uses_free_tier = False
        job_cost_for_engine = 0.0
        engine_key = f"{engine.provider}_{engine.model_id}"

        if engine_key in FREE_POOL:
            remaining_quota = _get_remaining_free_quota(user_id, engine_key)
            if remaining_quota >= char_len:
                uses_free_tier = True
                job_cost_for_engine = 0.0000001

        if not uses_free_tier:
            job_cost_for_engine = (
                _calculate_effective_cost_per_mchar(engine, char_len) / CHARS_IN_ONE_M
            ) * char_len

        if math.isinf(job_cost_for_engine):
            continue

        quality_gap = abs(engine.quality_tier.value - desired_quality.value)

        pref_lat_penalty = 1
        if prefer_low_latency:
            if (
                low_latency_possible_among_quality_candidates
                and engine.latency_ms <= LATENCY_THRESHOLD_MS
            ):
                pref_lat_penalty = 0
            elif not low_latency_possible_among_quality_candidates:
                pref_lat_penalty = 0  # Neutralize penalty if preference is impossible
        else:
            pref_lat_penalty = 0

        pref_emotion_penalty = 1
        if prefer_emotion_support:
            if emotion_possible_among_quality_candidates and engine.supports_emotion:
                pref_emotion_penalty = 0
            elif not emotion_possible_among_quality_candidates:
                pref_emotion_penalty = (
                    0  # Neutralize penalty if preference is impossible
                )
        else:
            pref_emotion_penalty = 0

        pref_clone_penalty = 1
        if prefer_voice_cloning:
            if (
                voice_cloning_possible_among_quality_candidates
                and engine.supports_voice_cloning
            ):
                pref_clone_penalty = 0
            elif not voice_cloning_possible_among_quality_candidates:
                pref_clone_penalty = 0  # Neutralize penalty if preference is impossible
        else:
            pref_clone_penalty = 0

        priced_candidates.append(
            (
                engine,
                job_cost_for_engine,
                uses_free_tier,
                quality_gap,
                pref_lat_penalty,
                pref_emotion_penalty,
                pref_clone_penalty,
            )
        )

    if not priced_candidates:
        raise RuntimeError("No priceable engines meet prior constraints.")

    if max_cost_usd_for_job is not None:
        budget_met_candidates = [
            (eng, cost, is_free, q_gap, lat_pen, emo_pen, clone_pen)
            for (
                eng,
                cost,
                is_free,
                q_gap,
                lat_pen,
                emo_pen,
                clone_pen,
            ) in priced_candidates
            if cost <= max_cost_usd_for_job
        ]
        if not budget_met_candidates:
            raise RuntimeError(
                f"No engines meet budget ${max_cost_usd_for_job:.4f} for {char_len} "
                f"chars from current candidates."
            )
        priced_candidates = budget_met_candidates

    if not priced_candidates:
        raise RuntimeError("No viable engines found after budget filtering.")

    # Step 5: Sort candidates based on the new criteria
    priced_candidates.sort(
        key=lambda x: (
            not x[2],  # 1. Free tier first (False for "not is_free" means free,
            # so free comes first)
            x[1],  # 2. Then by job cost
            x[3],  # 3. Then by quality_gap (absolute distance from desired quality)
            x[4],  # 4. Then by low latency preference penalty
            x[5],  # 5. Then by emotion support preference penalty
            x[6],  # 6. Then by voice cloning preference penalty
            x[0].latency_ms,  # 7. Then by raw latency
            -x[0].quality_tier.value,  # 8. Then by quality (higher is better,
            # hence negative)
        )
    )

    return priced_candidates[0][0]
