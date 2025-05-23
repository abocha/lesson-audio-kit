from unittest.mock import MagicMock, patch

from _pytest.logging import LogCaptureFixture
import pytest
from pytest import MonkeyPatch

from dialogue_tts_core.cost_router import (
    CHARS_IN_ONE_M,
    DEFAULT_CHARS_PER_AUDIO_MINUTE,
    LATENCY_THRESHOLD_MS,
    EngineMeta,
    QualityTier,
    _calculate_effective_cost_per_mchar,
    select_engine,
)


# --- Test Data and Fixtures ---
@pytest.fixture
def mock_engine_table(monkeypatch: MonkeyPatch) -> dict[str, EngineMeta]:
    # Provide a controlled engine table for most tests
    test_engines = {
        "test_cheapest_mid": EngineMeta(
            "test_provider",
            "cheapest_mid",
            price_per_mchar_output_audio_usd=10.0,
            latency_ms=300,
            quality_tier=QualityTier.MID,
            supports_emotion=False,
        ),
        "test_cheap_mid_low_latency": EngineMeta(
            "test_provider",
            "cheap_mid_low_latency",
            price_per_mchar_output_audio_usd=12.0,
            latency_ms=100,
            quality_tier=QualityTier.MID,
            supports_emotion=True,
        ),
        "test_expensive_high_fast": EngineMeta(
            "test_provider",
            "expensive_high_fast",
            price_per_mchar_output_audio_usd=50.0,
            latency_ms=150,
            quality_tier=QualityTier.HIGH,
            supports_emotion=True,
        ),
        "test_expensive_high_slow": EngineMeta(
            "test_provider",
            "expensive_high_slow",
            price_per_mchar_output_audio_usd=45.0,
            latency_ms=500,
            quality_tier=QualityTier.HIGH,
            supports_emotion=False,
        ),
        "test_ultra_expensive": EngineMeta(
            "test_provider",
            "ultra_expensive",
            price_per_mchar_output_audio_usd=200.0,
            latency_ms=200,
            quality_tier=QualityTier.ULTRA,
            supports_emotion=True,
        ),
        "test_minute_based_mid": EngineMeta(
            "test_provider",
            "minute_mid",
            price_per_audio_minute_usd=0.02,
            latency_ms=400,
            quality_tier=QualityTier.MID,
        ),
        "test_dual_meter_mid": EngineMeta(
            "test_provider",
            "dual_mid",
            price_per_mchar_input_usd=1.0,
            price_per_mchar_output_audio_usd=10.0,
            latency_ms=250,
            quality_tier=QualityTier.MID,
            supports_emotion=True,
        ),
        "test_unpriceable": EngineMeta(
            "test_provider", "unpriceable", latency_ms=300, quality_tier=QualityTier.MID
        ),
        "test_low_quality": EngineMeta(
            "test_provider",
            "low_q",
            price_per_mchar_output_audio_usd=5.0,
            latency_ms=600,
            quality_tier=QualityTier.LOW,
        ),
        "cartesia_mock": EngineMeta(
            "cartesia",
            "sonic-test",
            price_per_mchar_output_audio_usd=39.0,
            latency_ms=90,
            quality_tier=QualityTier.MID,
            supports_emotion=True,
        ),
    }
    monkeypatch.setattr("dialogue_tts_core.cost_router.ENGINE_TABLE", test_engines)
    return test_engines


@pytest.fixture
def mock_free_tiers_config(monkeypatch: MonkeyPatch) -> None:
    test_free_tiers = {"cartesia_sonic-test": 20000}  # 20k free chars
    monkeypatch.setattr("dialogue_tts_core.cost_router.FREE_TIERS", test_free_tiers)
    test_free_pool = ["cartesia_sonic-test"]
    monkeypatch.setattr("dialogue_tts_core.cost_router.FREE_POOL", test_free_pool)


# --- Tests for _calculate_effective_cost_per_mchar ---
def test_calculate_cost_direct_mchar() -> None:
    engine = EngineMeta(
        "p",
        "m1",
        price_per_mchar_output_audio_usd=20.0,
        latency_ms=100,
        quality_tier=QualityTier.MID,
    )
    assert _calculate_effective_cost_per_mchar(engine, 1000) == 20.0


def test_calculate_cost_direct_mchar_with_input_cost() -> None:
    engine = EngineMeta(
        "p",
        "m2",
        price_per_mchar_input_usd=5.0,
        price_per_mchar_output_audio_usd=20.0,
        latency_ms=100,
        quality_tier=QualityTier.MID,
    )
    assert _calculate_effective_cost_per_mchar(engine, 1000) == 25.0


def test_calculate_cost_per_minute(_monkeypatch: MonkeyPatch) -> None:
    engine = EngineMeta(
        "p",
        "m3",
        price_per_audio_minute_usd=0.10,
        latency_ms=100,
        quality_tier=QualityTier.MID,
    )
    expected_cost = (0.10 * CHARS_IN_ONE_M) / DEFAULT_CHARS_PER_AUDIO_MINUTE
    assert _calculate_effective_cost_per_mchar(engine, 1000) == pytest.approx(
        expected_cost
    )


def test_calculate_cost_per_minute_with_input_cost(_monkeypatch: MonkeyPatch) -> None:
    engine = EngineMeta(
        "p",
        "m4",
        price_per_mchar_input_usd=2.0,
        price_per_audio_minute_usd=0.10,
        latency_ms=100,
        quality_tier=QualityTier.MID,
    )
    expected_cost_audio_part = (0.10 * CHARS_IN_ONE_M) / DEFAULT_CHARS_PER_AUDIO_MINUTE
    assert _calculate_effective_cost_per_mchar(engine, 1000) == pytest.approx(
        expected_cost_audio_part + 2.0
    )


def test_calculate_cost_unpriceable() -> None:
    engine = EngineMeta(
        "p", "m5", latency_ms=100, quality_tier=QualityTier.MID
    )  # No pricing info
    assert _calculate_effective_cost_per_mchar(engine, 1000) == float("inf")


def test_calculate_cost_invalid_chars_per_minute(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dialogue_tts_core.cost_router.DEFAULT_CHARS_PER_AUDIO_MINUTE", 0
    )
    engine = EngineMeta(
        "p",
        "m_invalid",
        price_per_audio_minute_usd=0.10,
        latency_ms=100,
        quality_tier=QualityTier.MID,
    )
    with pytest.raises(
        ValueError, match="DEFAULT_CHARS_PER_AUDIO_MINUTE must be positive"
    ):
        _calculate_effective_cost_per_mchar(engine, 1000)


# --- Tests for select_engine ---
def test_select_engine_basic_selection(
    _mock_engine_table: dict[str, EngineMeta],
) -> None:
    selected = select_engine(char_len=1000)
    assert selected.model_id == "cheapest_mid"


def test_select_engine_quality_filter(mock_engine_table: dict[str, EngineMeta]) -> None:
    selected_high = select_engine(char_len=1000, desired_quality=QualityTier.HIGH)
    assert selected_high.model_id == "expensive_high_slow"

    selected_ultra = select_engine(char_len=1000, desired_quality=QualityTier.ULTRA)
    assert selected_ultra.model_id == "ultra_expensive"

    # Test for no engine meeting quality
    # Create a copy of the engine table and remove/downgrade ULTRA tier engines
    temp_engine_table = {
        k: v
        for k, v in mock_engine_table.items()
        if v.quality_tier != QualityTier.ULTRA
    }
    # If an ULTRA engine was the only one, make it HIGH
    if (
        "test_ultra_expensive" in temp_engine_table
    ):  # Should not be if logic above is correct
        pass  # it's already removed
    elif "test_ultra_expensive" in mock_engine_table:  # if it existed in original
        engine_to_downgrade = mock_engine_table["test_ultra_expensive"]
        temp_engine_table["test_ultra_expensive_downgraded"] = EngineMeta(
            provider=engine_to_downgrade.provider,
            model_id=engine_to_downgrade.model_id,
            price_per_mchar_output_audio_usd=engine_to_downgrade.price_per_mchar_output_audio_usd,
            latency_ms=engine_to_downgrade.latency_ms,
            quality_tier=QualityTier.HIGH,
            supports_emotion=engine_to_downgrade.supports_emotion,
            supports_voice_cloning=engine_to_downgrade.supports_voice_cloning,
            price_per_mchar_input_usd=engine_to_downgrade.price_per_mchar_input_usd,
            price_per_audio_minute_usd=engine_to_downgrade.price_per_audio_minute_usd,
        )

    with (
        patch("dialogue_tts_core.cost_router.ENGINE_TABLE", temp_engine_table),
        pytest.raises(
            RuntimeError, match="No engines found matching or exceeding desired quality"
        ),
    ):
        select_engine(char_len=1000, desired_quality=QualityTier.ULTRA)


def test_select_engine_low_latency_preference(
    mock_engine_table: dict[str, EngineMeta], caplog: LogCaptureFixture
) -> None:
    # Using LATENCY_THRESHOLD_MS from cost_router (default 350ms)
    selected = select_engine(
        char_len=500, desired_quality=QualityTier.MID, prefer_low_latency=True
    )
    # Candidates within MID+ and <= LATENCY_THRESHOLD_MS (350ms):
    # test_cheapest_mid (300ms, $10)
    # test_cheap_mid_low_latency (100ms, $12)
    # test_dual_meter_mid (250ms, $11)
    # cartesia_mock (90ms, $39)
    # Sorted by cost, then latency:
    # cheapest_mid, dual_mid, cheap_mid_low_latency, cartesia_mock
    assert selected.model_id == "cheapest_mid"

    # Test fallback if no low-latency options meet quality
    # Modify table so all MID engines are high latency
    high_latency_mid_engines = {
        k: (
            EngineMeta(
                provider=v.provider,
                model_id=v.model_id,
                price_per_mchar_output_audio_usd=v.price_per_mchar_output_audio_usd,
                latency_ms=LATENCY_THRESHOLD_MS + 100,
                quality_tier=v.quality_tier,
                supports_emotion=v.supports_emotion,
                supports_voice_cloning=v.supports_voice_cloning,
                price_per_mchar_input_usd=v.price_per_mchar_input_usd,
                price_per_audio_minute_usd=v.price_per_audio_minute_usd,
            )
            if v.quality_tier == QualityTier.MID
            else v
        )
        for k, v in mock_engine_table.items()
    }
    # Ensure at least one MID engine exists, even if high latency
    if not any(
        e.quality_tier == QualityTier.MID for e in high_latency_mid_engines.values()
    ):
        high_latency_mid_engines["fallback_mid_high_latency"] = EngineMeta(
            "p",
            "fallback_mid",
            price_per_mchar_output_audio_usd=10.0,
            latency_ms=LATENCY_THRESHOLD_MS + 100,
            quality_tier=QualityTier.MID,
        )

    with patch("dialogue_tts_core.cost_router.ENGINE_TABLE", high_latency_mid_engines):
        caplog.clear()
        selected_fallback = select_engine(
            char_len=500, desired_quality=QualityTier.MID, prefer_low_latency=True
        )
        # Should still select a MID engine, but log a warning
        assert selected_fallback.quality_tier == QualityTier.MID
        assert any(
            "No low-latency engines found" in record.message
            for record in caplog.records
        )


def test_select_engine_emotion_support_preference(
    mock_engine_table: dict[str, EngineMeta], caplog: LogCaptureFixture
) -> None:
    selected = select_engine(
        char_len=500, desired_quality=QualityTier.MID, prefer_emotion_support=True
    )
    # Emotion supporting MID: cheap_mid_low_latency ($12, 100ms), dual_mid ($11, 250ms),
    # cartesia_mock ($39, 90ms)
    # Sorted by cost, then latency: dual_mid, cheap_mid_low_latency, cartesia_mock
    assert selected.model_id == "dual_mid"

    # Test fallback
    no_emotion_mid_engines = {
        k: (
            EngineMeta(
                provider=v.provider,
                model_id=v.model_id,
                price_per_mchar_output_audio_usd=v.price_per_mchar_output_audio_usd,
                latency_ms=v.latency_ms,
                quality_tier=v.quality_tier,
                supports_emotion=False,
                supports_voice_cloning=v.supports_voice_cloning,
                price_per_mchar_input_usd=v.price_per_mchar_input_usd,
                price_per_audio_minute_usd=v.price_per_audio_minute_usd,
            )
            if v.quality_tier == QualityTier.MID
            else v
        )
        for k, v in mock_engine_table.items()
    }
    # Ensure at least one MID engine exists
    if not any(
        e.quality_tier == QualityTier.MID for e in no_emotion_mid_engines.values()
    ):
        no_emotion_mid_engines["fallback_mid_no_emotion"] = EngineMeta(
            "p",
            "fallback_mid_ne",
            price_per_mchar_output_audio_usd=10.0,
            latency_ms=300,
            quality_tier=QualityTier.MID,
            supports_emotion=False,
        )

    with patch("dialogue_tts_core.cost_router.ENGINE_TABLE", no_emotion_mid_engines):
        caplog.clear()
        selected_fallback = select_engine(
            char_len=500, desired_quality=QualityTier.MID, prefer_emotion_support=True
        )
        assert (
            selected_fallback.quality_tier == QualityTier.MID
        )  # Should still pick a MID engine
        assert (
            not selected_fallback.supports_emotion
        )  # Confirms it picked one without emotion
        assert any(
            "No emotion-supporting engines found" in record.message
            for record in caplog.records
        )


def test_select_engine_budget_constraint(
    _mock_engine_table: dict[str, EngineMeta],
) -> None:
    selected = select_engine(
        char_len=10000, desired_quality=QualityTier.MID, max_cost_usd_for_job=0.115
    )
    # cheapest_mid: 10/M * 10k = $0.10 (OK)
    # dual_mid: 11/M * 10k = $0.11 (OK)
    # Expected: cheapest_mid
    assert selected.model_id == "cheapest_mid"

    with pytest.raises(RuntimeError, match="No engines meet the budget constraint"):
        select_engine(
            char_len=10000, desired_quality=QualityTier.MID, max_cost_usd_for_job=0.05
        )


@patch("dialogue_tts_core.cost_router._get_remaining_free_quota")
def test_select_engine_free_tier_logic(
    mock_get_quota: MagicMock,
    _mock_engine_table: dict[str, EngineMeta],
    _mock_free_tiers_config: None,
) -> None:
    # Scenario 1: User has enough free quota for cartesia_mock
    mock_get_quota.return_value = 25000
    selected = select_engine(
        char_len=1000, user_id="user123", desired_quality=QualityTier.MID
    )
    assert selected.model_id == "sonic-test"
    mock_get_quota.assert_called_with("user123", "cartesia_sonic-test")

    # Scenario 2: User has insufficient free quota for cartesia_mock
    mock_get_quota.return_value = 50
    selected = select_engine(
        char_len=1000, user_id="user123", desired_quality=QualityTier.MID
    )
    assert selected.model_id == "cheapest_mid"

    # Scenario 3: No user_id, global free tier for cartesia_mock considered
    mock_get_quota.reset_mock()
    # Simulate _get_remaining_free_quota returning the default from FREE_TIERS
    # when user_id is None. This requires _get_remaining_free_quota to handle
    # user_id=None appropriately or a different mock strategy. For simplicity,
    # let's assume _get_remaining_free_quota is called and returns the global
    # quota. The actual _get_remaining_free_quota needs to be designed to fetch
    # from DEFAULT_FREE_TIERS if user_id is None. Let's assume it does for this
    # test.

    # If we patch FREE_TIERS directly for the no-user_id case:
    with patch.dict(
        "dialogue_tts_core.cost_router.FREE_TIERS",
        {"cartesia_sonic-test": 25000},
    ):
        # Ensure _get_remaining_free_quota will be called with
        # (None, "cartesia_sonic-test") and we need it to return this patched value.
        def mock_get_quota_for_none(uid: str | None, engine_id_str: str) -> int:
            if uid is None and engine_id_str == "cartesia_sonic-test":
                return 25000  # Sufficient global free quota
            return 0  # Default for others or if uid is present

        mock_get_quota.side_effect = mock_get_quota_for_none
        selected_no_user_sufficient = select_engine(
            char_len=1000, user_id=None, desired_quality=QualityTier.MID
        )
        assert selected_no_user_sufficient.model_id == "sonic-test"

    with patch.dict(
        "dialogue_tts_core.cost_router.FREE_TIERS", {"cartesia_sonic-test": 50}
    ):

        def mock_get_quota_for_none_insufficient(
            uid: str | None, engine_id_str: str
        ) -> int:
            if uid is None and engine_id_str == "cartesia_sonic-test":
                return 50  # Insufficient global free quota
            return 0

        mock_get_quota.side_effect = mock_get_quota_for_none_insufficient
        selected_no_user_insufficient = select_engine(
            char_len=1000, user_id=None, desired_quality=QualityTier.MID
        )
        assert selected_no_user_insufficient.model_id == "cheapest_mid"


def test_select_engine_char_len_zero(_mock_engine_table: dict[str, EngineMeta]) -> None:
    selected = select_engine(char_len=0)
    # Costs are 0. Tie-break: latency, then quality (higher is better).
    # cartesia_mock (90ms, MID)
    # cheap_mid_low_latency (100ms, MID)
    # expensive_high_fast (150ms, HIGH)
    # ultra_expensive (200ms, ULTRA)
    # dual_mid (250ms, MID)
    # cheapest_mid (300ms, MID)
    # minute_mid (400ms, MID)
    # expensive_high_slow (500ms, HIGH)
    # low_q (600ms, LOW)
    # Expected: cartesia_mock (90ms, MID) is fastest among MID+
    assert selected.model_id == "sonic-test"


def test_select_engine_empty_table(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("dialogue_tts_core.cost_router.ENGINE_TABLE", {})
    with pytest.raises(
        RuntimeError, match="No engines found matching or exceeding desired quality"
    ):  # Default is MID
        select_engine(char_len=100)


def test_select_engine_tie_breaking(monkeypatch: MonkeyPatch) -> None:
    # Same cost and latency, different quality
    engines_qual = {
        "e1_qual": EngineMeta(
            "p",
            "m1_qual",
            price_per_mchar_output_audio_usd=10.0,
            latency_ms=100,
            quality_tier=QualityTier.MID,
        ),
        "e2_qual": EngineMeta(
            "p",
            "m2_qual",
            price_per_mchar_output_audio_usd=10.0,
            latency_ms=100,
            quality_tier=QualityTier.HIGH,
        ),
        "e3_qual": EngineMeta(
            "p",
            "m3_qual",
            price_per_mchar_output_audio_usd=10.0,
            latency_ms=100,
            quality_tier=QualityTier.LOW,
        ),
    }
    monkeypatch.setattr("dialogue_tts_core.cost_router.ENGINE_TABLE", engines_qual)
    selected = select_engine(char_len=100, desired_quality=QualityTier.LOW)
    assert selected.model_id == "m2_qual"  # Higher quality preferred

    # Same cost, different latency, same quality
    engines_lat = {
        "e1_lat": EngineMeta(
            "p",
            "m1_lat",
            price_per_mchar_output_audio_usd=10.0,
            latency_ms=200,
            quality_tier=QualityTier.MID,
        ),
        "e2_lat": EngineMeta(
            "p",
            "m2_lat",
            price_per_mchar_output_audio_usd=10.0,
            latency_ms=100,
            quality_tier=QualityTier.MID,
        ),
    }
    monkeypatch.setattr("dialogue_tts_core.cost_router.ENGINE_TABLE", engines_lat)
    selected_lat = select_engine(char_len=100, desired_quality=QualityTier.MID)
    assert selected_lat.model_id == "m2_lat"  # Lower latency preferred
