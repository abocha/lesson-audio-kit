import pytest

from dialogue_tts_core.dialogue_script_parser import (
    CHARS_PER_SECOND_ESTIMATE,
    GPT_4O_MINI_TTS_COST_PER_SECOND,
    MAX_SCRIPT_LENGTH,
    TTS_1_HD_COST_PER_CHAR,
    calculate_cost,
    parse_dialogue_script,
)


def test_parse_dialogue_script_valid_script() -> None:
    """Tests parsing a basic valid script with speakers and narration."""
    script = "[Alice] Hello.\n[Bob] Hi there.\nNarrator: And so it was."
    parsed_lines, total_chars = parse_dialogue_script(script)

    assert len(parsed_lines) == 3
    assert parsed_lines[0] == {"id": 0, "speaker": "Alice", "text": "Hello."}
    assert parsed_lines[1] == {"id": 1, "speaker": "Bob", "text": "Hi there."}
    # Assuming narration lines are kept as is with 'Narrator' speaker
    assert parsed_lines[2] == {
        "id": 2,
        "speaker": "Narrator",
        "text": "Narrator: And so it was.",
    }

    # Calculate expected total_chars based on the text content
    expected_chars = len("Hello.") + len("Hi there.") + len("Narrator: And so it was.")
    assert total_chars == expected_chars


def test_parse_dialogue_script_empty_speaker_tag() -> None:
    """Tests parsing a line with an empty speaker tag."""
    script = "[] An utterance."
    parsed_lines, _ = parse_dialogue_script(script)

    assert len(parsed_lines) == 1
    assert parsed_lines[0]["speaker"] == "UnknownSpeaker"
    assert parsed_lines[0]["text"] == "An utterance."


def test_parse_dialogue_script_only_narration() -> None:
    """Tests parsing a script with only narration lines."""
    script = "This is a story.\nAbout a test."
    parsed_lines, _ = parse_dialogue_script(script)

    assert len(parsed_lines) == 2
    assert parsed_lines[0]["speaker"] == "Narrator"
    assert parsed_lines[0]["text"] == "This is a story."
    assert parsed_lines[1]["speaker"] == "Narrator"
    assert parsed_lines[1]["text"] == "About a test."


def test_parse_dialogue_script_empty_input() -> None:
    """Tests parsing an empty script string."""
    script = ""
    parsed_lines, total_chars = parse_dialogue_script(script)

    assert parsed_lines == []
    assert total_chars == 0


def test_parse_dialogue_script_whitespace_input() -> None:
    """Tests parsing a script with only whitespace and newlines."""
    script = "   \n \n   "
    parsed_lines, total_chars = parse_dialogue_script(script)

    assert parsed_lines == []
    assert total_chars == 0


def test_parse_dialogue_script_speaker_tag_only_no_utterance() -> None:
    """Tests parsing lines with only speaker tags and no utterance."""
    script = "[Alice]\n[Bob] \n[Charlie] Message"
    parsed_lines, _ = parse_dialogue_script(script)

    assert len(parsed_lines) == 1
    assert parsed_lines[0]["speaker"] == "Charlie"
    assert parsed_lines[0]["text"] == "Message"


def test_parse_dialogue_script_character_count() -> None:
    """Tests the total character count calculation."""
    script = "[A] 123\n[B] 4567"
    _, total_chars = parse_dialogue_script(script)

    assert total_chars == len("123") + len("4567")


def test_parse_dialogue_script_max_length_exceeded() -> None:
    """Tests that a ValueError is raised when the script exceeds MAX_SCRIPT_LENGTH."""
    script = "a" * (MAX_SCRIPT_LENGTH + 1)
    with pytest.raises(ValueError, match="Script is too long"):
        parse_dialogue_script(script)


def test_parse_dialogue_script_at_max_length() -> None:
    """Tests that a script exactly at MAX_SCRIPT_LENGTH is parsed without error."""
    script = "a" * MAX_SCRIPT_LENGTH
    # Should not raise ValueError
    parsed_lines, total_chars = parse_dialogue_script(script)
    assert len(parsed_lines) == 1  # Will be one "Narrator" line
    assert total_chars == MAX_SCRIPT_LENGTH


def test_calculate_cost_tts_1_hd() -> None:
    """Tests cost calculation for 'tts-1-hd' model."""
    chars = 1000
    cost = calculate_cost(chars, _num_lines=5, model_name="tts-1-hd")
    assert cost == pytest.approx(chars * TTS_1_HD_COST_PER_CHAR)


def test_calculate_cost_tts_1() -> None:
    """Tests cost calculation for 'tts-1' model (should use same rate as tts-1-hd)."""
    chars = 1000
    cost = calculate_cost(chars, _num_lines=5, model_name="tts-1")
    assert cost == pytest.approx(chars * TTS_1_HD_COST_PER_CHAR)  # Same rate


def test_calculate_cost_gpt_4o_mini_tts() -> None:
    """Tests cost calculation for 'gpt-4o-mini-tts' model."""
    chars = 1000
    expected_seconds = chars / CHARS_PER_SECOND_ESTIMATE
    expected_cost = expected_seconds * GPT_4O_MINI_TTS_COST_PER_SECOND
    cost = calculate_cost(chars, _num_lines=5, model_name="gpt-4o-mini-tts")
    assert cost == pytest.approx(expected_cost)


def test_calculate_cost_unknown_model() -> None:
    """Tests cost calculation for an unknown model (fallbacks to tts-1-hd rate)."""
    chars = 1000
    cost = calculate_cost(chars, _num_lines=5, model_name="some-random-model-v7")
    assert cost == pytest.approx(chars * TTS_1_HD_COST_PER_CHAR)  # Fallback rate


def test_calculate_cost_zero_chars() -> None:
    """Tests cost calculation with zero characters."""
    cost = calculate_cost(0, _num_lines=0, model_name="tts-1-hd")
    assert cost == 0.0
