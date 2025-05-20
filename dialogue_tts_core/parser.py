import re

# Module-level constants
MAX_SCRIPT_LENGTH = 10000
TTS_1_HD_COST_PER_CHAR = 0.00003
GPT_4O_MINI_TTS_COST_PER_SECOND = 0.015 / 60
CHARS_PER_SECOND_ESTIMATE = 12


def parse_dialogue_script(script_text: str) -> tuple[list[dict], int]:
    """
    Parses a raw dialogue script string into a list of structured lines
    and calculates the total character count for utterances.

    Args:
        script_text: The raw multiline string containing the dialogue script.
                     Lines are separated by newlines. Expected format for
                     lines with explicit speakers is `[Speaker Name] Utterance text`.

    Returns:
        A tuple: (parsed_lines, total_chars)
        - parsed_lines (list[dict]): A list where each dictionary represents a
                                     parsed line of dialogue with keys "id",
                                     "speaker", and "text".
        - total_chars (int): The sum of the lengths of all utterance texts.

    Raises:
        ValueError: If the script_text exceeds MAX_SCRIPT_LENGTH.
    """
    parsed_lines_list = []
    total_char_count = 0

    if len(script_text) > MAX_SCRIPT_LENGTH:
        raise ValueError(
            f"Script is too long. Maximum {MAX_SCRIPT_LENGTH} characters allowed. "
            f"Your script has {len(script_text)} characters."
        )

    source_lines = script_text.strip().split("\n")

    for i, line_content in enumerate(source_lines):
        line_content_stripped = line_content.strip()

        if not line_content_stripped:
            continue  # Skip empty lines

        # Regex: r'\[(.*?)\]\s*(.*)'
        match_object = re.match(r"\[(.*?)\]\s*(.*)", line_content_stripped)

        if match_object:
            speaker_candidate = match_object.group(1).strip()
            utterance_candidate = match_object.group(2).strip()

            speaker_name = speaker_candidate if speaker_candidate else "UnknownSpeaker"
            utterance_text = utterance_candidate
        else:  # No speaker tag match
            speaker_name = "Narrator"
            utterance_text = line_content_stripped

        if not utterance_text:  # E.g., "[SpeakerName]" or "[]" with no following text
            continue  # Skip lines that result in no utterance

        parsed_lines_list.append(
            {"id": i, "speaker": speaker_name, "text": utterance_text}
        )
        total_char_count += len(utterance_text)

    return (parsed_lines_list, total_char_count)


def calculate_cost(
    total_chars: int, _num_lines: int, model_name: str = "tts-1-hd"
) -> float:
    """
    Estimates the monetary cost of TTS synthesis based on character count and model.

    Args:
        total_chars: The total number of characters to be synthesized.
        num_lines: The total number of lines to be synthesized
            (currently unused in logic).
        model_name: The identifier for the TTS model. Defaults to "tts-1-hd".

    Returns:
        The estimated monetary cost in USD.
    """
    cost_result = 0.0

    if model_name in ["tts-1", "tts-1-hd"]:
        cost_result = total_chars * TTS_1_HD_COST_PER_CHAR
    elif model_name == "gpt-4o-mini-tts":
        # Note: This logic is based on per-second pricing for gpt-4o-mini-tts
        # as per the original script's interpretation.
        if CHARS_PER_SECOND_ESTIMATE <= 0:
            estimated_duration_seconds = total_chars / 10.0  # Fallback division factor
        else:
            estimated_duration_seconds = total_chars / CHARS_PER_SECOND_ESTIMATE
        cost_result = estimated_duration_seconds * GPT_4O_MINI_TTS_COST_PER_SECOND
    else:
        # Fallback for unknown models: use tts-1-hd rate (as per original script)
        cost_result = total_chars * TTS_1_HD_COST_PER_CHAR

    return cost_result


if __name__ == "__main__":
    # Test cases (copied from original script_parser.py)
    print("Running parser.py test cases...")

    # Test case 1: Basic script
    script1 = (
        "[Alice] Hello Bob.\n"
        "[Bob] Hi Alice!\n"
        "This is narration.\n"
        "[] An unknown speaker."
    )
    print(f"\nParsing script 1:\n---\n{script1}\n---")
    try:
        parsed_script1, chars1 = parse_dialogue_script(script1)
        print("Parsed lines:")
        for line in parsed_script1:
            print(line)
        print(f"Total characters: {chars1}")
        # Expected chars1: 10 + 9 + 18 + 19 = 56
        assert chars1 == 56
        print("Assertion for total characters passed.")

        # Test cost calculation for script 1
        cost_tts_hd_1 = calculate_cost(
            chars1, len(parsed_script1), model_name="tts-1-hd"
        )
        print(f"Estimated cost (tts-1-hd): ${cost_tts_hd_1:.6f}")
        assert abs(cost_tts_hd_1 - (56 * TTS_1_HD_COST_PER_CHAR)) < 1e-9
        print("Assertion for tts-1-hd cost passed.")

        cost_gpt_mini_1 = calculate_cost(
            chars1, len(parsed_script1), model_name="gpt-4o-mini-tts"
        )
        print(f"Estimated cost (gpt-4o-mini-tts): ${cost_gpt_mini_1:.6f}")
        expected_duration_1 = (
            chars1 / CHARS_PER_SECOND_ESTIMATE
            if CHARS_PER_SECOND_ESTIMATE > 0
            else chars1 / 10.0
        )
        expected_cost_gpt_mini_1 = expected_duration_1 * GPT_4O_MINI_TTS_COST_PER_SECOND
        assert abs(cost_gpt_mini_1 - expected_cost_gpt_mini_1) < 1e-9
        print("Assertion for gpt-4o-mini-tts cost passed.")

    except ValueError as e:
        print(f"Error parsing script 1: {e}")

    # Test case 2: Empty script
    script2 = ""
    print(f"\nParsing script 2:\n---\n{script2}\n---")
    try:
        parsed_script2, chars2 = parse_dialogue_script(script2)
        print("Parsed lines:", parsed_script2)
        print(f"Total characters: {chars2}")
        assert parsed_script2 == []
        assert chars2 == 0
        print("Assertion for empty script passed.")
        cost_empty = calculate_cost(chars2, len(parsed_script2))
        print(f"Estimated cost (empty script): ${cost_empty:.6f}")
        assert cost_empty == 0.0
        print("Assertion for empty script cost passed.")

    except ValueError as e:
        print(f"Error parsing script 2: {e}")

    # Test case 3: Script with only whitespace and blank lines
    script3 = "   \n\n [Speaker]  \n  Utterance \n\n"
    print(f"\nParsing script 3:\n---\n{script3}\n---")
    try:
        parsed_script3, chars3 = parse_dialogue_script(script3)
        print("Parsed lines:")
        for line in parsed_script3:
            print(line)
        print(f"Total characters: {chars3}")
        # Expected: only "Utterance" line should be parsed
        assert len(parsed_script3) == 1
        assert parsed_script3[0]["text"] == "Utterance"
        assert chars3 == len("Utterance")
        print("Assertion for whitespace/blank lines passed.")
        cost_ws = calculate_cost(chars3, len(parsed_script3))
        print(f"Estimated cost (whitespace script): ${cost_ws:.6f}")
        assert abs(cost_ws - (chars3 * TTS_1_HD_COST_PER_CHAR)) < 1e-9
        print("Assertion for whitespace script cost passed.")

    except ValueError as e:
        print(f"Error parsing script 3: {e}")

    # Test case 4: Script with empty speaker tag
    script4 = "[] This is an unknown speaker."
    print(f"\nParsing script 4:\n---\n{script4}\n---")
    try:
        parsed_script4, chars4 = parse_dialogue_script(script4)
        print("Parsed lines:")
        for line in parsed_script4:
            print(line)
        print(f"Total characters: {chars4}")
        assert len(parsed_script4) == 1
        assert parsed_script4[0]["speaker"] == "UnknownSpeaker"
        assert parsed_script4[0]["text"] == "This is an unknown speaker."
        assert chars4 == len("This is an unknown speaker.")
        print("Assertion for empty speaker tag passed.")

    except ValueError as e:
        print(f"Error parsing script 4: {e}")

    # Test case 5: Script line with only speaker tag
    script5 = "[SpeakerName]"
    print(f"\nParsing script 5:\n---\n{script5}\n---")
    try:
        parsed_script5, chars5 = parse_dialogue_script(script5)
        print("Parsed lines:", parsed_script5)
        print(f"Total characters: {chars5}")
        assert parsed_script5 == []  # Should be skipped
        assert chars5 == 0
        print("Assertion for line with only speaker tag passed.")

    except ValueError as e:
        print(f"Error parsing script 5: {e}")

    # Test case 6: Script exceeding max length
    script6 = "a" * (MAX_SCRIPT_LENGTH + 1)
    print(f"\nParsing script 6 (too long): length {len(script6)}")
    try:
        parse_dialogue_script(script6)
        print("Error: ValueError was not raised for script exceeding max length.")
    except ValueError as e:
        print(f"Successfully caught expected error: {e}")
        assert f"Maximum {MAX_SCRIPT_LENGTH} characters allowed" in str(e)
        print("Assertion for max length error message passed.")

    print("\nparser.py test cases finished.")
