from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import AsyncOpenAI, OpenAIError, RateLimitError

# from openai._response import AsyncAPIResponse  # No longer needed, Stream is used
import pytest

from dialogue_tts_core.tts_client import (
    INITIAL_BACKOFF_SECONDS,
    MAX_RETRIES,  # Import constants for retry tests
    is_content_safe,
    synthesize_speech_line,
)

# The global semaphore is not directly tested here, so the fixture is not
# needed for these tests.
# If tests involving concurrency with the semaphore are added, a proper
# fixture to manage its state would be required.
# @pytest.fixture(autouse=True)
# def reset_semaphore():
#     pass


# --- Tests for is_content_safe ---


@pytest.mark.asyncio
async def test_is_content_safe_no_url() -> None:
    """Test is_content_safe returns True when no API URL template is provided."""
    result = await is_content_safe("some text", None)
    assert result is True


@pytest.mark.asyncio
async def test_is_content_safe_invalid_url_template(mocker: MagicMock) -> None:
    """Test is_content_safe returns True and warns for invalid URL template."""
    mock_print = mocker.patch("builtins.print")
    result = await is_content_safe(
        "some text",
        "http://example.com/check",  # Missing {text}
    )
    assert result is True
    mock_print.assert_called_once()
    assert "Warning: NSFW_API_URL_TEMPLATE" in mock_print.call_args[0][0]


@pytest.mark.asyncio
async def test_is_content_safe_api_success(mocker: MagicMock) -> None:
    """Test is_content_safe returns True when API returns 200."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()  # Does nothing for 200

    mock_async_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client_instance.get.return_value = mock_response

    # Mock the AsyncClient context manager
    mock_async_client_context_manager = AsyncMock()
    mock_async_client_context_manager.__aenter__.return_value = (
        mock_async_client_instance
    )
    mock_async_client_context_manager.__aexit__.return_value = (
        None  # Ensure __aexit__ is mocked
    )

    mocker.patch("httpx.AsyncClient", return_value=mock_async_client_context_manager)

    result = await is_content_safe("safe text", "http://example.com/check?text={text}")
    assert result is True
    mock_async_client_instance.get.assert_called_once()
    # Can also check the URL called if needed by inspecting call_args


@pytest.mark.asyncio
async def test_is_content_safe_api_unsafe(mocker: MagicMock) -> None:
    """Test is_content_safe returns False when API returns non-200."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 400
    mock_response.text = "Content is unsafe"
    mock_response.request = MagicMock(spec=httpx.Request)
    mock_response.request.url = "http://example.com/check?text=unsafe"

    # Configure raise_for_status to raise HTTPStatusError
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Bad Request", request=mock_response.request, response=mock_response
    )

    mock_async_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client_instance.get.return_value = mock_response

    mock_async_client_context_manager = AsyncMock()
    mock_async_client_context_manager.__aenter__.return_value = (
        mock_async_client_instance
    )
    mock_async_client_context_manager.__aexit__.return_value = None

    mocker.patch("httpx.AsyncClient", return_value=mock_async_client_context_manager)
    mock_print = mocker.patch("builtins.print")

    result = await is_content_safe(
        "unsafe text", "http://example.com/check?text={text}"
    )
    assert result is False
    mock_async_client_instance.get.assert_called_once()
    mock_print.assert_called_once()
    assert "NSFW Check: API request failed." in mock_print.call_args[0][0]


@pytest.mark.asyncio
async def test_is_content_safe_api_request_error(mocker: MagicMock) -> None:
    """Test is_content_safe returns True (fail open) on request errors."""
    mock_async_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client_instance.get.side_effect = httpx.RequestError(
        "Network error",
        request=MagicMock(
            spec=httpx.Request, url="http://example.com/check?text=error"
        ),
    )

    mock_async_client_context_manager = AsyncMock()
    mock_async_client_context_manager.__aenter__.return_value = (
        mock_async_client_instance
    )
    mock_async_client_context_manager.__aexit__.return_value = None

    mocker.patch("httpx.AsyncClient", return_value=mock_async_client_context_manager)
    mock_print = mocker.patch("builtins.print")

    result = await is_content_safe("text", "http://example.com/check?text={text}")
    assert result is True  # Fail open
    mock_async_client_instance.get.assert_called_once()
    mock_print.assert_called_once()
    assert "NSFW Check: API request error:" in mock_print.call_args[0][0]


@pytest.mark.asyncio
async def test_is_content_safe_api_unexpected_error(mocker: MagicMock) -> None:
    """Test is_content_safe returns True (fail open) on unexpected errors."""
    mock_async_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client_instance.get.side_effect = Exception("Unexpected issue")

    mock_async_client_context_manager = AsyncMock()
    mock_async_client_context_manager.__aenter__.return_value = (
        mock_async_client_instance
    )
    mock_async_client_context_manager.__aexit__.return_value = None

    mocker.patch("httpx.AsyncClient", return_value=mock_async_client_context_manager)
    mock_print = mocker.patch("builtins.print")

    result = await is_content_safe("text", "http://example.com/check?text={text}")
    assert result is True  # Fail open
    mock_async_client_instance.get.assert_called_once()
    mock_print.assert_called_once()
    assert "NSFW Check: An unexpected error occurred:" in mock_print.call_args[0][0]


# --- Tests for synthesize_speech_line ---


@pytest.mark.asyncio
async def test_synthesize_speech_line_empty_text(mocker: MagicMock) -> None:
    """Test synthesize_speech_line returns None for empty input text."""
    mock_openai_client = AsyncMock(spec=AsyncOpenAI)
    mock_print = mocker.patch("builtins.print")

    result = await synthesize_speech_line(
        client=mock_openai_client,
        text="",
        voice="alloy",
        output_path="output.mp3",
        line_index=1,
        cache_base_dir="test_cache",
    )
    assert result[0] is None
    assert result[1] is False  # Not from cache
    mock_openai_client.audio.speech.create.assert_not_called()
    mock_print.assert_called_once()
    assert "Input text is empty. Skipping synthesis." in mock_print.call_args[0][0]


@pytest.mark.asyncio
async def test_synthesize_speech_line_unsafe_content(mocker: MagicMock) -> None:
    """Test synthesize_speech_line returns None if content is flagged unsafe."""
    mock_openai_client = AsyncMock(spec=AsyncOpenAI)
    mocker.patch(
        "dialogue_tts_core.tts_client.is_content_safe", AsyncMock(return_value=False)
    )
    mock_print = mocker.patch("builtins.print")

    result = await synthesize_speech_line(
        client=mock_openai_client,
        text="unsafe text",
        voice="alloy",
        output_path="output.mp3",
        nsfw_api_url_template="http://example.com/check?text={text}",
        line_index=2,
        cache_base_dir="test_cache",
    )
    assert result[0] is None
    assert result[1] is False  # Not from cache
    mock_openai_client.audio.speech.create.assert_not_called()
    mock_print.assert_called_once()
    assert (
        "Content flagged as potentially unsafe. Skipping synthesis."
        in mock_print.call_args[0][0]
    )


@pytest.mark.asyncio
async def test_synthesize_speech_line_successful(mocker: MagicMock) -> None:
    """Test successful speech synthesis."""
    mock_openai_client = AsyncMock(spec=AsyncOpenAI)
    # Mock the response object returned by create
    # This is the object returned by client.audio.speech.create()
    mock_stream_response = AsyncMock()

    async def _mock_astream_to_file_successful(*_args: Any, **_kwargs: Any) -> None:
        return None

    mock_stream_response.astream_to_file = AsyncMock(
        side_effect=_mock_astream_to_file_successful
    )

    mock_openai_client.audio.speech.create.return_value = mock_stream_response

    mocker.patch(
        "dialogue_tts_core.tts_client.is_content_safe", AsyncMock(return_value=True)
    )
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.path.getsize", return_value=1024)

    output_p = "test_successful.mp3"
    result = await synthesize_speech_line(
        client=mock_openai_client,
        text="hello",
        voice="alloy",
        output_path=output_p,
        line_index=3,
        cache_base_dir="test_cache",
    )  # Make sure line_index is passed if your function uses it for prints
    assert result[0] == output_p
    assert result[1] is False  # Not from cache
    mock_openai_client.audio.speech.create.assert_called_once_with(
        model="tts-1-hd",  # Default model
        input="hello",
        voice="alloy",
        response_format="mp3",
    )
    mock_stream_response.astream_to_file.assert_awaited_once_with(output_p)


@pytest.mark.asyncio
async def test_synthesize_speech_line_openai_error(mocker: MagicMock) -> None:
    """Test synthesize_speech_line returns None on OpenAI API error
    (not RateLimitError)."""
    mock_openai_client = AsyncMock(spec=AsyncOpenAI)
    # OpenAIError constructor does not accept response or body in this
    # version
    mock_openai_client.audio.speech.create.side_effect = OpenAIError(
        "Some OpenAI error",
    )

    mocker.patch(
        "dialogue_tts_core.tts_client.is_content_safe", AsyncMock(return_value=True)
    )
    mock_print = mocker.patch("builtins.print")

    result = await synthesize_speech_line(
        client=mock_openai_client,
        text="text",
        voice="alloy",
        output_path="output.mp3",
        line_index=4,
        cache_base_dir="test_cache",
    )
    assert result[0] is None
    assert result[1] is False  # Not from cache
    mock_openai_client.audio.speech.create.assert_called_once()
    mock_print.assert_called_once()
    assert "OpenAI API error during synthesis:" in mock_print.call_args[0][0]


@pytest.mark.asyncio
async def test_synthesize_speech_line_rate_limit_then_success(
    mocker: MagicMock,
) -> None:
    """Test synthesize_speech_line retries on RateLimitError and succeeds."""
    mock_openai_client = AsyncMock(spec=AsyncOpenAI)
    # Mock the response object returned by create
    mock_stream_response_success = AsyncMock()

    async def _mock_astream_to_file_rate_limit(*_args: Any, **_kwargs: Any) -> None:
        return None

    mock_stream_response_success.astream_to_file = AsyncMock(
        side_effect=_mock_astream_to_file_rate_limit
    )

    # Simulate RateLimitError twice, then success
    mock_openai_client.audio.speech.create.side_effect = [
        RateLimitError(
            "rate limited 1", response=MagicMock(), body=None
        ),  # Keep response/body for RateLimitError as it might be different
        RateLimitError("rate limited 2", response=MagicMock(), body=None),
        mock_stream_response_success,  # This is the successful response
    ]
    mock_asyncio_sleep = mocker.patch("asyncio.sleep", AsyncMock())
    mocker.patch(
        "dialogue_tts_core.tts_client.is_content_safe", AsyncMock(return_value=True)
    )
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.path.getsize", return_value=1024)

    output_p = "test_retry_success.mp3"
    result = await synthesize_speech_line(
        client=mock_openai_client,
        text="hello retry",
        voice="alloy",
        output_path=output_p,
        line_index=5,
        cache_base_dir="test_cache",
    )
    assert result[0] == output_p
    assert result[1] is False  # Not from cache
    assert (
        mock_openai_client.audio.speech.create.call_count == 3
    )  # Initial call + 2 retries
    assert mock_asyncio_sleep.call_count == 2
    # Check sleep durations (approximate due to min/max logic)
    call_args = mock_asyncio_sleep.call_args_list
    assert call_args[0].args[0] >= INITIAL_BACKOFF_SECONDS
    assert call_args[1].args[0] >= INITIAL_BACKOFF_SECONDS * 2
    mock_stream_response_success.astream_to_file.assert_awaited_once_with(
        output_p
    )  # Check on the correct mock


@pytest.mark.asyncio
async def test_synthesize_speech_line_max_retries_reached(mocker: MagicMock) -> None:
    """Test synthesize_speech_line returns None after max retries for RateLimitError."""
    mock_openai_client = AsyncMock(spec=AsyncOpenAI)

    # Simulate RateLimitError for all attempts (MAX_RETRIES + 1)
    side_effects = [
        RateLimitError(f"rate limited {i + 1}", response=MagicMock(), body=None)
        for i in range(MAX_RETRIES + 1)
    ]
    mock_openai_client.audio.speech.create.side_effect = side_effects

    mock_asyncio_sleep = mocker.patch("asyncio.sleep", AsyncMock())
    mocker.patch(
        "dialogue_tts_core.tts_client.is_content_safe", AsyncMock(return_value=True)
    )
    mock_print = mocker.patch("builtins.print")

    output_p = "test_max_retries.mp3"
    result = await synthesize_speech_line(
        client=mock_openai_client,
        text="hello max retry",
        voice="alloy",
        output_path=output_p,
        line_index=6,
        cache_base_dir="test_cache",
    )
    assert result[0] is None
    assert result[1] is False  # Not from cache
    assert mock_openai_client.audio.speech.create.call_count == MAX_RETRIES + 1
    assert mock_asyncio_sleep.call_count == MAX_RETRIES
    # Remove strict assert_called_once as print is called multiple times during retries
    # assert "Max retries reached due to RateLimitError." in mock_print.call_args[0][0]
    # This assertion is also too strict
    # We can assert that the final print call contains the expected message
    mock_print.assert_called_with(
        f"Line 6: Max retries reached due to RateLimitError. Error: "
        f"rate limited {MAX_RETRIES + 1}",
    )
    # Also assert that the intermediate retry messages were printed
    assert any("Rate limit hit" in call[0][0] for call in mock_print.call_args_list)


@pytest.mark.asyncio
async def test_synthesize_speech_line_output_file_missing_or_empty(
    mocker: MagicMock,
) -> None:
    """Test synthesize_speech_line returns None if output file is missing or empty."""
    mock_openai_client = AsyncMock(spec=AsyncOpenAI)
    # Mock the response object returned by create
    mock_stream_response = AsyncMock()

    async def _mock_astream_to_file_missing(*_args: Any, **_kwargs: Any) -> None:
        return None

    mock_stream_response.astream_to_file = AsyncMock(
        side_effect=_mock_astream_to_file_missing
    )

    mock_openai_client.audio.speech.create.return_value = mock_stream_response

    mocker.patch(
        "dialogue_tts_core.tts_client.is_content_safe", AsyncMock(return_value=True)
    )
    # Simulate file not existing OR being empty
    mocker.patch("os.path.exists", return_value=False)
    # os.path.getsize won't be called if os.path.exists is False,
    # but if testing specifically for empty file, mock os.path.exists=True and
    # os.path.getsize=0. For this test name, let's assume it first checks
    # exists then size.
    mocker.patch("os.path.getsize", return_value=0)  # Keep this mock for completeness

    mock_print = mocker.patch("builtins.print")

    output_p = "test_missing_empty.mp3"
    result = await synthesize_speech_line(
        client=mock_openai_client,
        text="hello",
        voice="alloy",
        output_path=output_p,
        line_index=7,
        cache_base_dir="test_cache",
    )
    assert result[0] is None
    assert result[1] is False  # Not from cache
    mock_openai_client.audio.speech.create.assert_called_once()
    # astream_to_file should still be called before the os.path.exists check
    mock_stream_response.astream_to_file.assert_awaited_once_with(output_p)
    mock_print.assert_called_once()
    assert (
        "Synthesis appeared to succeed but output file is missing or empty:"
        in mock_print.call_args[0][0]
    )


@pytest.mark.asyncio
async def test_synthesize_speech_line_different_model_params(mocker: MagicMock) -> None:
    """Test synthesize_speech_line handles different model parameters (speed,
    instructions)."""
    mock_openai_client = AsyncMock(spec=AsyncOpenAI)
    # Mock the response object returned by create
    mock_stream_response = AsyncMock()

    async def _mock_astream_to_file_params(*_args: Any, **_kwargs: Any) -> None:
        return None

    mock_stream_response.astream_to_file = AsyncMock(
        side_effect=_mock_astream_to_file_params
    )

    mock_openai_client.audio.speech.create.return_value = mock_stream_response

    mocker.patch(
        "dialogue_tts_core.tts_client.is_content_safe", AsyncMock(return_value=True)
    )
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.path.getsize", return_value=1024)

    output_p_1 = "test_model_params_1.mp3"
    # Test tts-1 with speed
    path_1, was_cached_1 = await synthesize_speech_line(
        client=mock_openai_client,
        text="hello fast",
        voice="alloy",
        output_path=output_p_1,
        model="tts-1",
        speed=1.5,
        line_index=8,
        cache_base_dir="test_cache",
    )
    assert path_1 == output_p_1
    assert was_cached_1 is False
    mock_openai_client.audio.speech.create.assert_called_with(
        model="tts-1",
        input="hello fast",
        voice="alloy",
        response_format="mp3",
        speed=1.5,
    )
    mock_stream_response.astream_to_file.assert_awaited_with(output_p_1)

    output_p_2 = "test_model_params_2.mp3"
    # Test gpt-4o-mini-tts with instructions
    path_2, was_cached_2 = await synthesize_speech_line(
        client=mock_openai_client,
        text="hello instructed",
        voice="alloy",
        output_path=output_p_2,
        model="gpt-4o-mini-tts",
        instructions="speak like a robot",
        line_index=9,
        cache_base_dir="test_cache",
    )
    assert path_2 == output_p_2
    assert was_cached_2 is False
    mock_openai_client.audio.speech.create.assert_called_with(
        model="gpt-4o-mini-tts",
        input="hello instructed",
        voice="alloy",
        response_format="mp3",
        instructions="speak like a robot",
    )
    mock_stream_response.astream_to_file.assert_awaited_with(output_p_2)

    output_p_3 = "test_model_params_3.mp3"
    # Test tts-1-hd with default speed (should not include speed param)
    path_3, was_cached_3 = await synthesize_speech_line(
        client=mock_openai_client,
        text="hello default speed",
        voice="alloy",
        output_path=output_p_3,
        model="tts-1-hd",
        speed=1.0,
        line_index=10,
        cache_base_dir="test_cache",
    )
    assert path_3 == output_p_3
    assert was_cached_3 is False
    mock_openai_client.audio.speech.create.assert_called_with(
        model="tts-1-hd",
        input="hello default speed",
        voice="alloy",
        response_format="mp3",
        # speed=1.0 should NOT be present
    )
    mock_stream_response.astream_to_file.assert_awaited_with(output_p_3)

    output_p_4 = "test_model_params_4.mp3"
    # Test tts-1-hd with instructions (should not include instructions param)
    path_4, was_cached_4 = await synthesize_speech_line(
        client=mock_openai_client,
        text="hello no instructions",
        voice="alloy",
        output_path=output_p_4,
        model="tts-1-hd",
        instructions="speak like a robot",
        line_index=11,
        cache_base_dir="test_cache",
    )
    assert path_4 == output_p_4
    assert was_cached_4 is False
    mock_openai_client.audio.speech.create.assert_called_with(
        model="tts-1-hd",
        input="hello no instructions",
        voice="alloy",
        response_format="mp3",
        # instructions should NOT be present
    )
    mock_stream_response.astream_to_file.assert_awaited_with(output_p_4)
