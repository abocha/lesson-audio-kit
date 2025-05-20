import asyncio
import os
from typing import Any  # Removed Dict, use dict instead
import urllib.parse  # For URL encoding text in NSFW check

import httpx  # For NSFW check
from openai import AsyncOpenAI, OpenAIError, RateLimitError

OPENAI_VOICES = [
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
]

# Concurrency limiter for OpenAI API calls
MAX_CONCURRENT_REQUESTS = 2
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Retry mechanism parameters
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0  # Start with 1 second
MAX_BACKOFF_SECONDS = 16.0  # Cap backoff to avoid excessively long waits


async def is_content_safe(text: str, api_url_template: str | None) -> bool:
    """
    Checks if the content is safe using an external NSFW API.
    Returns True if safe, API URL is not provided, or check fails open.
    Returns False if content is flagged as unsafe by the API.
    """
    if not api_url_template:
        return True  # No NSFW check configured, assume safe

    if "{text}" not in api_url_template:
        print(
            f"Warning: NSFW_API_URL_TEMPLATE ('{api_url_template}') does not contain "
            "{text} placeholder. Skipping NSFW check."
        )
        return True  # Configuration error, fail open (assume safe)

    try:
        encoded_text = urllib.parse.quote(text)  # Ensure text is URL-safe
        url = api_url_template.replace(
            "{text}", encoded_text
        )  # Use replace for simplicity

        # Using a timeout for the external API call
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)

        response.raise_for_status()  # Will raise an exception for 4xx/5xx responses

        # Assuming the API returns a specific response to indicate safety.
        # This part needs to be adapted to the actual API's response format.
        # For example, if it returns JSON: `data = response.json()`
        # If it returns 200 for safe, and non-200 for unsafe,
        # raise_for_status handles it.
        # For this placeholder, we'll assume 200 means safe.
        return True  # Content is safe based on API response

    except httpx.HTTPStatusError as e:
        # Log specific HTTP errors from the NSFW API
        print(
            f"NSFW Check: API request failed. Status: {e.response.status_code}. "
            f"URL: {e.request.url}. Response: {e.response.text[:200]}"
        )
        # Depending on policy, you might "fail closed" (treat as unsafe) or "fail open"
        return False  # Content flagged as unsafe or API error
    except httpx.RequestError as e:
        url = e.request.url if e.request else "N/A"
        print(f"NSFW Check: API request error: {e}. URL: {url}")
        return (
            True  # Fail open (assume safe) on network/request errors to not block TTS
        )
    except Exception as e:  # noqa: BLE001 # Final fallback for NSFW check
        print(f"NSFW Check: An unexpected error occurred: {e}")
        return True  # Fail open (assume safe) on other unexpected errors


async def synthesize_speech_line(  # noqa: C901
    client: AsyncOpenAI,
    text: str,
    voice: str,
    output_path: str,
    model: str = "tts-1-hd",
    speed: float = 1.0,  # Speed parameter (0.25 to 4.0). Default 1.0.
    instructions: str | None = None,  # For models like gpt-4o-mini-tts potentially
    nsfw_api_url_template: str | None = None,
    line_index: int = -1,  # For logging purposes
) -> str | None:
    """
    Synthesizes a single line of text to speech using OpenAI TTS.
    Handles rate limiting with exponential backoff and NSFW checks.
    Returns the output_path if successful, None otherwise.
    """
    if not text.strip():
        print(
            f"Line {line_index if line_index != -1 else '(unknown)'}: "
            "Input text is empty. Skipping synthesis."
        )
        return None

    if nsfw_api_url_template and not await is_content_safe(text, nsfw_api_url_template):
        print(
            f"Line {line_index if line_index != -1 else '(unknown)'}: "
            "Content flagged as potentially unsafe. Skipping synthesis."
        )
        return None  # Skip synthesis for flagged content

    current_retry = 0
    backoff_seconds = INITIAL_BACKOFF_SECONDS

    # Acquire semaphore before entering retry loop
    async with semaphore:
        while current_retry <= MAX_RETRIES:
            try:
                request_params: dict[str, Any] = {  # Explicitly typed
                    "model": model,
                    "input": text,
                    "voice": voice,
                    "response_format": "mp3",  # Explicitly request mp3
                }

                # Add speed if model is tts-1 or tts-1-hd and speed is not default 1.0
                if model in ["tts-1", "tts-1-hd"]:
                    # OpenAI API speed range is 0.25 to 4.0.
                    # Clamp speed to be safe, although UI should also enforce this.
                    clamped_speed = max(0.25, min(float(speed), 4.0))
                    if clamped_speed != 1.0:  # Only send if not default
                        request_params["speed"] = clamped_speed

                # Add instructions if provided and model is gpt-4o-mini-tts
                # (or other future models supporting it)
                # tts-1 and tts-1-hd do not support an 'instructions' parameter.
                if model == "gpt-4o-mini-tts" and instructions and instructions.strip():
                    request_params["instructions"] = instructions.strip()

                # Log the request params being sent (excluding sensitive parts
                # like full text if too long)
                # print(
                #     f"Line {line_index}: Sending request to OpenAI TTS with params: "
                #     f"{{'model': '{model}', 'voice': '{voice}', 'speed': "
                #     f"{request_params.get('speed', 1.0)}, 'has_instructions': "
                #     f"{bool(request_params.get('instructions'))}}}"
                # )
                response = await client.audio.speech.create(**request_params)

                # Stream response to file
                await response.astream_to_file(output_path)

                # Verify file was created and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return output_path
                line_msg_prefix = f"Line {line_index if line_index != -1 else ''}: "
                print(
                    f"{line_msg_prefix}Synthesis appeared to succeed but "
                    f"output file is missing or empty: {output_path}"
                )
                return None  # File not created or empty

            except RateLimitError as e:
                current_retry += 1
                if current_retry > MAX_RETRIES:
                    print(
                        f"Line {line_index if line_index != -1 else ''}: "
                        f"Max retries reached due to RateLimitError. Error: {e}"
                    )
                    return None

                # Exponential backoff with jitter could be added,
                # but simple exponential for now
                print(
                    f"Line {line_index if line_index != -1 else ''}: "
                    f"Rate limit hit (Attempt {current_retry}/{MAX_RETRIES}). "
                    f"Retrying in {backoff_seconds:.2f}s..."
                )
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(
                    backoff_seconds * 2, MAX_BACKOFF_SECONDS
                )  # Increase backoff, cap at max

            except OpenAIError as e:  # Catch other specific OpenAI errors
                print(
                    f"Line {line_index if line_index != -1 else ''}: "
                    f"OpenAI API error during synthesis: {type(e).__name__} - {e}"
                )
                return None

            except Exception as e:  # noqa: BLE001 # Last-resort catch for unexpected synthesis errors
                print(
                    f"Line {line_index if line_index != -1 else ''}: "
                    f"An unexpected error occurred during synthesis: "
                    f"{type(e).__name__} - {e}"
                )
                # current_retry += 1
                # Could also retry on generic errors if deemed transient
                # if current_retry > MAX_RETRIES: return None
                # await asyncio.sleep(backoff_seconds)
                # backoff_seconds = min(backoff_seconds * 2, MAX_BACKOFF_SECONDS)
                return (
                    None  # For most unexpected errors, safer not to retry indefinitely
                )

        # If loop finishes due to max retries without returning output_path
        print(
            f"Line {line_index if line_index != -1 else ''}: "
            "Failed to synthesize after all retries or due to non-retryable error."
        )
        return None
