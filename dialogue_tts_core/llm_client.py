"""
Provides an interface to interact with various LLMs for text generation tasks,
such as creating dialogue scripts or educational content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai import OpenAIError

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# Default LLM model names
DEFAULT_OPENAI_CHAT_MODEL = "gpt-3.5-turbo"

# API call parameters (can be overridden in function calls)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1500


async def generate_openai_chat_completion(
    client: AsyncOpenAI | None,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str | None:
    """
    Generates text content using OpenAI's chat completion API.

    Args:
        client: An initialized AsyncOpenAI client instance.
        model: The OpenAI model identifier (e.g., "gpt-3.5-turbo", "gpt-4").
        system_prompt: The system message to guide the LLM's behavior.
        user_prompt: The user's main request or prompt.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        The generated text content from the LLM if successful, otherwise None.
    """
    if client is None:
        print("Error: OpenAI client is not initialized.")
        return None

    messages_list = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages_list,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if response and response.choices and len(response.choices) > 0:
            generated_content = response.choices[0].message.content
            if generated_content:
                return generated_content.strip()
            # This case implies generated_content is None or empty string
            print("Error: LLM response content is empty.")
            return None
        # This case implies response or response.choices is problematic
        print("Error: Invalid or empty response from LLM.")
        return None
    except OpenAIError as e:
        # Specific OpenAI errors are caught and reported
        print(f"OpenAI API error during chat completion: {type(e).__name__} - {e}")
        return None
    except RuntimeError as e:  # pylint: disable=broad-except
        # Catching general exceptions to prevent crashes and log the error
        print(
            f"Unexpected runtime error during chat completion: {type(e).__name__} - {e}"
        )
        return None


async def generate_text(
    client_provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    # Provider-specific arguments:
    # For OpenAI - ensure AsyncOpenAI is imported or use forward ref if needed
    openai_client: AsyncOpenAI | None = None,
    fal_api_key: str | None = None,  # For Fal
) -> str | None:
    if client_provider == "openai":
        if openai_client is None:
            print(
                "Error: OpenAI client is required for 'openai' provider "
                "but not provided."
            )
            # Consider logging instead of print for library code
            return None

        # Call existing OpenAI generation logic
        # Ensure generate_openai_chat_completion is awaitable and called correctly
        # Assuming generate_openai_chat_completion is the name of your existing function
        return await generate_openai_chat_completion(
            client=openai_client,
            model=model,  # This model is specific to OpenAI
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if client_provider == "fal":
        if fal_api_key is None:
            print("Error: Fal API key is required for 'fal' provider but not provided.")
            # Consider logging
            return None
        # Placeholder for Fal LLM client implementation
        # generated_content = await _generate_fal_text(
        #     api_key=fal_api_key,
        #     model=model, # This model is specific to Fal
        #     system_prompt=system_prompt,
        #     user_prompt=user_prompt,
        #     temperature=temperature,
        #     max_tokens=max_tokens
        # )
        # return generated_content
        print(
            f"Info: Fal provider with model '{model}' not yet implemented. "
            f"System prompt: '{system_prompt}', User prompt: '{user_prompt}'"
        )  # More informative print
        return None  # Placeholder

    print(f"Error: Unknown LLM provider '{client_provider}'.")
    # Consider logging
    return None


# Placeholder for Fal text generation (can be added later,
# not part of this immediate task)
# async def _generate_fal_text(
# api_key: str, model: str, system_prompt: str, user_prompt: str,
# temperature: float, max_tokens: int
# ) -> str | None:
#   # SETUP headers with api_key
#   # SETUP payload with prompts, model, temp, max_tokens
#   # TRY:
#   #     POST request to Fal API endpoint for text generation
#   #     (e.g. using httpx or aiohttp for async)
#   #     CHECK response status
#   #     EXTRACT text from response
#   #     RETURN extracted text
#   # CATCH Fal API errors or network errors:
#   #     PRINT error details
#   #     RETURN None
#   # CATCH other exceptions:
#   #     PRINT error details
#   #     RETURN None
#   print("Fal text generation not implemented.")
#   return None
