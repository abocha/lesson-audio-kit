#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Export the OpenAI API key from 1Password
# Ensure you are logged into the 'op' CLI and have access to this secret.
export OPENAI_API_KEY=$(op read "op://personal/openai/key")

# Check if the API key was successfully retrieved
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY could not be retrieved from 1Password."
    echo "Please ensure you are logged in to 'op' CLI and have access to 'op://personal/openai/key'."
    exit 1
fi

echo "OPENAI_API_KEY has been set."

# Run pytest targeting only tests marked 'live' in the specified file
# Assumes pytest is installed in the environment and tests are run from the project root.
pytest tests/core/test_tts_client.py -m "live"

echo "Live API tests completed."
