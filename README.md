# Lesson Audio Kit

A modular application for generating educational lesson content and converting it into audio using Text-to-Speech (TTS) technologies.

## Project Structure

- `dialogue_tts_core/`: Core library for parsing, TTS client interactions, and audio utilities.
- `gradio_frontend/`: Gradio-based web interface for the application.
- `streamlit_frontend/`: (Planned) Streamlit-based web interface.
- `tests/`: Pytest unit and integration tests.
- `scripts/`: Helper scripts for development and maintenance.
- `.vscode/`: VS Code workspace settings.
- `pyproject.toml`: Python project configuration and dependencies (Poetry).

## API Documentation

- [OpenAPI 3.1.0 Specification](docs/api/openapi.yaml)

## Setup

1. **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd lesson-audio-kit
    ```

2. **Install Python and Poetry:**
    Ensure you have Python (e.g., 3.11+) and Poetry installed.

3. **Create and configure environment variables:**
    Copy `.env.example` to `.env` and fill in your API keys and other configurations:

    ```bash
    cp .env.example .env
    # Edit .env with your actual secrets
    ```

    *At a minimum, `OPENAI_API_KEY` is required for the Gradio app to function.*

4. **Install dependencies using Poetry:**

    ```bash
    poetry install
    ```

## Running the Gradio Application

1. Activate the Poetry environment:

    ```bash
    poetry shell
    ```

2. Navigate to the Gradio frontend directory and run the app:

    ```bash
    cd gradio_frontend
    python app.py
    ```

    The application should now run successfully if `OPENAI_API_KEY` is set in your `.env` file (which is loaded by `python-dotenv` in `app.py`).

## Development

- **Linting and Formatting:** This project uses Ruff for linting and Black for formatting. VS Code is configured to format on save.
- **Testing:** Run tests using Pytest:

  ```bash
  poetry shell
  pytest
  ```

### Live API Testing (VCR Cassette Recording)

Tests that interact with the live OpenAI API (e.g., for recording VCR cassettes) are marked with `@pytest.mark.live`.

To run these tests in CI in a way that records new cassettes (e.g., on your fork if you've made changes requiring new recordings), you will need to add an `OPENAI_API_KEY` secret to your GitHub repository settings:

1. Go to your forked repository on GitHub.
2. Navigate to `Settings` > `Secrets and variables` > `Actions`.
3. Click `New repository secret`.
4. Name: `OPENAI_API_KEY`
5. Value: Your actual OpenAI API key.

Without this secret, CI will use existing cassettes or skip these tests if cassettes are missing and `OPENAI_API_KEY` is not set in the environment.

#### Running Live API Tests Locally

To run tests that hit the live OpenAI API locally (e.g., to record new VCR cassettes or debug live interactions), you can use the provided script. This script retrieves your OpenAI API key from 1Password (ensure `op` CLI is configured and you have access to the specified secret path `op://personal/openai/key`).

1. Make the script executable:

    ```bash
    chmod +x scripts/run_tests_live.sh
    ```

2. Run the script from the project root:

    ```bash
    ./scripts/run_tests_live.sh
    ```

This will set the `OPENAI_API_KEY` environment variable for the session and run `pytest tests/core/test_tts_client.py -m "live"`.

## TODO

- Implement Streamlit frontend.
- Add more TTS providers.
- Develop LLM client for content generation.
- Set up CI/CD pipeline.
