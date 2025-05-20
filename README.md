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

## TODO

- Implement Streamlit frontend.
- Add more TTS providers.
- Develop LLM client for content generation.
- Set up CI/CD pipeline.
