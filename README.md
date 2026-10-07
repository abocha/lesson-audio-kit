# Lesson Audio Kit

A modular Python toolkit for turning ESL scripts into reusable lesson audio.

The project separates script parsing, speaker configuration, TTS routing, caching, audio assembly, and UI concerns so the same core can support teacher-facing interfaces or an API without duplicating synthesis logic.

## What it does today

- parses dialogue scripts into speaker-attributed lines;
- resolves per-speaker TTS configuration;
- synthesizes lines with OpenAI TTS;
- caches generated audio to avoid paying for identical work twice;
- estimates/records routing and per-line cost information;
- merges line audio into lesson-ready output;
- exposes a Gradio UI;
- includes a FastAPI/OpenAPI surface for programmatic use;
- tests live-provider interactions with pytest recording/cassettes.

The routing model already describes multiple provider/quality/cost options, but the current synthesis orchestrator dispatches to **OpenAI only**. Broader provider dispatch remains future work.

## Why this exists

A short listening activity has a surprisingly repetitive production loop:

```text
idea / script
    |
    v
speaker parsing
    |
    v
voice configuration
    |
    v
TTS per line
    |
    +--> cache hit: reuse audio
    |
    +--> cache miss: synthesize + store
    v
audio assembly
    |
    v
lesson-ready output
```

The useful abstraction is not "call a TTS API". It is keeping speaker identity, cost, caching, output assembly, and provider-specific details outside the lesson-authoring workflow.

## Architecture

```text
gradio_frontend/
        |
        v
dialogue_tts_core/
  dialogue_script_parser.py
  speaker_config_resolver.py
  cost_router.py
  tts_orchestrator.py
  tts_client.py
  cache_manager.py
  audio_utils.py
        |
        +--> OpenAI TTS
        |
        +--> local audio cache
        v
   assembled audio
```

The core package is intentionally UI-independent. The Gradio application is one client of that core; the repository also contains API documentation and a planned/experimental Streamlit surface.

## Core modules

| Module | Responsibility |
| --- | --- |
| `dialogue_script_parser.py` | turn scripts into structured speaker lines |
| `config_models.py` | typed configuration models |
| `speaker_config_resolver.py` | resolve voice/config per speaker |
| `cost_router.py` | model TTS engines by cost, quality, latency, and capabilities |
| `tts_orchestrator.py` | coordinate per-line synthesis and collect job details |
| `tts_client.py` | provider-facing synthesis calls |
| `cache_manager.py` | reusable generated-audio cache |
| `audio_utils.py` | merge/assemble audio output |
| `llm_client.py` | text-generation client utilities |

## Routing model

The router represents TTS engines using explicit metadata such as:

- provider and model ID;
- quality tier;
- estimated latency;
- per-character or per-minute cost;
- emotion support;
- voice-cloning support.

That makes routing decisions inspectable and keeps cost policy out of the UI.

Current caveat: the engine catalog contains metadata for OpenAI, Fal.ai, ElevenLabs, Cartesia, and other candidates, but the production dispatch path in `tts_orchestrator.py` currently supports OpenAI synthesis only.

## Stack

- Python 3.13
- OpenAI Python SDK
- Gradio
- FastAPI + Uvicorn
- pydub
- Poetry
- pytest + pytest-recording
- Ruff
- GitHub Actions

## Setup

Requirements:

- Python 3.13
- Poetry
- FFmpeg available to pydub where required

Clone the repository and install dependencies:

```bash
git clone https://github.com/abocha/lesson-audio-kit.git
cd lesson-audio-kit
poetry install
```

Create a local environment file:

```bash
cp .env.example .env
```

On PowerShell:

```powershell
Copy-Item .env.example .env
```

At minimum, set:

```text
OPENAI_API_KEY=...
```

Optional settings in `.env.example` cover other provider keys, cache/output paths, and experimental integrations.

## Run the Gradio app

```bash
poetry run python gradio_frontend/app.py
```

The app loads configuration from the local environment and uses the shared core package for parsing and synthesis.

## Development

Run tests:

```bash
poetry run pytest
```

Run lint and formatting checks:

```bash
poetry run ruff check .
poetry run ruff format --check .
```

The GitHub Actions workflow runs those checks plus pytest on Python 3.13 for pushes and pull requests targeting `main`, `master`, or `develop`.

### Recorded/live API tests

Tests that require live OpenAI access are marked with `@pytest.mark.live`.

The repository uses pytest recording so most development can run against existing cassettes instead of making repeated paid/network calls. Live cassette recording requires `OPENAI_API_KEY`.

A helper script is available for the author's local 1Password-based workflow:

```bash
./scripts/run_tests_live.sh
```

That script is optional; ordinary contributors can provide `OPENAI_API_KEY` through their own environment.

## API documentation

The OpenAPI 3.1 specification is available at:

[docs/api/openapi.yaml](docs/api/openapi.yaml)

## Project status

This repository is an active prototype/core extraction rather than a finished multi-provider product.

Implemented foundations include script parsing, speaker configuration, OpenAI synthesis, caching, cost-aware engine metadata, audio assembly, UI integration, API schema, and automated tests.

Likely next steps are:

- implement provider dispatch beyond OpenAI;
- reconcile the provider catalog with real runtime adapters;
- simplify/retire older frontend experiments;
- add a concise demo showing the script → audio workflow.
