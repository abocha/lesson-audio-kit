---
title: Dialogue TTS
emoji: 🗣️🎙️
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Dialogue Script to Speech Synthesis

This Hugging Face Space converts dialogue scripts into speech using OpenAI's TTS models (`tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`).

## Features

* **Input Script**: Provide a dialogue script with lines in the format `[Speaker] Utterance`.
* **TTS Models**: Choose from `tts-1`, `tts-1-hd`, or `gpt-4o-mini-tts`.
* **Voice Configuration**:
  * **Single Global Voice**: Use one voice for all speakers.
  * **Random per Speaker**: Assigns a unique random voice to each speaker consistently within a run.
  * **A/B Round Robin**: Cycles through available voices for each unique speaker.
  * **Detailed Per-Speaker UI**: Configure voice, speed (for `tts-1/hd`), and emotional vibe/custom instructions (for `gpt-4o-mini-tts`) for each speaker individually.
* **Output**:
  * A ZIP file containing individual MP3s for each line.
  * A single merged MP3 of the entire dialogue with custom pauses.
* **Cost Estimation**: Displays an estimated cost before generating audio.
* **NSFW Check**: Optional content safety check using an external API (if `NSFW_API_URL_TEMPLATE` is configured).

## How to Use

1. **Enter your dialogue script** in the text area.
    Example:

    ```
    [Alice] Hello Bob, how are you today?
    [Bob] I'm doing great, Alice! Thanks for asking.
    [Narrator] And so their conversation began.
    ```

2. **Select the TTS Model**.
3. **Set the pause duration** (in milliseconds) between lines for the merged audio.
4. **Choose a Speaker Configuration Method**:
    * If "Single Voice (Global)", select the voice.
    * If "Detailed Configuration...", click "Load/Refresh Per-Speaker Settings UI" and adjust settings for each speaker.
    * Other methods will apply voices automatically.
5. (Optional) Adjust **Global Speed** or **Global Instructions** if applicable to your chosen model and configuration.
6. Click **"Calculate Cost"** to see an estimate.
7. Click **"Generate Audio"**.
8. Download the ZIP file or listen to/download the merged MP3.

## Secrets

This Space requires the following secrets to be set in the Hugging Face Space settings:

* `OPENAI_API_KEY`: Your OpenAI API key.
* `NSFW_API_URL_TEMPLATE` (Optional): URL template for NSFW checking, e.g., `https://api.example.com/check?text={text}`. The placeholder `{text}` will be URL-encoded.
* `MODEL_DEFAULT` (Optional): Default TTS model (e.g., `tts-1-hd`).

## Smoke Test Script

Use the following script to test basic functionality:
[Gandalf] You shall not pass!
[Frodo] I will take the Ring to Mordor.
[Gandalf] So be it.

Choose your desired model and settings (e.g., "Random per Speaker"), then generate.

## Deployment

This application is designed to be deployed as a Hugging Face Space.
Ensure `ffmpeg` is available (handled by `container.yaml` for Classic Spaces).
Set the necessary secrets in your Space settings on Hugging Face Hub.
