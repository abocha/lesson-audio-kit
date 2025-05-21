# streamlit_frontend/app.py
import time
from typing import Any, Optional  # Added for type hinting

import requests  # Or httpx for async if preferred for API calls
import streamlit as st

# Import constants for selectbox options (or define them here for simplicity in v0)
# from dialogue_tts_core.config_models import (
# OPENAI_VOICES_TUPLE, TTS_MODELS_AVAILABLE_TUPLE
# )
# For v0, hardcode for simplicity if direct import is complex now
OPENAI_VOICES_TUPLE = (
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
)  # Added "verse" as per previous phases
TTS_MODELS_AVAILABLE_TUPLE = ("tts-1-hd", "gpt-4o-mini-tts", "tts-1")
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "alloy"
MOCK_AUDIO_URL = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"


# --- Helper Function for API Calls (Conceptual) ---
def submit_tts_request(api_base_url: str, _payload: dict) -> Optional[dict]:
    try:
        # MOCK IMPLEMENTATION FOR V0
        # In a real scenario, this would be:
        # response = requests.post(f"{api_base_url}/tts", json=payload, timeout=10)
        # response.raise_for_status()
        # return response.json()

        st.session_state.status_message = "Simulating API submission..."
        time.sleep(1)  # Simulate network delay
        mock_job_id = f"fake_job_{int(time.time())}"
        # Ensure the status_url is constructed correctly based on api_base_url
        # If api_base_url is "http://localhost:8000/api", then status_url
        # should be "http://localhost:8000/api/tts/status/{mock_job_id}"
        # However, the provided pseudocode for poll_job_status suggests
        # status_url_from_api might be a full URL.
        # Let's assume the submit function returns a full status_url.
        # If api_base_url = "http://host/api", then status_url = "http://host/api/tts/status/job_id"
        # If api_base_url = "http://host", then status_url would be
        # "http://host/tts/status/job_id" (if /api is not part of base)
        # For consistency with the polling logic, let's assume the API returns
        # a full URL or a path that needs careful joining.
        # The pseudocode has:
        # st.session_state.status_url_from_api = job_creation_response["status_url"]
        # And then poll_job_status(st.session_state.status_url_from_api)
        # This implies status_url is absolute.

        # Let's ensure api_base_url ends with /api if it's meant to be
        # the root for /tts and /tts/status
        # Or, if /api is already part of the paths,
        # then api_base_url is just the server root.
        # The default is "http://localhost:8000/api", so /tts becomes
        # "http://localhost:8000/api/tts"
        # and status would be
        # "http://localhost:8000/api/tts/status/{job_id}"

        # For the mock, let's construct the full status URL
        # based on the current api_base_url
        # Ensure no double slashes if api_base_url ends with / and path starts with /
        base_for_status = api_base_url.strip("/")
        status_url = f"{base_for_status}/tts/status/{mock_job_id}"

        return {
            "job_id": mock_job_id,
            "status_url": status_url,
            "message": "Job submitted successfully (mocked).",
        }
    except requests.exceptions.RequestException as e:
        st.session_state.status_message = f"API Error (submit): {e}"
        return None
    except Exception as e:  # Catch any other error during mock  # noqa: BLE001
        st.session_state.status_message = f"Mock Submit Error: {e}"
        return None


def poll_job_status(status_url: str) -> Optional[dict]:
    try:
        # MOCK IMPLEMENTATION FOR V0
        # In a real scenario, this would be:
        # response = requests.get(status_url, timeout=10)
        # response.raise_for_status()
        # return response.json()

        st.session_state.status_message = (
            f"Simulating polling status from {status_url}..."
        )
        time.sleep(1)  # Simulate network delay

        # Simple mock: alternate between processing and completed/failed
        if "last_poll_status" not in st.session_state:
            st.session_state.last_poll_status = "processing"
            # Ensure outputs are structured as expected by the main logic
            return {
                "status": "processing",
                "message": "Job is currently processing (mocked).",
                "outputs": {},
            }
        if st.session_state.last_poll_status == "processing":
            st.session_state.last_poll_status = "completed"
            # For st.audio, a real URL or local file path is needed.
            # For mock, we can use a placeholder or
            # a known small audio file if available.
            # Using placeholder URLs for now.
            # The main code expects "merged_mp3_url" and "zip_file_url" under "outputs"
            return {
                "status": "completed",
                "message": "Job completed successfully (mocked)!",
                "outputs": {
                    "merged_mp3_url": MOCK_AUDIO_URL,  # Placeholder
                    "zip_file_url": "mock_files.zip",  # Placeholder
                },
            }
        # To allow re-polling to show "completed" again or reset for new job:
        # This simple mock will always go processing -> completed.
        # For a new job, job_id changes, so last_poll_status should ideally be
        # reset or tied to job_id.
        # For simplicity in v0, this is okay. A new job submission will reset job_id.
        # If we want to simulate failure:
        # elif st.session_state.last_poll_status == "processing":
        #     st.session_state.last_poll_status = "failed"
        #     return {
        #         "status": "failed",
        #         "error_message": "Job failed during processing (mocked).",
        #         "outputs": {}
        #     }
        # e.g. if it was completed, and we poll again for same job_id
        return {
            "status": "completed",
            "message": "Job already completed (mocked).",
            "outputs": {
                "merged_mp3_url": MOCK_AUDIO_URL,
                "zip_file_url": "mock_files.zip",
            },
        }

    except requests.exceptions.RequestException as e:
        st.session_state.status_message = f"API Error (poll): {e}"
        return None
    except Exception as e:  # Catch any other error during mock  # noqa: BLE001
        st.session_state.status_message = f"Mock Poll Error: {e}"
        return None


# --- Initialize Session State ---
# Helper to initialize a key if not present
def init_session_state(key: str, value: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


init_session_state("job_id", None)
init_session_state(
    "job_status", "idle"
)  # "idle", "submitted", "polling", "completed", "failed"
init_session_state("merged_audio_url", None)
init_session_state("zip_audio_url", None)
init_session_state("status_message", "Enter script and generate audio.")
init_session_state("api_base_url", "http://localhost:8000/api")  # Default
init_session_state("tts_model_selected", DEFAULT_TTS_MODEL)
init_session_state(
    "status_url_from_api", None
)  # To store the full status URL from API response
# init_session_state('last_poll_status', None) # For mock polling logic,
# reset with job_id

# --- UI Layout ---
st.set_page_config(layout="wide")  # Set this first
st.title("Lesson Audio Kit - Streamlit Interface v0")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    # API Base URL
    st.session_state.api_base_url = st.text_input(
        "API Base URL", value=st.session_state.api_base_url
    )

    st.divider()
    st.header("Global TTS Options")

    # TTS Model Selection
    # Store the originally selected model to detect changes

    selected_model_for_ui = st.selectbox(
        "TTS Model",
        options=TTS_MODELS_AVAILABLE_TUPLE,
        index=(
            TTS_MODELS_AVAILABLE_TUPLE.index(st.session_state.tts_model_selected)
            if st.session_state.tts_model_selected in TTS_MODELS_AVAILABLE_TUPLE
            else 0
        ),
        key="tts_model_selector_key",
    )

    # Update session_state.tts_model_selected if the selectbox value changed
    if selected_model_for_ui != st.session_state.tts_model_selected:
        st.session_state.tts_model_selected = selected_model_for_ui
        # No need to st.rerun() here if conditional elements are just below
        # and script re-runs anyway.
        # However, if the change needs to affect something before this point
        # or ensure clean state, rerun is safer.
        # For this simple conditional display, direct use of
        # st.session_state.tts_model_selected is fine.
        # If st.rerun() is used, ensure it's handled correctly to avoid infinite loops.
        # Let's follow the original pseudocode's guidance on st.rerun() for clarity.
        st.rerun()

    # Global Voice
    global_voice = st.selectbox(
        "Global Voice",
        options=OPENAI_VOICES_TUPLE,
        index=(
            OPENAI_VOICES_TUPLE.index(DEFAULT_VOICE)
            if DEFAULT_VOICE in OPENAI_VOICES_TUPLE
            else 0
        ),
        key="global_voice_selector_key",
    )

    # Conditional UI for Speed
    # These values need to be available when constructing the payload
    global_speed_value = 1.0  # Default if not shown/used
    if st.session_state.tts_model_selected in ["tts-1", "tts-1-hd"]:
        global_speed_value = st.slider(
            "Global Speed", 0.25, 4.0, 1.0, 0.05, key="global_speed_slider"
        )

    # Conditional UI for Instructions
    global_instructions_value = ""  # Default if not shown/used
    if st.session_state.tts_model_selected == "gpt-4o-mini-tts":
        global_instructions_value = st.text_area(
            "Global Instructions",
            placeholder="e.g., Speak with a calm tone.",
            key="global_instructions_text_area",
        )

    # Pause
    global_pause_ms_value = st.number_input(
        "Pause Between Lines (ms)", 0, 5000, 500, 50, key="global_pause_ms_input"
    )

# --- Main Content Area ---
st.subheader("Dialogue Input")
dialogue_script_input = st.text_area(
    "Enter Dialogue Script:",
    height=200,  # As per instruction
    placeholder="[Alice] Hello there!\n[Bob] General Kenobi!",
    key="dialogue_script_input_area",
)

# Placeholder for future "Generate Dialogue Idea (LLM)" button
# st.button("Generate Dialogue Idea (LLM)", disabled=True)

if st.button("Generate & Synthesize Audio", key="generate_audio_button"):
    if not dialogue_script_input.strip():
        st.session_state.status_message = "Error: Dialogue script cannot be empty."
        st.session_state.job_status = "idle"
        # Clear previous results if any
        st.session_state.job_id = None
        st.session_state.merged_audio_url = None
        st.session_state.zip_audio_url = None
        st.session_state.status_url_from_api = None
        if "last_poll_status" in st.session_state:  # Reset mock poll state
            del st.session_state.last_poll_status
    else:
        # Reset state for a new job
        st.session_state.job_id = None
        st.session_state.merged_audio_url = None
        st.session_state.zip_audio_url = None
        st.session_state.job_status = "submitted"  # Initial status before API call
        st.session_state.status_message = "Submitting TTS job..."
        st.session_state.status_url_from_api = None
        if "last_poll_status" in st.session_state:  # Reset mock poll state
            del st.session_state.last_poll_status

        # Construct payload
        payload = {
            "script_text": dialogue_script_input,
            "tts_global_model": st.session_state.tts_model_selected,
            "global_pause_ms": global_pause_ms_value,
            "speaker_config_method": "global",  # Hardcoded for v0
            "global_speaker_config": {
                "voice": global_voice,
                # Speed and instructions depend on the selected model
            },
        }

        if st.session_state.tts_model_selected in ["tts-1", "tts-1-hd"]:
            payload["global_speaker_config"]["speed"] = global_speed_value
        else:  # For gpt-4o-mini-tts, ensure speed is 1.0 or not sent if model is strict
            payload["global_speaker_config"]["speed"] = 1.0

        if (
            st.session_state.tts_model_selected == "gpt-4o-mini-tts"
            and global_instructions_value
            and global_instructions_value.strip()
        ):
            payload["global_speaker_config"]["custom_instructions"] = (
                global_instructions_value
            )
            # else: custom_instructions can be omitted if empty

        # Simulate API call
        # This will re-run the script, so status_message should be updated by it.
        st.session_state.status_message = (
            "Attempting to submit TTS job..."  # Intermediate message
        )
        # Rerun to show "Attempting to submit..." before potential delay
        st.rerun()

# This block will run after the rerun from the button click, or on any interaction
if (
    st.session_state.job_status == "submitted" and st.session_state.job_id is None
):  # Check if we need to submit
    # This ensures submission happens after the rerun triggered by the button.
    # The actual payload construction and call are done here.
    # Re-retrieve values as they might be reset or not directly passed
    # if not careful with rerun.
    # However, streamlit widgets preserve their state across reruns if keys are used.
    # The payload constructed above the button is fine.

    # Reconstruct payload (ensure values are current from widgets)
    # This is slightly redundant but ensures values from this script run are used.
    # The values like global_voice, global_speed_value etc. are from
    # the current run's widget states.
    current_dialogue_script = st.session_state.get(
        "dialogue_script_input_area", ""
    )  # Get from state if available

    payload = {
        "script_text": current_dialogue_script,  # Use state-retrieved value
        "tts_global_model": st.session_state.tts_model_selected,
        "global_pause_ms": st.session_state.get(
            "global_pause_ms_input", 500
        ),  # Get from state
        "speaker_config_method": "global",
        "global_speaker_config": {
            "voice": st.session_state.get(
                "global_voice_selector_key", DEFAULT_VOICE
            ),  # Get from state
        },
    }
    if st.session_state.tts_model_selected in ["tts-1", "tts-1-hd"]:
        payload["global_speaker_config"]["speed"] = st.session_state.get(
            "global_speed_slider", 1.0
        )  # Get from state
    else:
        payload["global_speaker_config"]["speed"] = 1.0

    if st.session_state.tts_model_selected == "gpt-4o-mini-tts":
        instr = st.session_state.get(
            "global_instructions_text_area", ""
        )  # Get from state
        if instr and instr.strip():
            payload["global_speaker_config"]["custom_instructions"] = instr

    api_url_value = st.session_state.api_base_url
    if api_url_value is None:
        job_creation_response = None
    else:
        job_creation_response = submit_tts_request(api_url_value, payload)

    if job_creation_response and "job_id" in job_creation_response:
        st.session_state.job_id = job_creation_response["job_id"]
        st.session_state.status_url_from_api = job_creation_response.get(
            "status_url"
        )  # Store full status URL
        st.session_state.status_message = job_creation_response.get(
            "message",
            (f"Job submitted. ID: {st.session_state.job_id}. Waiting for status..."),
        )
        st.session_state.job_status = "polling"
        if "last_poll_status" in st.session_state:  # Reset mock poll state for new job
            del st.session_state.last_poll_status
    else:  # job_creation_response is None or doesn't have "job_id"
        if api_url_value is None:
            st.session_state.status_message = (
                "Error: API Base URL is not configured. Please check the sidebar."
            )
        elif st.session_state.status_message in [
            "Submitting TTS job...",
            "Attempting to submit TTS job...",
        ]:
            # This case implies submit_tts_request might have returned None
            # without setting a specific error,
            # or the response was malformed.
            st.session_state.status_message = (
                "Failed to submit TTS job. Check API connection or console for details."
            )
        elif (
            job_creation_response is not None and "job_id" not in job_creation_response
        ):
            st.session_state.status_message = (
                "API response error: 'job_id' missing "
                "from successful-looking submission."
            )
        # If none of the above, the status_message was likely set by
        # submit_tts_request itself (e.g., on requests.RequestException)
        # or by the api_url_value check, so we preserve it.
        st.session_state.job_status = "failed"
    st.rerun()  # Rerun to update UI based on new state (e.g., show Refresh button)


# --- Output Section ---
st.divider()
st.subheader("Results")

status_placeholder = st.empty()
status_placeholder.info(st.session_state.status_message)  # Display current status

# Add a refresh button if job is polling and has a job_id and status_url
if (
    st.session_state.job_status == "polling"
    and st.session_state.job_id
    and st.session_state.status_url_from_api
    and st.button("Refresh Job Status", key="refresh_status_button")
):
    # Call poll_job_status with the stored full URL
    job_status_response = poll_job_status(st.session_state.status_url_from_api)

    if job_status_response:
        new_status = job_status_response.get("status", "unknown")
        # Update message based on response, fallback to generic status
        st.session_state.status_message = job_status_response.get(
            "message",
            job_status_response.get("error_message", f"Current status: {new_status}"),
        )

        # Only update job_status if it's a recognized one,
        # or keep polling if unknown
        if new_status in [
            "completed",
            "failed",
            "processing",
            "pending",
        ]:  # "pending" and "processing" are effectively "polling" for UI
            st.session_state.job_status = (
                new_status if new_status not in ["pending", "processing"] else "polling"
            )

        if st.session_state.job_status == "completed":  # Check against the new status
            outputs = job_status_response.get("outputs", {})
            st.session_state.merged_audio_url = outputs.get("merged_mp3_url")
            st.session_state.zip_audio_url = outputs.get("zip_file_url")
            # Message already set from response, or can refine here
            st.session_state.status_message = (
                f"Job '{st.session_state.job_id}' completed: "
                f"{job_status_response.get('message', 'Audio ready!')}"
            )

        elif st.session_state.job_status == "failed":
            st.session_state.status_message = (
                f"Job '{st.session_state.job_id}' failed: "
                f"{job_status_response.get('error_message', 'Unknown error')}"
            )
    else:
        # poll_job_status should set status_message on error, but fallback
        if (
            not st.session_state.status_message
            or "Polling" in st.session_state.status_message
        ):
            st.session_state.status_message = (
                f"Failed to get status for job ID: {st.session_state.job_id}. "
                "Check connection."
            )
        # Optionally, set job_status to 'failed' or keep 'polling' to allow retry
        # st.session_state.job_status = "failed" # Or keep as polling
    st.rerun()  # Rerun to reflect updated status and potentially audio player
# else: # Button not clicked, status_placeholder already updated above

# Display audio player and download links if URLs are available
if st.session_state.merged_audio_url and st.session_state.job_status == "completed":
    st.audio(st.session_state.merged_audio_url, format="audio/mp3")
    st.markdown(f"[Download Merged MP3]({st.session_state.merged_audio_url})")
else:
    # Clear audio if not completed or URL is gone
    if (
        st.session_state.job_status != "completed"
        and st.session_state.merged_audio_url is not None
    ):
        st.session_state.merged_audio_url = (
            None  # Ensure audio player disappears if job fails or resets
        )

if st.session_state.zip_audio_url and st.session_state.job_status == "completed":
    st.markdown(f"[Download Individual Lines (ZIP)]({st.session_state.zip_audio_url})")
else:
    if (
        st.session_state.job_status != "completed"
        and st.session_state.zip_audio_url is not None
    ):
        st.session_state.zip_audio_url = None

# Debugging: Show session state
# with st.expander("Session State (Debug)"):
#    st.json(st.session_state.to_dict())
