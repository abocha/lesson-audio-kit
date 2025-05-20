import subprocess
import unittest.mock as mock

import pytest
from pytest_mock import MockerFixture

# Import the functions to be tested
from dialogue_tts_core.audio_utils import _make_silent_mp3, merge_mp3_files


def test_make_silent_mp3_success(mocker: MockerFixture) -> None:
    """Tests successful generation of a silent MP3 file."""
    mock_mktemp = mocker.patch("tempfile.mktemp", return_value="dummy_silent.mp3")
    # Mock ffprobe output: sample_rate, channels, bit_rate
    mock_ffprobe = mocker.patch(
        "subprocess.check_output", return_value="44100\n2\n128000"
    )
    mock_ffmpeg_run = mocker.patch("subprocess.run")

    duration_ms = 1000
    template_mp3 = "dummy_template.mp3"

    result_path = _make_silent_mp3(duration_ms, template_mp3)

    assert result_path == "dummy_silent.mp3"
    mock_mktemp.assert_called_once_with(suffix=".mp3")
    mock_ffprobe.assert_called_once_with(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,bit_rate",
            "-of",
            "default=nw=1:nk=1",
            template_mp3,
        ],
        text=True,  # Ensure text=True is used for decoding output
    )
    expected_ffmpeg_anullsrc_call_args = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=stereo",  # Use stereo based on ffprobe mock
        "-t",
        "1.0",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-b:a",
        "128000",
        "dummy_silent.mp3",
    ]
    mock_ffmpeg_run.assert_called_once_with(
        expected_ffmpeg_anullsrc_call_args,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# Add other test cases here in subsequent steps


def test_merge_mp3_files_no_input() -> None:
    """Tests merge_mp3_files with an empty input list."""
    result = merge_mp3_files([], "output.mp3")
    assert result is None


def test_merge_mp3_files_no_valid_input(mocker: MockerFixture) -> None:
    """Tests merge_mp3_files when no input files are valid."""
    mocker.patch("os.path.exists", return_value=False)
    mocker.patch("os.path.getsize", return_value=0)

    input_files = ["file1.mp3", "file2.mp3"]
    result = merge_mp3_files(input_files, "output.mp3")
    assert result is None


def test_merge_mp3_files_with_pause(mocker: MockerFixture) -> None:
    """Tests successful merging of MP3 files with a pause."""
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.path.getsize", return_value=1024)
    mock_make_silent = mocker.patch(
        "dialogue_tts_core.audio_utils._make_silent_mp3", return_value="silent.mp3"
    )

    # Mock NamedTemporaryFile to capture its written content
    mock_temp_file_obj = mock.MagicMock()
    mock_temp_file_obj.name = "concat_list.txt"
    mock_temp_file_context_manager = mock.MagicMock()
    mock_temp_file_context_manager.__enter__.return_value = mock_temp_file_obj
    mock_temp_file_context_manager.__exit__.return_value = None
    mocker.patch(
        "tempfile.NamedTemporaryFile", return_value=mock_temp_file_context_manager
    )

    mock_ffmpeg_concat_run = mocker.patch("subprocess.run")

    input_files = ["file1.mp3", "file2.mp3"]
    output_file = "merged.mp3"
    pause_duration = 500

    result = merge_mp3_files(input_files, output_file, pause_ms=pause_duration)

    assert result == output_file
    mock_make_silent.assert_called_once_with(
        duration_ms=pause_duration, template_mp3="file1.mp3"
    )

    # Check calls to the mocked file object's write method
    expected_writes = [
        mock.call("file 'file1.mp3'\n"),
        mock.call("file 'silent.mp3'\n"),
        mock.call("file 'file2.mp3'\n"),
    ]
    mock_temp_file_obj.write.assert_has_calls(expected_writes)
    mock_temp_file_obj.flush.assert_called_once()

    mock_ffmpeg_concat_run.assert_called_once_with(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            "concat_list.txt",
            "-c",
            "copy",
            output_file,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_merge_mp3_files_no_pause(mocker: MockerFixture) -> None:
    """Tests successful merging of MP3 files without a pause."""
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.path.getsize", return_value=1024)

    # Mock NamedTemporaryFile to capture its written content
    mock_temp_file_obj = mock.MagicMock()
    mock_temp_file_obj.name = "concat_list.txt"
    mock_temp_file_context_manager = mock.MagicMock()
    mock_temp_file_context_manager.__enter__.return_value = mock_temp_file_obj
    mock_temp_file_context_manager.__exit__.return_value = None
    mocker.patch(
        "tempfile.NamedTemporaryFile", return_value=mock_temp_file_context_manager
    )

    mock_ffmpeg_concat_run = mocker.patch("subprocess.run")
    mock_make_silent = mocker.patch(
        "dialogue_tts_core.audio_utils._make_silent_mp3"
    )  # Ensure it's not called

    input_files = ["file1.mp3", "file2.mp3"]
    output_file = "merged.mp3"
    pause_duration = 0

    result = merge_mp3_files(input_files, output_file, pause_ms=pause_duration)

    assert result == output_file
    mock_make_silent.assert_not_called()

    # Check calls to the mocked file object's write method
    expected_writes = [
        mock.call("file 'file1.mp3'\n"),
        mock.call("file 'file2.mp3'\n"),
    ]
    mock_temp_file_obj.write.assert_has_calls(expected_writes)
    mock_temp_file_obj.flush.assert_called_once()

    mock_ffmpeg_concat_run.assert_called_once_with(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            "concat_list.txt",
            "-c",
            "copy",
            output_file,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_merge_mp3_files_ffmpeg_fails(mocker: MockerFixture) -> None:
    """Tests merge_mp3_files when the ffmpeg command fails."""
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.path.getsize", return_value=1024)

    # Mock NamedTemporaryFile (needed even if ffmpeg fails)
    mock_temp_file_obj = mock.MagicMock()
    mock_temp_file_obj.name = "concat_list.txt"
    mock_temp_file_context_manager = mock.MagicMock()
    mock_temp_file_context_manager.__enter__.return_value = mock_temp_file_obj
    mock_temp_file_context_manager.__exit__.return_value = None
    mocker.patch(
        "tempfile.NamedTemporaryFile", return_value=mock_temp_file_context_manager
    )

    mock_ffmpeg_concat_run = mocker.patch(
        "subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")
    )

    input_files = ["file1.mp3", "file2.mp3"]
    output_file = "merged.mp3"

    with pytest.raises(subprocess.CalledProcessError):
        merge_mp3_files(input_files, output_file)

    mock_ffmpeg_concat_run.assert_called_once()  # Ensure ffmpeg was attempted
