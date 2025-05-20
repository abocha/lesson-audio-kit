import os
import pathlib
import subprocess
import tempfile


def merge_mp3_files(
    file_paths: list[str], output_filename: str, pause_ms: int = 500
) -> str | None:
    """
    Concatenate MP3s *without re-encoding*.
    Adds a silent gap (`pause_ms`) between clips by generating a
    temporary silent MP3 at the same encoding params as the sources.
    """
    if not file_paths:
        print("Warning: no input files.")
        return None

    # keep only files that exist and are non-empty
    valid = [
        p for p in file_paths if p and os.path.exists(p) and os.path.getsize(p) > 0
    ]
    if not valid:
        print("No valid audio segments found.")
        return None

    concat_list = []

    # optional pause: create one silent mp3 we can reuse
    if pause_ms > 0:
        silent_path = _make_silent_mp3(duration_ms=pause_ms, template_mp3=valid[0])
    else:
        silent_path = None

    for i, p in enumerate(valid):
        concat_list.append(p)
        if silent_path and i < len(valid) - 1:
            concat_list.append(silent_path)

    # ffmpeg concat demuxer expects a text file listing the parts
    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
        for part in concat_list:
            tf.write(f"file '{pathlib.Path(part).as_posix()}'\n")
        tf.flush()

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                tf.name,
                "-c",
                "copy",  # <- no re-encode!
                output_filename,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return output_filename


def _make_silent_mp3(duration_ms: int, template_mp3: str) -> str:
    """
    Create a silent MP3 (CBR, same sample-rate & channels as template)
    so we can insert a pause without changing codecs later.
    """
    silent_path = tempfile.mktemp(suffix=".mp3")
    # Extract params from the template
    probe = (
        subprocess.check_output(
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
            text=True,
        )
        .strip()
        .splitlines()
    )
    sr, ch, br = probe  # e.g. "44100", "2", "128000"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={sr}:cl=mono",
            "-t",
            str(duration_ms / 1000),
            "-ac",
            ch,
            "-ar",
            sr,
            "-b:a",
            br,
            silent_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return silent_path
