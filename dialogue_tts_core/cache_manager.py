# dialogue_tts_core/cache_manager.py

import hashlib
import json  # For metadata or complex key components
import os
import pathlib
import shutil
from typing import Any, Optional


# --- Cache Key Generation ---
def generate_cache_key(
    text: str,
    voice: str,
    model: str,
    speed: Optional[float],
    instructions: Optional[str],
) -> str:
    # Normalize inputs
    normalized_text = text.lower().strip()
    normalized_voice = voice.lower().strip()
    normalized_model = model.lower().strip()

    # Create a dictionary of parameters for consistent hashing order.
    # Include speed/instructions only if not None.
    key_components: dict[str, Any] = {
        "text": normalized_text,
        "voice": normalized_voice,
        "model": normalized_model,
    }
    if speed is not None:  # Pythonic check for None
        key_components["speed"] = round(
            speed, 3
        )  # Round to avoid floating point inconsistencies
    if instructions is not None and instructions.strip():
        key_components["instructions"] = instructions.strip()

    # Serialize the dictionary to a JSON string with sorted keys for consistency
    # This handles different orders of optional parameters correctly.
    serialized_components = json.dumps(
        key_components, sort_keys=True, ensure_ascii=False
    )

    hasher = hashlib.sha256()
    hasher.update(serialized_components.encode("utf-8"))
    return hasher.hexdigest()


# --- Cache Storage and Retrieval ---
# Cache layout: cache_base_dir/model_subdir/first_two_chars_hash/full_hash.mp3
# e.g., .cache/tts/tts-1-hd/ab/abcdef12345....mp3


def _get_cache_file_path(
    cache_base_dir: str, model_subdir_name: str, cache_key: str
) -> pathlib.Path:
    # Sanitize model_subdir_name to be a valid directory name
    safe_model_subdir = "".join(c if c.isalnum() else "_" for c in model_subdir_name)

    hash_prefix = cache_key[:2]
    filename = f"{cache_key}.mp3"
    return pathlib.Path(cache_base_dir) / safe_model_subdir / hash_prefix / filename


def get_cached_audio(
    cache_key: str, model_for_subdir: str, cache_base_dir: str
) -> Optional[str]:
    """
    Retrieves an audio file from the cache.
    `model_for_subdir` determines the subdirectory (e.g., from tts_global_model).
    """
    target_path = _get_cache_file_path(cache_base_dir, model_for_subdir, cache_key)

    if (
        target_path.exists()
        and target_path.is_file()
        and target_path.stat().st_size > 0
    ):
        # Optionally, update access time for LRU
        # os.utime(target_path, None) # or target_path.touch() if exist_ok=True
        target_path.touch(exist_ok=True)  # Update access and modification time
        return str(target_path)
    return None


def store_audio_to_cache(
    cache_key: str,
    model_for_subdir: str,  # Used to determine the subdirectory
    audio_file_path: str,  # Path of the file to be copied into cache
    cache_base_dir: str,
) -> Optional[str]:
    """
    Stores a copy of the audio file into the cache.
    `model_for_subdir` is used to determine the subdirectory.
    """
    if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
        print(
            f"Error: Source audio file for caching is invalid or empty: "
            f"{audio_file_path}"
        )
        return None

    target_cache_path = _get_cache_file_path(
        cache_base_dir, model_for_subdir, cache_key
    )

    try:
        # Ensure the target directory structure exists
        target_cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file to the cache location
        shutil.copy2(
            audio_file_path, target_cache_path
        )  # copy2 preserves metadata like timestamps

        if target_cache_path.exists() and target_cache_path.stat().st_size > 0:
            return str(target_cache_path)
        print(f"Error: Failed to verify cached file after copying: {target_cache_path}")
        if target_cache_path.exists():  # If it exists but is empty
            target_cache_path.unlink(missing_ok=True)
        return None
    except OSError as e:
        print(f"Error storing audio to cache at {target_cache_path}: {e}")
        if target_cache_path.exists():  # Attempt cleanup if copy failed midway
            target_cache_path.unlink(missing_ok=True)
        return None


# --- Cache Management (GC) ---
def _get_directory_size(directory_path: pathlib.Path) -> int:
    total_size = 0
    for item in directory_path.rglob("*"):  # Recursively glob all files
        if item.is_file():
            total_size += item.stat().st_size
    return total_size


def manage_cache_size(
    cache_base_dir: str,
    max_cache_size_gb: float = 2.0,
    target_usage_percentage_after_gc: float = 0.75,  # e.g. 0.75 for 75%
) -> None:
    """
    Manages cache size by deleting oldest files if max size is exceeded.
    Uses Last Recently Used (LRU) based on file access times.
    """
    cache_path_obj = pathlib.Path(cache_base_dir)
    if not cache_path_obj.exists() or not cache_path_obj.is_dir():
        print(f"Cache directory '{cache_base_dir}' does not exist. No GC performed.")
        return

    max_cache_size_bytes = int(max_cache_size_gb * (1024**3))
    target_cache_size_bytes_after_gc = int(
        max_cache_size_bytes * target_usage_percentage_after_gc
    )

    current_cache_size_bytes = _get_directory_size(cache_path_obj)

    if current_cache_size_bytes <= max_cache_size_bytes:
        print(
            f"Cache size ({current_cache_size_bytes / (1024**3):.2f} GB) is "
            f"within limit ({max_cache_size_gb:.2f} GB). No GC needed."
        )
        return

    print(
        f"Cache size ({current_cache_size_bytes / (1024**3):.2f} GB) "
        f"exceeds limit ({max_cache_size_gb:.2f} GB). Starting GC..."
    )

    # Collect all cache files with their access times and sizes
    # Files are expected to be .mp3 directly under .../hash_prefix/ directories
    all_cache_files: list[
        tuple[pathlib.Path, float, int]
    ] = []  # (path, access_time, size)

    for item in cache_path_obj.rglob("*.mp3"):  # Iterate through all .mp3 files
        if item.is_file():
            try:
                stats = item.stat()
                all_cache_files.append((item, stats.st_atime, stats.st_size))
            except (
                FileNotFoundError
            ):  # File might be deleted by another process during iteration
                continue

    # Sort files by access time (oldest first) for LRU
    all_cache_files.sort(key=lambda x: x[1])

    bytes_deleted = 0
    files_deleted_count = 0

    for file_path, _access_time, file_size in all_cache_files:
        if current_cache_size_bytes <= target_cache_size_bytes_after_gc:
            break

        try:
            file_path.unlink()  # Delete the file
            current_cache_size_bytes -= file_size
            bytes_deleted += file_size
            files_deleted_count += 1
            print(f"GC: Deleted {file_path} (size: {file_size} bytes)")
        except FileNotFoundError:
            # File might have been deleted by another concurrent GC process or manually
            print(f"GC: File {file_path} not found for deletion (already deleted?).")
            continue  # Don't decrement current_cache_size_bytes if it was already gone
        except OSError as e:  # More specific exception for file operations
            print(f"GC: Error deleting file {file_path}: {e}")
            continue  # Try next file

    print(
        f"GC finished. Deleted {files_deleted_count} files, "
        f"freed {bytes_deleted / (1024**2):.2f} MB. "
        f"New cache size: {current_cache_size_bytes / (1024**3):.2f} GB."
    )

    _cleanup_empty_cache_directories(cache_path_obj)


def _cleanup_empty_cache_directories(cache_path_obj: pathlib.Path) -> None:
    """
    Removes empty subdirectories within the cache structure.
    Iterates through model subdirectories and their prefix subdirectories.
    """
    # Clean up empty subdirectories (hash_prefix and model_subdir) after deleting files
    for model_subdir in cache_path_obj.iterdir():
        if model_subdir.is_dir():
            # Clean up prefix subdirectories first
            for prefix_subdir in model_subdir.iterdir():
                if prefix_subdir.is_dir() and not any(prefix_subdir.iterdir()):
                    try:
                        prefix_subdir.rmdir()
                        print(f"GC: Removed empty prefix directory: {prefix_subdir}")
                    except OSError as e:
                        print(
                            f"GC: Error removing empty prefix directory "
                            f"{prefix_subdir}: {e}"
                        )
            # Then check if the model subdirectory itself is empty
            if not any(model_subdir.iterdir()):  # If model subdir is now empty
                try:
                    model_subdir.rmdir()
                    print(f"GC: Removed empty model directory: {model_subdir}")
                except OSError as e:
                    print(
                        f"GC: Error removing empty model directory {model_subdir}: {e}"
                    )
