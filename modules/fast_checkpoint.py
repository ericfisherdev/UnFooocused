"""Fast checkpoint storage module.

Caches checkpoint files on fast local storage (NVMe/SSD) for quicker
loading. On first use, copies a checkpoint atomically from slow storage
(NAS/HDD) to fast storage. Subsequent loads use the cached copy.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import time

logger = logging.getLogger(__name__)


def _find_in_folders(name: str, folders: list[str]) -> str:
    """Search folders for a file by name, returning the first match.

    Returns the absolute real path of the first match, or a constructed
    path in the first folder when no match is found.
    """
    for folder in folders:
        candidate = os.path.abspath(os.path.realpath(os.path.join(folder, name)))
        if os.path.isfile(candidate):
            return candidate

    return os.path.abspath(os.path.realpath(os.path.join(folders[0], name)))


def resolve_checkpoint_path(
    checkpoint_name: str,
    checkpoint_folders: list[str],
    fast_path: str | None = None,
) -> str:
    """Resolve the path for a checkpoint, caching on fast storage if configured.

    Args:
        checkpoint_name: Checkpoint filename (e.g. ``model.safetensors``).
        checkpoint_folders: Directories to search for the original checkpoint.
        fast_path: Path to the fast cache directory, or ``None``/empty to disable.

    Returns:
        Absolute path to the checkpoint file -- from fast storage when
        available, otherwise from the original location.
    """
    if not fast_path:
        return _find_in_folders(checkpoint_name, checkpoint_folders)

    if not _is_safe_checkpoint_name(checkpoint_name):
        logger.warning("Refusing unsafe checkpoint path for fast cache: %s", checkpoint_name)
        return _find_in_folders(checkpoint_name, checkpoint_folders)

    fast_root = os.path.abspath(os.path.realpath(fast_path))
    safe_name = os.path.normpath(checkpoint_name)
    fast_file = os.path.abspath(os.path.realpath(os.path.join(fast_root, safe_name)))

    if os.path.commonpath([fast_root, fast_file]) != fast_root:
        logger.warning("Resolved fast-cache path escapes cache root: %s", checkpoint_name)
        return _find_in_folders(checkpoint_name, checkpoint_folders)

    if os.path.isfile(fast_file):
        return fast_file

    original_path = _find_in_folders(checkpoint_name, checkpoint_folders)

    if not os.path.isfile(original_path):
        return original_path

    return _copy_to_fast_storage(original_path, fast_file)


def _is_safe_checkpoint_name(name: str) -> bool:
    """Return True if the checkpoint name is safe for fast cache use."""
    normalized = os.path.normpath(name)
    if os.path.isabs(normalized):
        return False
    return not (normalized.startswith(".." + os.sep) or normalized == "..")


def _copy_to_fast_storage(source_path: str, dest_path: str) -> str:
    """Copy a checkpoint file atomically to fast storage.

    Writes to a ``.tmp`` file first, then renames to prevent corruption
    from interrupted copies.

    Args:
        source_path: Path to the original checkpoint file.
        dest_path: Target path on fast storage.

    Returns:
        ``dest_path`` on success, ``source_path`` on failure.
    """
    tmp_path = dest_path + ".tmp"
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        file_size_mb = os.path.getsize(source_path) / (1024 * 1024)
        logger.info(
            "Copying checkpoint to fast storage: %s (%.0f MB)",
            os.path.basename(source_path),
            file_size_mb,
        )

        start_time = time.time()
        shutil.copy2(source_path, tmp_path)
        os.rename(tmp_path, dest_path)
        elapsed = time.time() - start_time

        logger.info("Checkpoint cached on fast storage in %.1fs: %s", elapsed, dest_path)
        return dest_path

    except OSError as exc:
        logger.warning(
            "Failed to cache checkpoint on fast storage: %s. Loading from original location.",
            exc,
        )
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        return source_path
