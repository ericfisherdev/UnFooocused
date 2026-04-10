"""
Session State Persistence Module

Stores and retrieves UI state per base model family using SQLite.
State is saved after each successful generation and restored at startup
so the user's prompt, settings, and LoRAs persist between browser sessions.
"""

import json
import logging
import sqlite3
import threading
import time
from typing import NewType, TypedDict, cast

logger = logging.getLogger(__name__)


BaseModelFamily = NewType("BaseModelFamily", str)


class LoraEntry(TypedDict):
    """A single LoRA adapter with its blending weight."""

    filename: str
    weight: float


class SessionState(TypedDict, total=False):
    """Schema for persisted UI state per base model family.

    All fields are optional (total=False) because callers may persist
    a partial subset of settings — e.g. only prompt and cfg_scale.
    """

    prompt: str
    negative_prompt: str
    style_selections: list[str]
    base_model_name: str
    refiner_model_name: str
    vae_name: str
    loras: list[LoraEntry]
    sampler: str
    scheduler: str
    steps: int
    cfg_scale: float
    performance: str
    image_number: int
    sharpness: float
    seed: int
    aspect_ratios_selection: str


_db_path: str = "./session_states.db"
_connection: sqlite3.Connection | None = None
_lock: threading.RLock = threading.RLock()


def _get_connection() -> sqlite3.Connection:
    """
    Get or create the SQLite database connection.

    Creates the database and table on first access.
    Thread-safe via double-checked locking.
    """
    global _connection
    if _connection is None:
        with _lock:
            if _connection is None:
                _connection = sqlite3.connect(_db_path, check_same_thread=False)
                _connection.execute("""
                    CREATE TABLE IF NOT EXISTS session_states (
                        base_model TEXT PRIMARY KEY,
                        state_json TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    )
                """)
                _connection.commit()
    return _connection


def save_state(base_model: BaseModelFamily, state: SessionState) -> None:
    """
    Save UI state for a base model family.

    Upserts the state, replacing any previous state for this base model.
    Strips seed if it equals -1 (random).

    Args:
        base_model: Base model family key (e.g. 'pony', 'sdxl').
        state: Typed session state payload to persist.
    """
    state_copy = dict(state)
    if state_copy.get("seed") == -1:
        state_copy.pop("seed", None)

    try:
        with _lock:
            conn = _get_connection()
            conn.execute(
                """INSERT INTO session_states (base_model, state_json, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(base_model) DO UPDATE SET
                       state_json = excluded.state_json,
                       updated_at = excluded.updated_at""",
                (base_model, json.dumps(state_copy), time.time()),
            )
            conn.commit()
        logger.debug(f"Session state saved for base model: {base_model}")
    except sqlite3.Error as e:
        logger.warning(f"Failed to save session state: {e}")


def load_state(base_model: BaseModelFamily) -> SessionState | None:
    """
    Load saved UI state for a base model family.

    Args:
        base_model: Base model family key (e.g. 'pony', 'sdxl').

    Returns:
        Typed session state, or None if no state exists.
    """
    try:
        conn = _get_connection()
        cursor = conn.execute("SELECT state_json FROM session_states WHERE base_model = ?", (base_model,))
        row = cursor.fetchone()
        if row is None:
            return None
        return cast("SessionState", json.loads(row[0]))
    except (sqlite3.Error, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load session state: {e}")
        return None
