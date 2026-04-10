"""Unit tests for modules.session_state — inner-loop TDD.

Each test drives one behavior of the SQLite-backed session persistence.
Tests use a real SQLite database in tmp_path (managed dependency — no mocks).

Tests are ordered by TDD cycle:
1. save_state() creates the database and table on first call
2. save_state() then load_state() round-trips data correctly
3. load_state() returns None for unknown base model
4. save_state() strips seed=-1 from persisted state
5. save_state() upserts — second save for same model replaces first
6. load_state() returns None gracefully on corrupted JSON
"""

import sqlite3

import modules.session_state as session_state_module
import pytest
from modules.session_state import load_state, save_state


@pytest.fixture(autouse=True)
def _isolated_database(tmp_path, monkeypatch):
    """Give each test a fresh SQLite database with no cross-test contamination."""
    db_path = str(tmp_path / "test_session_states.db")
    monkeypatch.setattr(session_state_module, "_db_path", db_path)
    monkeypatch.setattr(session_state_module, "_connection", None)
    yield
    # Close the connection after the test to release the file handle
    conn = session_state_module._connection
    if conn is not None:
        conn.close()


class TestTableCreation:
    """Cycle 1: save_state() creates the database and table on first call."""

    def test_database_file_created_on_first_save(self, tmp_path):
        db_path = tmp_path / "test_session_states.db"
        save_state("pony", {"prompt": "a horse"})
        assert db_path.exists()

    def test_session_states_table_exists_after_save(self, tmp_path):
        db_path = str(tmp_path / "test_session_states.db")
        save_state("pony", {"prompt": "a horse"})
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='session_states'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_table_has_expected_columns(self, tmp_path):
        db_path = str(tmp_path / "test_session_states.db")
        save_state("pony", {"prompt": "a horse"})
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA table_info(session_states)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        assert columns == {"base_model", "state_json", "updated_at"}


class TestSaveLoadRoundTrip:
    """Cycle 2: save_state() then load_state() round-trips data correctly."""

    def test_simple_state_round_trips(self):
        state = {"prompt": "a cat", "cfg_scale": 7.5, "steps": 30}
        save_state("sdxl", state)
        loaded = load_state("sdxl")
        assert loaded is not None
        assert loaded["prompt"] == "a cat"
        assert loaded["cfg_scale"] == pytest.approx(7.5)
        assert loaded["steps"] == 30

    def test_complex_state_with_nested_data_round_trips(self):
        state = {
            "prompt": "landscape",
            "loras": [{"name": "detail_tweaker", "weight": 0.8}],
            "styles": ["cinematic", "dramatic"],
        }
        save_state("pony", state)
        loaded = load_state("pony")
        assert loaded is not None
        assert loaded["loras"] == [{"name": "detail_tweaker", "weight": 0.8}]
        assert loaded["styles"] == ["cinematic", "dramatic"]

    def test_different_base_models_stored_independently(self):
        save_state("sdxl", {"prompt": "sdxl prompt"})
        save_state("pony", {"prompt": "pony prompt"})
        assert load_state("sdxl")["prompt"] == "sdxl prompt"
        assert load_state("pony")["prompt"] == "pony prompt"


class TestLoadMissingModel:
    """Cycle 3: load_state() returns None for unknown base model."""

    def test_returns_none_for_unknown_model(self):
        result = load_state("nonexistent_model")
        assert result is None

    def test_returns_none_when_database_is_empty(self):
        # Force table creation by saving then check a different model
        save_state("sdxl", {"prompt": "exists"})
        result = load_state("pony")
        assert result is None


class TestSeedStripping:
    """Cycle 4: save_state() strips seed=-1 from persisted state."""

    def test_seed_minus_one_is_stripped(self):
        state = {"prompt": "a dog", "seed": -1, "steps": 20}
        save_state("sdxl", state)
        loaded = load_state("sdxl")
        assert loaded is not None
        assert "seed" not in loaded
        assert loaded["prompt"] == "a dog"
        assert loaded["steps"] == 20

    def test_non_negative_one_seed_is_preserved(self):
        state = {"prompt": "a dog", "seed": 42}
        save_state("sdxl", state)
        loaded = load_state("sdxl")
        assert loaded is not None
        assert loaded["seed"] == 42

    def test_seed_zero_is_preserved(self):
        state = {"prompt": "a dog", "seed": 0}
        save_state("sdxl", state)
        loaded = load_state("sdxl")
        assert loaded is not None
        assert loaded["seed"] == 0

    def test_original_state_dict_is_not_mutated(self):
        state = {"prompt": "a dog", "seed": -1}
        save_state("sdxl", state)
        assert state["seed"] == -1, "save_state must not mutate the caller's dict"


class TestUpsertBehavior:
    """Cycle 5: save_state() upserts — second save for same model replaces first."""

    def test_second_save_replaces_first(self):
        save_state("sdxl", {"prompt": "first"})
        save_state("sdxl", {"prompt": "second"})
        loaded = load_state("sdxl")
        assert loaded is not None
        assert loaded["prompt"] == "second"

    def test_upsert_does_not_create_duplicate_rows(self, tmp_path):
        db_path = str(tmp_path / "test_session_states.db")
        save_state("sdxl", {"prompt": "first"})
        save_state("sdxl", {"prompt": "second"})
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM session_states WHERE base_model = 'sdxl'")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1


class TestCorruptedJsonHandling:
    """Cycle 6: load_state() returns None gracefully on corrupted JSON."""

    def test_returns_none_on_corrupted_json(self, tmp_path):
        db_path = str(tmp_path / "test_session_states.db")
        # First save a valid state to create the table
        save_state("sdxl", {"prompt": "valid"})
        # Now inject corrupted JSON directly into the database
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE session_states SET state_json = ? WHERE base_model = ?",
            ("{not valid json!!!", "sdxl"),
        )
        conn.commit()
        conn.close()
        # load_state reads via the module's own connection, which has cached data
        # We need to use a fresh connection to see the corrupted data
        # Reset the module connection so it reopens and sees the corrupted row
        conn_to_close = session_state_module._connection
        if conn_to_close is not None:
            conn_to_close.close()
        session_state_module._connection = None
        result = load_state("sdxl")
        assert result is None
