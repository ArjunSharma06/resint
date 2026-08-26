"""Where model answers are kept between runs.

The cache is not a cost optimisation with a nice side effect. It is the
**determinism mechanism** for the whole tier: ``temperature=0`` does not exist
on current models, so run-to-run reproducibility cannot come from sampling
settings, and it has to come from not asking twice.

That makes this the difference between a development loop that is free and one
that is rationed. Iterating on a rule means running it over the same papers
twenty times; without a cache that is twenty times the calls, which is exactly
how the first real session exhausted a free-tier quota re-deriving answers it
already had.

sqlite rather than a directory of files, because the sweep runs papers in
separate processes and several of them will write at once. The stdlib module
handles that; a directory of JSON files handles it until it silently does not.

Keys come from :meth:`Request.cache_key`, which hashes the rendered prompt and
the model name. So editing a prompt invalidates exactly the entries that prompt
produced, with nobody having to remember to bump a version.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

DEFAULT_PATH = Path.home() / ".cache" / "resint" / "model-answers.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    key     TEXT PRIMARY KEY,
    model   TEXT NOT NULL,
    payload TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS answers_model ON answers (model);
"""


class DiskStore:
    """A sqlite table of model answers, keyed by prompt hash."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.executescript(_SCHEMA)

    def _connect(self):
        # check_same_thread=False because the resolver pool and the sweep both
        # touch this from worker threads; the lock above serialises writes.
        db = sqlite3.connect(self.path, timeout=10.0, check_same_thread=False)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def get(self, key: str):
        try:
            with self._connect() as db:
                row = db.execute(
                    "SELECT payload FROM answers WHERE key = ?", (key,)
                ).fetchone()
        except sqlite3.Error:
            return None  # A broken cache is a slow run, never a failed one.
        if not row:
            return None
        try:
            return json.loads(row[0])
        except ValueError:
            return None

    def put(self, key: str, payload, model: str = "") -> None:
        try:
            body = json.dumps(payload)
        except (TypeError, ValueError):
            return
        try:
            with self._lock, self._connect() as db:
                db.execute(
                    "INSERT OR REPLACE INTO answers (key, model, payload, created) "
                    "VALUES (?, ?, ?, ?)",
                    (key, model, body, time.time()),
                )
        except sqlite3.Error:
            pass

    def count(self) -> int:
        try:
            with self._connect() as db:
                return db.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
        except sqlite3.Error:
            return 0

    def export(self, path: str | Path, model: str | None = None) -> int:
        """Write the cache out as a JSON file a StaticProvider can replay.

        This is how a run against a real model becomes a fixture CI can use
        forever with no key and no network.
        """
        query = "SELECT key, payload FROM answers"
        params: tuple = ()
        if model:
            query += " WHERE model = ?"
            params = (model,)

        try:
            with self._connect() as db:
                rows = db.execute(query, params).fetchall()
        except sqlite3.Error:
            return 0

        out = {}
        for key, body in rows:
            try:
                out[key] = json.loads(body)
            except ValueError:
                continue

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")
        return len(out)
