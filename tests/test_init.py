"""Regression tests for concurrent first-touch schema initialisation.

Docker Compose starts two uvicorn worker processes that share one SQLite file.
On a *fresh* data volume both run lifespan's ``init_db`` at the same instant;
the WAL switch and DDL must be serialised/retried so both workers finish with
the same schema, instead of one dying with "database is locked".

The cross-process init lock is taken **non-blocking**: a peer that holds
``<db>.init.lock`` for the whole retry budget must make the waiter raise
``SchemaUnavailable`` and end startup (never hang on ``flock``), while a lock
released inside the budget lets the waiter proceed and verify the completed
schema.
"""
import dataclasses
import multiprocessing
import os
import sqlite3
import threading
import time

import pytest

from app import config
from app import db

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows etc.: the lock is a no-op there
    _fcntl = None

_requires_fcntl = pytest.mark.skipif(_fcntl is None, reason="requires fcntl.flock")


def _patch_settings(monkeypatch, tmp_path, **overrides):
    """Swap in a frozen Settings copy (Settings is a frozen dataclass)."""
    overrides.setdefault("db_path", str(tmp_path / "tiles.db"))
    new_settings = dataclasses.replace(config.settings, **overrides)
    monkeypatch.setattr(db, "settings", new_settings)
    return new_settings


def _open(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def test_init_db_is_idempotent_and_enables_wal(tmp_path, monkeypatch):
    nested = tmp_path / "nested" / "tiles.db"
    new_settings = dataclasses.replace(config.settings, db_path=str(nested))
    monkeypatch.setattr(db, "settings", new_settings)
    db.init_db()
    db.init_db()  # second worker / restart must not error or duplicate

    conn = _open(str(nested))
    try:
        assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
        tables = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"datasets", "revisions", "tile_cache"} <= tables
        # No duplicate objects of any kind survived.
        names = [r["name"] for r in conn.execute("SELECT name FROM sqlite_master")]
        assert len(names) == len(set(names))
    finally:
        conn.close()


def test_init_db_retries_busy_then_succeeds(tmp_path, monkeypatch):
    _patch_settings(
        monkeypatch,
        tmp_path,
        init_max_attempts=3,
        init_retry_delay_s=0.0,
    )

    original_connect = db._connect
    calls = {"n": 0}

    def flaky_connect():
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return original_connect()

    monkeypatch.setattr(db, "_connect", flaky_connect)
    db.init_db()
    assert calls["n"] >= 2


def test_init_db_retries_when_schema_not_yet_visible(tmp_path, monkeypatch):
    _patch_settings(
        monkeypatch,
        tmp_path,
        init_max_attempts=3,
        init_retry_delay_s=0.0,
    )

    real_assert = db._assert_db_ready
    calls = {"n": 0}

    def flaky_assert(conn):
        calls["n"] += 1
        if calls["n"] == 1:
            raise db.SchemaUnavailable(
                "schema initialisation incomplete, missing: ['datasets']"
            )
        return real_assert(conn)

    monkeypatch.setattr(db, "_assert_db_ready", flaky_assert)
    db.init_db()
    assert calls["n"] == 2


def test_init_db_gives_up_after_bounded_attempts(tmp_path, monkeypatch):
    _patch_settings(
        monkeypatch,
        tmp_path,
        init_max_attempts=3,
        init_retry_delay_s=0.0,
    )

    calls = {"n": 0}

    def always_busy():
        calls["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "_connect", always_busy)
    with pytest.raises(db.SchemaUnavailable):
        db.init_db()
    assert calls["n"] == 3


def test_init_db_non_busy_error_is_not_retried(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, tmp_path)

    calls = {"n": 0}

    def broken_connect():
        calls["n"] += 1
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(db, "_connect", broken_connect)
    with pytest.raises(sqlite3.OperationalError):
        db.init_db()
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Cross-process: mirrors the real two-worker Compose first-touch race.
# ---------------------------------------------------------------------------


def _init_worker(db_path, ready, start, result):
    os.environ["VT_DB_PATH"] = db_path
    # Reload configuration so the env var is picked up in the spawned process.
    import importlib

    from app import config as config_mod
    from app import db as db_mod

    importlib.reload(config_mod)
    importlib.reload(db_mod)
    ready.wait()
    start.wait()
    try:
        db_mod.init_db()
        with db_mod.connection() as conn:
            mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            ]
        result.put(("ok", mode, sorted(tables)))
    except Exception as exc:  # pragma: no cover - surfaces the raw failure
        result.put(("error", f"{type(exc).__name__}: {exc}", []))


@pytest.mark.skipif(
    "spawn" not in multiprocessing.get_all_start_methods(),
    reason="cross-process test requires the 'spawn' start method",
)
def test_two_processes_initialise_fresh_database_concurrently(tmp_path):
    db_path = str(tmp_path / "tiles.db")
    ctx = multiprocessing.get_context("spawn")
    # 3-party barriers: each child signals readiness / waits for release
    # together with the parent, so both workers call init_db at the same time.
    ready = ctx.Barrier(3)
    start = ctx.Barrier(3)
    result = ctx.Queue()

    procs = [
        ctx.Process(target=_init_worker, args=(db_path, ready, start, result))
        for _ in range(2)
    ]
    for proc in procs:
        proc.start()
    ready.wait(timeout=30)
    start.wait(timeout=30)
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0, f"worker exited with {proc.exitcode}"

    outcomes = [result.get_nowait() for _ in range(2)]
    assert all(outcome[0] == "ok" for outcome in outcomes), outcomes
    assert all(outcome[1] == "wal" for outcome in outcomes), outcomes
    for _, _, tables in outcomes:
        assert {"datasets", "revisions", "tile_cache"} <= set(tables)

    # The shared file is usable afterwards and contains no duplicate objects.
    conn = _open(db_path)
    try:
        rows = conn.execute("SELECT name, COUNT(*) FROM sqlite_master GROUP BY name")
        assert all(count == 1 for _, count in rows)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Non-blocking init lock: held lock must be retried within the shared budget,
# never block lifespan on an uninterruptible flock.
# ---------------------------------------------------------------------------


def _hold_init_lock(db_path: str):
    """Take a blocking exclusive flock on ``<db>.init.lock`` in this process."""
    fh = open(db_path + ".init.lock", "a+")
    _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
    return fh


@_requires_fcntl
def test_init_lock_is_non_blocking_and_retries_until_released(
    tmp_path, monkeypatch
):
    settings = _patch_settings(
        monkeypatch,
        tmp_path,
        init_max_attempts=50,
        init_retry_delay_s=0.01,
    )

    holder_fh = _hold_init_lock(settings.db_path)
    released = threading.Event()

    def release_soon():
        time.sleep(0.15)
        _fcntl.flock(holder_fh.fileno(), _fcntl.LOCK_UN)
        holder_fh.close()
        released.set()

    timer = threading.Timer(0.15, release_soon)
    start = time.monotonic()
    timer.start()
    try:
        # Must not hang: every attempt is LOCK_NB, and once the holder releases
        # inside the retry budget the worker proceeds and builds the schema.
        db.init_db()
    finally:
        timer.cancel()
        released.wait(timeout=5)
    assert time.monotonic() - start >= 0.1

    conn = _open(settings.db_path)
    try:
        assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
        tables = {
            r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"datasets", "revisions", "tile_cache"} <= tables
    finally:
        conn.close()


@_requires_fcntl
def test_persistently_held_init_lock_raises_schema_unavailable_without_hanging(
    tmp_path, monkeypatch
):
    settings = _patch_settings(
        monkeypatch,
        tmp_path,
        init_max_attempts=3,
        init_retry_delay_s=0.0,
    )

    sleeps: list[float] = []
    monkeypatch.setattr(db.time, "sleep", lambda delay: sleeps.append(delay))

    holder_fh = _hold_init_lock(settings.db_path)
    start = time.monotonic()
    try:
        with pytest.raises(db.SchemaUnavailable) as excinfo:
            db.init_db()
        # Startup ends promptly (no blocking flock) and names the held lock.
        assert time.monotonic() - start < 5
        assert "init lock held by another process" in str(excinfo.value)
        # Exactly the bounded budget; one backoff between each pair.
        assert len(sleeps) == 2
        # No database work happened while the lock was unavailable.
        assert not os.path.exists(settings.db_path)
    finally:
        _fcntl.flock(holder_fh.fileno(), _fcntl.LOCK_UN)
        holder_fh.close()

    # After the holder exits the lock is free and init succeeds immediately.
    db.init_db()
    conn = _open(settings.db_path)
    try:
        assert {"datasets", "revisions", "tile_cache"} <= {
            r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        conn.close()


@_requires_fcntl
def test_init_lock_is_released_when_schema_work_raises(tmp_path, monkeypatch):
    settings = _patch_settings(
        monkeypatch,
        tmp_path,
        init_max_attempts=2,
        init_retry_delay_s=0.0,
    )

    original_connect = db._connect

    def fail_then_succeed():
        fail_then_succeed.calls += 1
        if fail_then_succeed.calls == 1:
            raise sqlite3.OperationalError("database is locked")
        return original_connect()

    fail_then_succeed.calls = 0
    monkeypatch.setattr(db, "_connect", fail_then_succeed)

    db.init_db()
    # First attempt (failed connect) + second attempt (connect + post-lock
    # verification connection).
    assert fail_then_succeed.calls == 3

    # The failed first attempt released the flock: acquiring it non-blockingly
    # in a fresh file handle works right away.
    with open(settings.db_path + ".init.lock", "a+") as fh:
        _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)


def _external_lock_holder(db_path, hold_for_s, acquired, done):
    """Spawned-process helper: hold the init lock, then exit and release it."""
    import time as _time

    with open(db_path + ".init.lock", "a+") as fh:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        acquired.set()
        deadline = _time.monotonic() + hold_for_s
        while not done.is_set() and _time.monotonic() < deadline:
            _time.sleep(0.05)
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@_requires_fcntl
@pytest.mark.skipif(
    "spawn" not in multiprocessing.get_all_start_methods(),
    reason="cross-process test requires the 'spawn' start method",
)
def test_external_holder_blocks_then_unblocks_startup(tmp_path, monkeypatch):
    from starlette.testclient import TestClient

    from app.main import app

    db_path = str(tmp_path / "tiles.db")
    ctx = multiprocessing.get_context("spawn")
    acquired = ctx.Event()
    done = ctx.Event()
    holder = ctx.Process(
        target=_external_lock_holder,
        args=(db_path, 20.0, acquired, done),
    )
    holder.start()
    assert acquired.wait(timeout=10)

    _patch_settings(
        monkeypatch,
        tmp_path,
        init_max_attempts=4,
        init_retry_delay_s=0.05,
    )

    try:
        # Persistent external holder -> bounded failure, not a hang, and no
        # schema work ever reached the database file.
        start = time.monotonic()
        with pytest.raises(db.SchemaUnavailable):
            db.init_db()
        assert time.monotonic() - start < 10
        assert not os.path.exists(db_path)

        # Holder releases within the budget -> startup completes.  A second
        # init (as lifespan would run) sees the complete schema and /healthz
        # serves 200; the HTTP API stays callable.
        done.set()
        holder.join(timeout=10)
        assert holder.exitcode == 0
        db.init_db()

        with TestClient(app) as client:
            assert client.get("/healthz").status_code == 200
            resp = client.get("/datasets")
            assert resp.status_code == 200
            assert resp.json() == {"datasets": []}
    finally:
        done.set()
        holder.join(timeout=10)
