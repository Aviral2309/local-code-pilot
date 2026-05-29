"""
tests/test_telemetry.py
-----------------------
Tests for the SQLite telemetry logger.

Uses an in-memory SQLite database (:memory: path won't work across
connections, so we use a temp file per test).
"""

import os
import tempfile
import time
import pytest
from src.telemetry import TelemetryLogger, CompletionEvent


@pytest.fixture
def tmp_logger():
    """Create a TelemetryLogger backed by a temp file, clean up after test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    logger = TelemetryLogger(db_path=db_path)
    yield logger

    # Cleanup
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


def make_event(**kwargs) -> CompletionEvent:
    defaults = dict(
        file_path="file:///home/user/project/main.py",
        language_id="python",
        context_used=["chunk_001", "chunk_002"],
        ttft_ms=85.3,
        total_ms=210.7,
        completion_text="    return np.mean(data)",
        completion_length=24,
    )
    defaults.update(kwargs)
    return CompletionEvent(**defaults)


class TestTelemetryLogger:

    def test_log_completion_returns_row_id(self, tmp_logger):
        """log_completion returns an integer row ID."""
        event = make_event()
        row_id = tmp_logger.log_completion(event)
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_row_ids_are_sequential(self, tmp_logger):
        """Each new completion gets a higher row ID."""
        id1 = tmp_logger.log_completion(make_event())
        id2 = tmp_logger.log_completion(make_event())
        id3 = tmp_logger.log_completion(make_event())
        assert id1 < id2 < id3

    def test_mark_accepted(self, tmp_logger):
        """Marking as accepted stores accepted=1 in the DB."""
        row_id = tmp_logger.log_completion(make_event())
        tmp_logger.mark_accepted(row_id)

        import sqlite3
        conn = sqlite3.connect(str(tmp_logger.db_path))
        row = conn.execute(
            "SELECT accepted FROM completions WHERE id = ?", (row_id,)
        ).fetchone()
        conn.close()
        assert row[0] == 1

    def test_mark_rejected(self, tmp_logger):
        """Marking as rejected stores accepted=0 in the DB."""
        row_id = tmp_logger.log_completion(make_event())
        tmp_logger.mark_rejected(row_id)

        import sqlite3
        conn = sqlite3.connect(str(tmp_logger.db_path))
        row = conn.execute(
            "SELECT accepted FROM completions WHERE id = ?", (row_id,)
        ).fetchone()
        conn.close()
        assert row[0] == 0

    def test_initial_accepted_is_null(self, tmp_logger):
        """Freshly logged completion has NULL accepted (unknown outcome)."""
        row_id = tmp_logger.log_completion(make_event())

        import sqlite3
        conn = sqlite3.connect(str(tmp_logger.db_path))
        row = conn.execute(
            "SELECT accepted FROM completions WHERE id = ?", (row_id,)
        ).fetchone()
        conn.close()
        assert row[0] is None

    def test_get_metrics_empty_db(self, tmp_logger):
        """Metrics on empty DB returns zeros/Nones without error."""
        metrics = tmp_logger.get_metrics()
        assert metrics["total_completions"] == 0
        assert metrics["acceptance_rate_pct"] is None
        assert metrics["avg_ttft_ms"] == 0.0

    def test_get_metrics_with_data(self, tmp_logger):
        """Metrics compute correctly with known data."""
        # Log 3 completions: 2 accepted, 1 rejected
        id1 = tmp_logger.log_completion(make_event(ttft_ms=80.0, total_ms=200.0))
        id2 = tmp_logger.log_completion(make_event(ttft_ms=90.0, total_ms=220.0))
        id3 = tmp_logger.log_completion(make_event(ttft_ms=100.0, total_ms=300.0))

        tmp_logger.mark_accepted(id1)
        tmp_logger.mark_accepted(id2)
        tmp_logger.mark_rejected(id3)

        metrics = tmp_logger.get_metrics()

        assert metrics["total_completions"] == 3
        assert metrics["accepted"] == 2
        assert metrics["rejected"] == 1
        assert metrics["outcome_unknown"] == 0

        # Acceptance rate: 2/3 = 66.7%
        assert abs(metrics["acceptance_rate_pct"] - 66.7) < 0.5

        # Avg TTFT: (80 + 90 + 100) / 3 = 90
        assert abs(metrics["avg_ttft_ms"] - 90.0) < 1.0

    def test_get_metrics_all_unknown(self, tmp_logger):
        """If no outcomes are logged, acceptance_rate_pct is None."""
        tmp_logger.log_completion(make_event())
        tmp_logger.log_completion(make_event())

        metrics = tmp_logger.get_metrics()
        assert metrics["acceptance_rate_pct"] is None
        assert metrics["outcome_unknown"] == 2

    def test_db_persists_across_instances(self, tmp_logger):
        """Data written by one TelemetryLogger is readable by another on same DB."""
        row_id = tmp_logger.log_completion(make_event())
        tmp_logger.mark_accepted(row_id)

        # Create a new instance pointing to the same DB
        logger2 = TelemetryLogger(db_path=str(tmp_logger.db_path))
        metrics = logger2.get_metrics()

        assert metrics["total_completions"] == 1
        assert metrics["accepted"] == 1

    def test_latency_percentiles_present(self, tmp_logger):
        """p50 and p95 latency values are computed when data exists."""
        for i in range(10):
            tmp_logger.log_completion(make_event(total_ms=float(100 + i * 10)))

        metrics = tmp_logger.get_metrics()
        assert metrics["p50_total_ms"] is not None
        assert metrics["p95_total_ms"] is not None
        assert metrics["p95_total_ms"] >= metrics["p50_total_ms"]
