"""
telemetry.py
------------
Local SQLite telemetry logger.

Logs every completion event with:
- What context was injected
- Time-to-first-token and total roundtrip latency
- Whether the user accepted (Tab) or rejected the completion

This data powers two things:
1. The /metrics FastAPI endpoint (your "ops dashboard")
2. Future: logistic regression to learn which retrieval strategy
   leads to higher acceptance rates for a specific codebase.

Interview talking point: This is a real ML product feedback loop.
Most tools have zero observability on completion quality. By logging
accept/reject with the exact context used, you can run a retrospective
analysis: "chunks from call-graph context are accepted 23% more often
than chunks from raw semantic search alone."

Schema:
    completions(id, timestamp, file_path, language_id, context_used,
                ttft_ms, total_ms, accepted, completion_length)
"""

import asyncio
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)


@dataclass
class CompletionEvent:
    """
    A single completion event to be logged.

    Fields populated at completion time:
        file_path, language_id, context_used, ttft_ms, total_ms,
        completion_text, completion_length

    Fields populated later (when user accepts/rejects):
        accepted (default None = unknown / timed out)
    """
    file_path: str
    language_id: str
    context_used: list          # List of chunk IDs/summaries that were injected
    ttft_ms: float              # Time-to-first-token in milliseconds
    total_ms: float             # Total roundtrip latency in milliseconds
    completion_text: str        # What we generated
    completion_length: int      # Character count of completion
    accepted: Optional[bool] = None   # None = unknown, True = accepted, False = rejected
    timestamp: float = field(default_factory=time.time)
    row_id: Optional[int] = None      # Set after DB insert


class TelemetryLogger:
    """
    Thread-safe SQLite telemetry logger.

    Uses WAL (Write-Ahead Logging) mode for concurrent reads without
    blocking writes — important because the /metrics endpoint reads
    while the LSP server writes.
    """

    def __init__(self, db_path: str = "auralsp_telemetry.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist. Safe to call multiple times."""
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # Concurrent reads
            conn.execute("PRAGMA synchronous=NORMAL") # Balance safety/speed
            conn.execute("""
                CREATE TABLE IF NOT EXISTS completions (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp        REAL    NOT NULL,
                    file_path        TEXT    NOT NULL,
                    language_id      TEXT    NOT NULL,
                    context_used     TEXT    NOT NULL,  -- JSON array
                    ttft_ms          REAL    NOT NULL,
                    total_ms         REAL    NOT NULL,
                    completion_text  TEXT    NOT NULL,
                    completion_length INTEGER NOT NULL,
                    accepted         INTEGER             -- 0=rejected, 1=accepted, NULL=unknown
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_completions_timestamp
                ON completions (timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_completions_file
                ON completions (file_path)
            """)
        logger.info(f"Telemetry DB initialized at {self.db_path}")

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for SQLite connections with auto-commit."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def log_completion(self, event: CompletionEvent) -> int:
        """
        Insert a new completion event. Returns the row ID.

        Args:
            event: CompletionEvent to log.

        Returns:
            SQLite row ID (used later to update accepted/rejected status).
        """
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO completions
                    (timestamp, file_path, language_id, context_used,
                     ttft_ms, total_ms, completion_text, completion_length, accepted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.timestamp,
                event.file_path,
                event.language_id,
                json.dumps(event.context_used),
                event.ttft_ms,
                event.total_ms,
                event.completion_text,
                event.completion_length,
                None,  # accepted unknown until user acts
            ))
            row_id = cursor.lastrowid

        logger.debug(f"Logged completion event row_id={row_id} | {event.file_path}")
        return row_id

    def mark_accepted(self, row_id: int) -> None:
        """Mark a completion as accepted (user pressed Tab)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE completions SET accepted = 1 WHERE id = ?",
                (row_id,)
            )
        logger.debug(f"Completion {row_id} marked ACCEPTED")

    def mark_rejected(self, row_id: int) -> None:
        """Mark a completion as rejected (user ignored or typed over it)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE completions SET accepted = 0 WHERE id = ?",
                (row_id,)
            )
        logger.debug(f"Completion {row_id} marked REJECTED")

    def get_metrics(self) -> dict:
        """
        Compute aggregate metrics for the /metrics endpoint.

        Returns:
            Dict with acceptance rate, avg latency, total completions, etc.
        """
        with self._connect() as conn:
            # Overall stats
            row = conn.execute("""
                SELECT
                    COUNT(*)                                         AS total,
                    AVG(ttft_ms)                                     AS avg_ttft_ms,
                    AVG(total_ms)                                    AS avg_total_ms,
                    AVG(completion_length)                           AS avg_length,
                    SUM(CASE WHEN accepted = 1 THEN 1 ELSE 0 END)   AS accepted_count,
                    SUM(CASE WHEN accepted = 0 THEN 1 ELSE 0 END)   AS rejected_count,
                    SUM(CASE WHEN accepted IS NULL THEN 1 ELSE 0 END) AS unknown_count
                FROM completions
            """).fetchone()

            total = row["total"] or 0
            accepted = row["accepted_count"] or 0
            rejected = row["rejected_count"] or 0
            decided = accepted + rejected

            acceptance_rate = (accepted / decided * 100) if decided > 0 else None

            # Recent 24h
            cutoff = time.time() - 86400
            recent = conn.execute("""
                SELECT COUNT(*) AS count
                FROM completions
                WHERE timestamp > ?
            """, (cutoff,)).fetchone()

            # Latency percentiles (simple approximation)
            latencies = [
                r[0] for r in conn.execute(
                    "SELECT total_ms FROM completions ORDER BY total_ms"
                ).fetchall()
            ]

            p50 = _percentile(latencies, 50) if latencies else None
            p95 = _percentile(latencies, 95) if latencies else None

        return {
            "total_completions": total,
            "completions_last_24h": recent["count"],
            "acceptance_rate_pct": round(acceptance_rate, 1) if acceptance_rate else None,
            "accepted": accepted,
            "rejected": rejected,
            "outcome_unknown": row["unknown_count"] or 0,
            "avg_ttft_ms": round(row["avg_ttft_ms"] or 0, 1),
            "avg_total_ms": round(row["avg_total_ms"] or 0, 1),
            "p50_total_ms": round(p50, 1) if p50 else None,
            "p95_total_ms": round(p95, 1) if p95 else None,
            "avg_completion_length": round(row["avg_length"] or 0, 1),
        }


def _percentile(sorted_values: list, pct: int) -> Optional[float]:
    """Compute a percentile from a sorted list."""
    if not sorted_values:
        return None
    idx = int(len(sorted_values) * pct / 100)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]
