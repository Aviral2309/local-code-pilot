"""
metrics_api.py
--------------
FastAPI server exposing AuraLSP telemetry via HTTP.

Runs as a separate process on localhost:8765 alongside the LSP server.
Reads from the shared SQLite telemetry database (WAL mode = safe concurrent reads).

Endpoints:
    GET /          — health check
    GET /metrics   — aggregate stats (acceptance rate, latency, total completions)
    GET /recent    — last N completion events
    GET /docs      — FastAPI auto-generated Swagger UI (free!)

Usage:
    python -m src.metrics_api
    curl http://localhost:8765/metrics | python -m json.tool

Interview talking point:
    This is your "ops dashboard" — it makes the project observable.
    Acceptance rate over time shows whether your retrieval strategy
    actually helps. Latency percentiles expose bottlenecks.
    The /metrics endpoint is the same pattern used in production
    services (Prometheus, OpenMetrics format).
"""

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.config import load_config
from src.telemetry import TelemetryLogger

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AuraLSP Metrics",
    description="Telemetry and performance metrics for the AuraLSP server",
    version="1.0.0",
)

# Initialized on startup via lifespan or module-level (simple for now)
_telemetry: TelemetryLogger = None


@app.on_event("startup")
async def startup() -> None:
    global _telemetry
    config = load_config()
    _telemetry = TelemetryLogger(db_path=config.metrics.db_path)
    logger.info(f"Metrics API started. DB: {config.metrics.db_path}")


@app.get("/", summary="Health check")
async def root() -> dict:
    return {"status": "ok", "service": "AuraLSP Metrics API", "version": "1.0.0"}


@app.get("/metrics", summary="Aggregate telemetry metrics")
async def get_metrics() -> JSONResponse:
    """
    Returns aggregate completion statistics:
    - Total completions and last-24h count
    - Acceptance rate (% of completions user accepted via Tab)
    - TTFT and total latency averages + percentiles
    - Average completion length
    """
    if not _telemetry:
        raise HTTPException(status_code=503, detail="Telemetry not initialized")

    try:
        metrics = _telemetry.get_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error(f"Metrics query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recent", summary="Recent completion events")
async def get_recent(limit: int = 20) -> JSONResponse:
    """
    Returns the most recent N completion events.
    Useful for debugging what context was injected for each completion.
    """
    if not _telemetry:
        raise HTTPException(status_code=503, detail="Telemetry not initialized")

    if limit > 100:
        limit = 100

    try:
        import sqlite3, json
        conn = sqlite3.connect(str(_telemetry.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, timestamp, file_path, language_id, ttft_ms,
                   total_ms, completion_length, accepted
            FROM completions
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()
        conn.close()

        events = [dict(row) for row in rows]
        # Convert accepted int to bool/None
        for e in events:
            if e["accepted"] == 1:
                e["accepted"] = True
            elif e["accepted"] == 0:
                e["accepted"] = False
            else:
                e["accepted"] = None

        return JSONResponse(content={"count": len(events), "events": events})

    except Exception as e:
        logger.error(f"Recent query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def start_metrics_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Start the metrics API server. Blocks until shutdown."""
    uvicorn.run(
        "src.metrics_api:app",
        host=host,
        port=port,
        log_level="warning",
        reload=False,
    )


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    config = load_config()
    start_metrics_server(port=config.metrics.metrics_port)
