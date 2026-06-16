"""FastAPI server — currently just a liveness probe.

clear-api enqueues pipeline work (manual signals, translation requests,
etc.) by pushing Celery messages directly to the shared Redis broker,
not via HTTP into this container. That keeps clear-api ↔ pipeline
decoupled at the network layer: only the broker needs to be reachable.

Leaving the FastAPI server in place so the docker / VM healthcheck
endpoint (`GET /health`) keeps working. New endpoints can be added
here when something genuinely needs synchronous request/response — for
fire-and-forget work, use Celery via the broker instead.
"""

import logging

from fastapi import FastAPI

logger = logging.getLogger(__name__)

app = FastAPI(title="CLEAR Pipeline API", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok"}
