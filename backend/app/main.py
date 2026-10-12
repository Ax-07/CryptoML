"""CryptoML P0 API: never place orders, no keys, no training inside web requests."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import router
from app.core.config import get_settings

app = FastAPI(
    title="CryptoML API",
    description="P0: public Binance USD-M Futures M5 historical ingestion, datasets and candles",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().cors_origins),
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={
        "error": {"code": f"HTTP_{exc.status_code}", "message": str(exc.detail),
                  "request_id": str(uuid.uuid4())}
    })


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={
        "error": {"code": "VALIDATION_ERROR", "message": "Invalid request parameters",
                  "request_id": str(uuid.uuid4())}
    })


@app.get("/health", tags=["System"])
def health() -> dict:
    return {"status": "ok", "app": "cryptoml", "version": "0.1.0",
            "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}


app.include_router(router)
