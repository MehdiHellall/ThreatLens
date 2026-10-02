"""Request guards and response security headers for the public API."""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from web.backend.settings import AppSettings

PREDICTION_PATHS = frozenset({"/predict", "/v1/predict"})
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cache-Control": "no-store",
}
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")
MAX_PENDING_PREDICTIONS = 5
PREDICTION_QUEUE_TIMEOUT_SECONDS = 5.0
logger = logging.getLogger(__name__)


class RequestRateLimiter:
    """Small per-process fixed-window limiter for unauthenticated inference calls."""

    def __init__(self, limit_per_minute: int, *, max_clients: int = 10_000) -> None:
        self._limit_per_minute = limit_per_minute
        self._max_clients = max(max_clients, 1)
        self._lock = threading.Lock()
        self._requests_by_client: OrderedDict[str, deque[float]] = OrderedDict()

    def allow(self, client_id: str, now: float | None = None) -> bool:
        if self._limit_per_minute <= 0:
            return True

        current_time = now if now is not None else time.monotonic()
        window_start = current_time - 60
        with self._lock:
            timestamps = self._requests_by_client.pop(client_id, None)
            if timestamps is None:
                if len(self._requests_by_client) >= self._max_clients:
                    self._requests_by_client.popitem(last=False)
                timestamps = deque()
            self._requests_by_client[client_id] = timestamps
            while timestamps and timestamps[0] < window_start:
                timestamps.popleft()
            if len(timestamps) >= self._limit_per_minute:
                return False
            timestamps.append(current_time)
            return True


def _client_id(request: Request) -> str:
    if request.client is None:
        return "unknown"
    return request.client.host


def _content_length(request: Request) -> int | None:
    raw_content_length = request.headers.get("content-length")
    if raw_content_length is None:
        return None
    try:
        return int(raw_content_length)
    except ValueError:
        return -1


def _guard_prediction_request(
    request: Request,
    settings: AppSettings,
    rate_limiter: RequestRateLimiter,
) -> JSONResponse | None:
    content_length = _content_length(request)
    if content_length is None:
        return JSONResponse(
            status_code=411,
            content={"detail": "Content-Length is required for prediction requests."},
        )
    if content_length < 0 or content_length > settings.max_body_bytes:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request body must be {settings.max_body_bytes} bytes or fewer."},
        )
    if not rate_limiter.allow(_client_id(request)):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many prediction requests. Please wait and try again."},
            headers={"Retry-After": "60"},
        )
    return None


def _request_id(request: Request) -> str:
    supplied = request.headers.get("x-request-id", "")
    if REQUEST_ID_PATTERN.fullmatch(supplied):
        return supplied
    return uuid.uuid4().hex


class PredictionSecurityMiddleware(BaseHTTPMiddleware):
    """Protect inference routes and attach browser-safe response headers."""

    def __init__(self, app, *, settings: AppSettings) -> None:
        super().__init__(app)
        self._settings = settings
        self._rate_limiter = RequestRateLimiter(settings.rate_limit_per_minute)
        self._prediction_lock = asyncio.Lock()
        self._admission_lock = asyncio.Lock()
        self._pending_predictions = 0

    async def _admit_prediction(self) -> bool:
        async with self._admission_lock:
            if self._pending_predictions >= MAX_PENDING_PREDICTIONS:
                return False
            self._pending_predictions += 1
        try:
            await asyncio.wait_for(
                self._prediction_lock.acquire(),
                timeout=PREDICTION_QUEUE_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            async with self._admission_lock:
                self._pending_predictions -= 1
            return False
        return True

    async def _release_prediction(self) -> None:
        self._prediction_lock.release()
        async with self._admission_lock:
            self._pending_predictions -= 1

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = _request_id(request)
        started_at = time.monotonic()
        response: Response
        is_prediction = request.method.upper() == "POST" and request.url.path in PREDICTION_PATHS
        guard_response = (
            _guard_prediction_request(request, self._settings, self._rate_limiter)
            if is_prediction
            else None
        )
        admitted = False
        if guard_response is None and is_prediction:
            body = await request.body()
            if len(body) > self._settings.max_body_bytes:
                guard_response = JSONResponse(
                    status_code=413,
                    content={
                        "detail": (
                            f"Request body must be {self._settings.max_body_bytes} bytes or fewer."
                        )
                    },
                )
            else:
                admitted = await self._admit_prediction()
                if not admitted:
                    guard_response = JSONResponse(
                        status_code=503,
                        content={
                            "detail": "The prediction service is busy. Please try again shortly."
                        },
                        headers={"Retry-After": "5"},
                    )
        try:
            response = guard_response if guard_response is not None else await call_next(request)
        finally:
            if admitted:
                await self._release_prediction()
        duration_ms = (time.monotonic() - started_at) * 1_000
        if is_prediction:
            logger.info(
                "prediction_request request_id=%s status=%s duration_ms=%.1f",
                request_id,
                response.status_code,
                duration_ms,
            )
        response.headers.setdefault("X-Request-ID", request_id)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response
