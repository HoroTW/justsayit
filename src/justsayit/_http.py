"""Shared HTTP utilities: retry-capable urllib wrapper."""
from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

_RETRYABLE_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def request_with_retry(
    req: urllib.request.Request,
    *,
    timeout: float,
    retries: int,
    delay: float,
    label: str = "HTTP",
) -> bytes:
    """Execute *req* with retry on transient errors, returning the response body."""
    attempts = 1 + max(0, retries)
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            retryable = exc.code in _RETRYABLE_CODES
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                detail = ""
            if not retryable or attempt >= attempts:
                raise RuntimeError(
                    f"{label} endpoint returned HTTP {exc.code}: {exc.reason}\n  {detail}"
                ) from exc
            log.warning(
                "%s request failed with HTTP %d; retrying %d/%d in %.1fs",
                label, exc.code, attempt, attempts - 1, delay,
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            if attempt >= attempts:
                raise RuntimeError(f"{label} request failed: {reason}") from exc
            log.warning(
                "%s request failed; retrying %d/%d in %.1fs: %s",
                label, attempt, attempts - 1, delay, reason,
            )
        time.sleep(max(0.0, delay))
    raise AssertionError("unreachable")
