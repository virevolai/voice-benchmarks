"""Create a Presence session and return its websocket URL.

The benchmark needs a live session websocket. Rather than hand-pasting an
internal endpoint, this mirrors the documented integration path: authenticate
against the public API, create a session, and use the `realtime.url` it
returns. That is what an integrator actually does, so the benchmark exercises
the same path a customer would.

Set BOHITA_API_KEY (and optionally BOHITA_API_BASE_URL). Alternatively set
PRESENCE_WS_URL directly to skip session creation entirely — useful for
pointing the harness at a local or self-hosted deployment.
"""

from __future__ import annotations

import os
from uuid import uuid4

DEFAULT_API_BASE = "https://api.bohita.com"


async def resolve_ws_url(explicit: str = "") -> str:
    """Return a session websocket URL, creating a session if needed.

    Precedence: an explicit --url, then $PRESENCE_WS_URL, then a freshly
    created session via the public API. The first two exist so the harness can
    be pointed at any deployment; the third is the path a reader without an
    endpoint will take.
    """
    if explicit:
        return explicit
    preset = os.environ.get("PRESENCE_WS_URL", "").strip()
    if preset:
        return preset

    key = os.environ.get("BOHITA_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "set BOHITA_API_KEY to create a session, or PRESENCE_WS_URL to "
            "use an existing endpoint"
        )

    import httpx

    base = (
        os.environ.get("BOHITA_API_BASE_URL")
        or os.environ.get("BOHITA_API_BASE")
        or DEFAULT_API_BASE
    ).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await _post_session(client, base, key)
    except httpx.ConnectError as exc:
        raise SystemExit(
            f"could not reach {base}: {exc}. Override the host with "
            "BOHITA_API_BASE_URL, or set PRESENCE_WS_URL to skip session creation."
        ) from exc

    if response.status_code >= 400:
        raise SystemExit(
            f"session create failed ({response.status_code}): {response.text[:200]}"
        )
    body = response.json()

    # The session websocket lives under `realtime.url` and already carries a
    # short-lived, session-scoped token. Treat it as a credential: it is not
    # written to results, and it should not be logged.
    realtime = body.get("realtime") or {}
    url = realtime.get("url")
    if not url:
        raise SystemExit(
            "session response carried no realtime.url — is the surface type "
            f"voice-capable? keys: {sorted(body)}"
        )
    return url


async def _post_session(client, base: str, key: str):
    return await client.post(
        f"{base}/v1/sessions",
        headers={
            "Authorization": f"Bearer {key}",
            "Idempotency-Key": os.environ.get(
                "BENCH_IDEMPOTENCY_KEY",
                f"voice-benchmark-{uuid4().hex}",
            ),
        },
        json={
            "surface": {"type": "voice"},
            "external_id": os.environ.get("BENCH_EXTERNAL_ID", "voice-benchmark"),
        },
    )
