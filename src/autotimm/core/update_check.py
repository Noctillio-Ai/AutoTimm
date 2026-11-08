"""Tell users when a newer AutoTimm release is on PyPI.

``import autotimm`` calls :func:`check_for_updates`. The latest version is
looked up on PyPI at most once a day (cached under the user cache dir); the
lookup runs in a background thread with a short timeout, so it never blocks
or breaks the import, and failures (offline, firewall) are silent.

Disable it with ``AUTOTIMM_DISABLE_UPDATE_CHECK=1``. It is also skipped in
CI, in offline mode (``AUTOTIMM_OFFLINE`` / ``HF_HUB_OFFLINE``), in
DataLoader worker processes, on non-zero distributed ranks, and for
development builds.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
import urllib.request
from pathlib import Path

from loguru import logger

PACKAGE_NAME = "autotimm"
PYPI_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
CACHE_TTL_SECONDS = 24 * 60 * 60
REQUEST_TIMEOUT_SECONDS = 5.0

_TRUTHY = {"1", "true", "yes", "on"}
_notified = False
_lock = threading.Lock()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _cache_file() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / PACKAGE_NAME / "update_check.json"


def _parse(v: str):
    try:
        from packaging.version import Version

        return Version(v)
    except Exception:
        return None


def _should_skip(current: str) -> bool:
    if _env_flag("AUTOTIMM_DISABLE_UPDATE_CHECK"):
        return True
    if _env_flag("AUTOTIMM_OFFLINE") or _env_flag("HF_HUB_OFFLINE"):
        return True
    if os.environ.get("CI"):
        return True
    # DataLoader workers (spawn re-imports autotimm) and non-zero DDP ranks
    # would otherwise each print the notice.
    if multiprocessing.parent_process() is not None:
        return True
    for var in ("RANK", "LOCAL_RANK"):
        if os.environ.get(var, "0") != "0":
            return True
    parsed = _parse(current)
    # Unknown versions and dev/local builds (source checkouts) are not
    # comparable with releases.
    return parsed is None or parsed.is_devrelease or parsed.local is not None


def _read_cache() -> str | None:
    try:
        data = json.loads(_cache_file().read_text())
        if time.time() - float(data["checked_at"]) < CACHE_TTL_SECONDS:
            return str(data["latest"])
    except Exception:
        pass
    return None


def _write_cache(latest: str) -> None:
    try:
        path = _cache_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"checked_at": time.time(), "latest": latest}))
    except Exception:
        pass


def fetch_latest_version(timeout: float = REQUEST_TIMEOUT_SECONDS) -> str | None:
    """Return the latest released version on PyPI, or ``None`` on failure."""
    try:
        req = urllib.request.Request(PYPI_URL, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return str(json.load(resp)["info"]["version"])
    except Exception:
        return None


def _notify_if_outdated(current: str, latest: str | None) -> bool:
    """Log an upgrade notice (once per process) if ``latest`` is newer."""
    global _notified
    cur, new = _parse(current), _parse(latest) if latest else None
    if cur is None or new is None or new.is_prerelease or new <= cur:
        return False
    with _lock:
        if _notified:
            return False
        _notified = True
    logger.warning(
        f"A new version of {PACKAGE_NAME} is available: {current} -> {latest}. "
        f"Upgrade with: pip install -U {PACKAGE_NAME}  "
        "(set AUTOTIMM_DISABLE_UPDATE_CHECK=1 to silence this check)"
    )
    return True


def _fetch_and_notify(current: str) -> None:
    latest = fetch_latest_version()
    if latest is not None:
        _write_cache(latest)
        _notify_if_outdated(current, latest)


def check_for_updates(current: str | None = None, blocking: bool = False) -> None:
    """Warn if a newer AutoTimm release is available on PyPI.

    Args:
        current: Installed version; defaults to ``autotimm.__version__``.
        blocking: Wait for the PyPI lookup instead of running it in a
            background thread.
    """
    try:
        if current is None:
            from autotimm import __version__ as current
        if _should_skip(current):
            return
        cached = _read_cache()
        if cached is not None:
            _notify_if_outdated(current, cached)
            return
        if blocking:
            _fetch_and_notify(current)
        else:
            threading.Thread(
                target=_fetch_and_notify,
                args=(current,),
                name="autotimm-update-check",
                daemon=True,
            ).start()
    except Exception:
        # An update check must never break importing the library.
        pass
