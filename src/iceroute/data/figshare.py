"""Minimal Figshare client with resumable downloads.

The AI4Arctic dataset lives on ``data.dtu.dk``, a Figshare instance. This module
knows how to list the files in an article and fetch them in a way that survives
a dropped connection, because the download is the longest pole on the critical
path (a slow or interrupted download blocks everything downstream) and restarting
a 40 GB transfer from zero is not an option.

Design notes
------------
* Every download writes to ``<name>.part`` and is renamed into place only after
  the size (and MD5, when the server supplied one) checks out. A file present
  without a ``.part`` sibling is therefore known-complete: that is what makes
  ``make download`` safely re-runnable.
* Resume uses HTTP Range. If the server ignores the range header we detect it
  (status 200 instead of 206) and restart the file rather than silently
  appending to it, which would corrupt the result in a way that only surfaces
  weeks later when a scene fails to parse.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)

FIGSHARE_API = "https://api.figshare.com/v2"

# Tuned for a home connection: long enough that a slow first byte does not kill
# a transfer, short enough that a black-holed socket does not hang overnight.
CONNECT_TIMEOUT = 15.0
READ_TIMEOUT = 120.0
CHUNK_BYTES = 1 << 20  # 1 MiB

MAX_RETRIES = 6
BACKOFF_BASE = 2.0  # seconds; doubles each retry, so 2, 4, 8, 16, 32, 64


class DownloadError(RuntimeError):
    """Raised when a file cannot be fetched after exhausting retries."""


class DiskSpaceError(RuntimeError):
    """Raised when continuing would take free space below the configured floor."""


@dataclass(frozen=True)
class FigshareFile:
    """One downloadable file in a Figshare article."""

    id: int
    name: str
    size: int
    download_url: str
    md5: str | None

    @classmethod
    def from_api(cls, payload: dict) -> FigshareFile:
        # Figshare calls it `supplied_md5` when the depositor provided one and
        # `computed_md5` when Figshare hashed it itself. Either is fine for our
        # purposes; prefer the supplied one.
        md5 = payload.get("supplied_md5") or payload.get("computed_md5") or None
        if md5 in ("", "null"):
            md5 = None
        return cls(
            id=int(payload["id"]),
            name=str(payload["name"]),
            size=int(payload["size"]),
            download_url=str(payload["download_url"]),
            md5=md5,
        )


def _get_json(url: str, *, params: dict | None = None) -> object:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 — retry on anything transient
            last_exc = exc
            wait = BACKOFF_BASE * (2**attempt)
            log.warning(
                "GET %s failed (attempt %d/%d): %s — retrying in %.0fs",
                url,
                attempt + 1,
                MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)
    raise DownloadError(f"Cannot read {url}: {last_exc}")


def list_collection_articles(collection_id: int) -> list[dict]:
    """Return the articles belonging to a Figshare collection."""
    out = _get_json(
        f"{FIGSHARE_API}/collections/{collection_id}/articles",
        params={"page_size": 100},
    )
    if not isinstance(out, list):
        raise DownloadError(f"Unexpected response for collection {collection_id}: {out!r}")
    return out


def list_article_files(article_id: int) -> list[FigshareFile]:
    """Return every file in a Figshare article, following pagination."""
    files: list[FigshareFile] = []
    page = 1
    while True:
        payload = _get_json(
            f"{FIGSHARE_API}/articles/{article_id}/files",
            params={"page": page, "page_size": 1000},
        )
        if not isinstance(payload, list):
            raise DownloadError(f"Unexpected response for article {article_id}: {payload!r}")
        if not payload:
            break
        files.extend(FigshareFile.from_api(item) for item in payload)
        if len(payload) < 1000:
            break
        page += 1
    return files


def free_bytes(path: Path) -> int:
    """Free space on the filesystem holding ``path`` (or its nearest parent)."""
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _md5_of(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 — integrity check against Figshare, not security
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK_BYTES), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, expected_size: int, expected_md5: str | None) -> tuple[bool, str]:
    """Check a downloaded file. Returns ``(ok, reason)``."""
    if not path.exists():
        return False, "missing"
    actual = path.stat().st_size
    if actual != expected_size:
        return False, f"size {actual} != expected {expected_size}"
    if expected_md5:
        digest = _md5_of(path)
        if digest != expected_md5:
            return False, f"md5 {digest} != expected {expected_md5}"
    return True, "ok"


def download_file(
    spec: FigshareFile,
    dest_dir: Path,
    *,
    min_free_bytes: int,
    progress: bool = True,
) -> Path:
    """Download one file into ``dest_dir``, resuming if a partial exists.

    Raises
    ------
    DiskSpaceError
        If completing this file would push free space below ``min_free_bytes``.
    DownloadError
        If the transfer fails after ``MAX_RETRIES`` attempts, or if the
        completed file fails verification.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / spec.name
    part = dest_dir / f"{spec.name}.part"

    # Already done and intact? Then this is a no-op — the property that makes
    # `make download` re-runnable.
    if final.exists():
        ok, reason = verify(final, spec.size, spec.md5)
        if ok:
            log.info("already present, skipping: %s", spec.name)
            return final
        log.warning("%s present but invalid (%s) — downloading again", spec.name, reason)
        final.unlink()

    already = part.stat().st_size if part.exists() else 0
    remaining = spec.size - already

    avail = free_bytes(dest_dir)
    if avail - remaining < min_free_bytes:
        raise DiskSpaceError(
            f"Not enough space for {spec.name}: {remaining / 1e9:.1f} GB needed, "
            f"{avail / 1e9:.1f} GB free, minimum to keep free "
            f"{min_free_bytes / 1e9:.1f} GB."
        )

    for attempt in range(MAX_RETRIES):
        already = part.stat().st_size if part.exists() else 0

        if already > spec.size:
            log.warning("%s: partial file larger than expected, restarting from zero", spec.name)
            part.unlink()
            already = 0
        if already == spec.size:
            break

        headers = {"Range": f"bytes={already}-"} if already else {}
        mode = "ab" if already else "wb"

        try:
            with requests.get(
                spec.download_url,
                headers=headers,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                allow_redirects=True,
            ) as resp:
                # If we asked to resume and the server sent the whole file
                # anyway, appending would corrupt the result. Start over.
                if already and resp.status_code == 200:
                    log.warning(
                        "%s: server ignores Range (200 instead of 206) — restarting from zero",
                        spec.name,
                    )
                    resp.close()
                    if part.exists():
                        part.unlink()
                    continue
                resp.raise_for_status()

                got = already
                t0 = time.monotonic()
                last_report = t0
                with part.open(mode) as fh:
                    for chunk in resp.iter_content(chunk_size=CHUNK_BYTES):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        got += len(chunk)
                        now = time.monotonic()
                        if progress and now - last_report > 2.0:
                            pct = 100.0 * got / spec.size if spec.size else 0.0
                            rate = (got - already) / max(now - t0, 1e-6) / 1e6
                            print(
                                f"\r    {spec.name}  {pct:5.1f}%  "
                                f"{got / 1e9:.2f}/{spec.size / 1e9:.2f} GB  {rate:5.1f} MB/s",
                                end="",
                                flush=True,
                            )
                            last_report = now
                if progress:
                    print("\r" + " " * 100 + "\r", end="", flush=True)

        except Exception as exc:  # noqa: BLE001 — retry on anything transient
            wait = BACKOFF_BASE * (2**attempt)
            log.warning(
                "%s: transfer interrupted (attempt %d/%d): %s — retrying in %.0fs",
                spec.name,
                attempt + 1,
                MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)
            continue

        if part.exists() and part.stat().st_size == spec.size:
            break
    else:
        raise DownloadError(f"{spec.name}: download attempts exhausted")

    ok, reason = verify(part, spec.size, spec.md5)
    if not ok:
        raise DownloadError(f"{spec.name}: verification failed after download ({reason})")

    part.rename(final)
    return final


def iter_with_disk_guard(
    specs: list[FigshareFile],
    dest_dir: Path,
    *,
    min_free_bytes: int,
) -> Iterator[tuple[FigshareFile, Path]]:
    """Download each spec in turn, yielding as they complete."""
    for spec in specs:
        yield spec, download_file(spec, dest_dir, min_free_bytes=min_free_bytes)
