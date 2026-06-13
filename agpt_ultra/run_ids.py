from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4


def make_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}-{uuid4().hex[:8]}"


def resolve_run_id(run_id: str | None, *, resume: bool = False) -> str | None:
    if run_id:
        return run_id
    if resume:
        return None
    return make_run_id()


def prefixed_path(path: Path, run_id: str | None) -> Path:
    if run_id is None:
        return path
    if path.name.startswith(f"{run_id}_"):
        return path
    return path.with_name(f"{run_id}_{path.name}")

