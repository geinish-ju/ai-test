from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def artifact_metadata(
    path: str | Path,
    artifact_type: str,
    role: str | None = None,
) -> Record:
    artifact_path = Path(path)
    metadata: Record = {
        "path": str(artifact_path),
        "type": artifact_type,
        "exists": artifact_path.exists(),
    }
    if role is not None:
        metadata["role"] = role
    if artifact_path.is_file():
        metadata["size_bytes"] = artifact_path.stat().st_size
        metadata["sha256"] = _sha256(artifact_path)
    return metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
