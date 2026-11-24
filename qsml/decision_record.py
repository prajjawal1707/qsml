
"""QSML 4.8 — Reproducible decision records and integrity verification."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def _json_safe(value):
    """Convert supported objects into finite JSON-safe values."""

    if is_dataclass(value):
        return _json_safe(asdict(value))

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Decision records cannot contain NaN or infinity")
        return value

    if isinstance(value, dict):
        output = {}

        for key, item in value.items():
            output[str(key)] = _json_safe(item)

        return output

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    raise ValueError(
        f"Unsupported value in decision record: {type(value).__name__}"
    )


def _canonical_json(record):
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _record_hash(record):
    payload = dict(record)
    payload.pop("record_hash", None)

    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def create_decision_record(
    configuration,
    result,
    project_version="4.8",
):
    """Create a timestamped, hash-verifiable decision record."""

    if not isinstance(configuration, dict):
        raise ValueError("configuration must be a dictionary")

    if not isinstance(result, dict):
        raise ValueError("result must be a dictionary")

    if not isinstance(project_version, str) or not project_version:
        raise ValueError("project_version must be non-empty")

    record = {
        "record_format": "QSML_DECISION_RECORD",
        "record_schema_version": 1,
        "project_version": project_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": _json_safe(configuration),
        "result": _json_safe(result),
    }

    record["record_hash"] = _record_hash(record)

    return record


def verify_decision_record(record):
    """Verify a decision record's SHA-256 content hash."""

    if not isinstance(record, dict):
        raise ValueError("record must be a dictionary")

    supplied_hash = record.get("record_hash")

    if not isinstance(supplied_hash, str):
        raise ValueError("record_hash is missing")

    calculated_hash = _record_hash(record)

    return {
        "valid": supplied_hash == calculated_hash,
        "supplied_hash": supplied_hash,
        "calculated_hash": calculated_hash,
        "algorithm": "SHA256",
    }


def save_decision_record(path, record):
    """Atomically save a decision record as JSON."""

    path = Path(path)

    if not isinstance(record, dict):
        raise ValueError("record must be a dictionary")

    verification = verify_decision_record(record)

    if not verification["valid"]:
        raise ValueError("Cannot save a decision record with invalid hash")

    path.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(
        record,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, path)

    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise

    return str(path)


def load_decision_record(path):
    """Load a record and verify its content hash."""

    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        record = json.load(file)

    verification = verify_decision_record(record)

    return {
        "record": record,
        "integrity": verification,
    }
