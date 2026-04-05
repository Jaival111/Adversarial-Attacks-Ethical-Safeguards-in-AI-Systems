from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RequestLogEntry:
    request_id: str
    ts_iso: str
    predicted_class: int
    label_name: str
    confidence: float
    detector_flags: dict[str, bool]
    detector_scores: dict[str, float]
    risk_tier: str
    input_anomaly_notes: list[str]


class JsonlLogger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path

    def append(self, entry: RequestLogEntry) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(entry), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def new_request_id() -> str:
    return str(uuid.uuid4())


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def summarize_shift(prev_conf: float | None, new_conf: float) -> dict[str, Any]:
    if prev_conf is None:
        return {"prediction_shift": None}
    return {"prediction_shift": abs(new_conf - prev_conf)}
