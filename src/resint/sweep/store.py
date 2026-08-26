"""Sweep results on disk: one JSON object per line.

JSONL rather than a database, for reasons about failure rather than querying.
A run that dies at paper 180 of 250 leaves 180 valid lines and needs no
recovery step. Schema changes cost nothing over a campaign that will change
the schema repeatedly. And under a process pool the parent is the only writer
anyway, which is the one thing a database would have been good for here.

The caches are a different matter — reference lookups and model verdicts want
concurrent readers and atomic single-key writes, and those use sqlite.
"""

from __future__ import annotations

import json
from pathlib import Path

from .record import PaperRecord


def write_record(handle, record: PaperRecord) -> None:
    """Append one record, flushed.

    Flushing per record is deliberate: an interrupted sweep should lose only
    the paper that was in flight, not everything since the last buffer.
    """
    handle.write(json.dumps(record.as_dict(), sort_keys=True) + "\n")
    handle.flush()


def read_records(path: str | Path) -> list[PaperRecord]:
    """Load a sweep.

    A truncated final line is skipped rather than fatal — that is what an
    interrupted run leaves behind, and refusing to read the other 179 records
    because of it would be absurd. A malformed line anywhere else is a real
    problem and raises.
    """
    out: list[PaperRecord] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            out.append(PaperRecord.from_dict(json.loads(line)))
        except (ValueError, TypeError) as exc:
            if number == len(lines):
                break  # a partial write from an interrupted run
            raise ValueError(f"{path} line {number}: {exc}") from exc

    return out
