"""Running resint over many papers, and measuring what came back.

Developer tooling, not a user feature. It lives in the library rather than in
``tools/`` because it needs tests and because ``Finding.from_dict`` is
generally useful; the driver that calls it lives in ``tools/sweep.py``.

The point is not to produce findings. It is to answer the two questions that
need no human judgement — did anything crash, and does every finding point at
text that actually exists — and then hand a person a short list worth reading.
"""

from .record import AnchorAudit, AnchorFailure, PaperRecord, audit_anchors, fingerprint
from .runner import check_one, source_texts
from .store import read_records, write_record

__all__ = [
    "AnchorAudit",
    "AnchorFailure",
    "PaperRecord",
    "audit_anchors",
    "check_one",
    "fingerprint",
    "read_records",
    "source_texts",
    "write_record",
]
