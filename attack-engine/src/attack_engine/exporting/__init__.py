"""Phase 5: ground truth, provenance and dataset export.

This package turns a validated :class:`~attack_engine.attacks.executor.ExecutionSummary`
(Phases 3-4) into a self-contained, versioned export bundle: an audit dataset, a
detector-safe input dataset, a ground-truth dataset, an event-level manifest, a
provenance record and a human-readable handover. It never re-runs or recomputes
attacks; it only reads already-committed results and writes files.
"""

from __future__ import annotations
