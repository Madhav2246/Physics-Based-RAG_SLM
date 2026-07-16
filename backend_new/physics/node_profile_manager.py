"""
NodeProfileManager — process-node-aware default parameters.

Loads JSON profiles from physics/node_profiles/ and detects the requested node
from a query by keyword. Node-specific params (mu, Cox, tox, Vth, ...) override
the generic 100nm defaults; everything else (k, q, T, ...) falls back to
ValueTracker.DEFAULTS so physical constants are never lost.

No model, no regex magic — just a keyword scan. Returns None when no node is
named, and the caller falls back to the 100nm baseline (unchanged behavior).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from physics.value_tracker import TrackedValue, ValueTracker


PROFILE_DIR = Path(__file__).resolve().parent / "node_profiles"

# Keyword → profile filename stem. Ordered most-specific first so that
# "16nm FinFET" matches FinFET, not a generic nm match.
_KEYWORD_MAP: list[tuple[str, str]] = [
    ("gate-all-around", "5nm_GAA"),
    ("gaa",             "5nm_GAA"),
    ("5nm",             "5nm_GAA"),
    ("finfet",          "16nm_FinFET"),
    ("16nm",            "16nm_FinFET"),
    ("fdsoi",           "28nm_FDSOI"),
    ("fd-soi",          "28nm_FDSOI"),
    ("28nm",            "28nm_FDSOI"),
    ("100nm",           "100nm_CMOS"),
]

DEFAULT_NODE = "100nm_CMOS"


class NodeProfileManager:
    def __init__(self, profile_dir: Path = PROFILE_DIR) -> None:
        self.profile_dir = Path(profile_dir)
        self._cache: dict[str, dict] = {}

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self, node_name: str) -> dict:
        """
        Load and return a profile dict by name (filename stem).
        Cached. Raises FileNotFoundError if the profile does not exist.
        """
        if node_name in self._cache:
            return self._cache[node_name]
        path = self.profile_dir / f"{node_name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Node profile not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        self._cache[node_name] = data
        return data

    def list_nodes(self) -> list[str]:
        """Return the available profile filename stems, sorted."""
        return sorted(p.stem for p in self.profile_dir.glob("*.json"))

    # ── Detection ────────────────────────────────────────────────────────────

    def detect_from_query(self, query: str) -> Optional[str]:
        """
        Return the profile stem named in the query, or None if no node keyword
        is present. Most-specific keyword wins (map is ordered).
        """
        q = query.lower()
        for keyword, node_name in _KEYWORD_MAP:
            if keyword in q:
                return node_name
        return None

    # ── Tracker defaults ─────────────────────────────────────────────────────

    def as_tracker_defaults(self, node_name: str) -> dict[str, TrackedValue]:
        """
        Build the defaults dict for a node: start from the generic 100nm
        DEFAULTS (so constants like k, q, T survive), then override with the
        node's specific params. Every entry is tagged provenance='default'.
        """
        # Copy the class-level baseline so we never mutate it
        merged: dict[str, TrackedValue] = dict(ValueTracker.DEFAULTS)

        try:
            profile = self.load(node_name)
        except FileNotFoundError:
            # Unknown node — return the untouched 100nm baseline
            return merged

        profile_label = profile.get("name", node_name)
        for sym, spec in profile.get("params", {}).items():
            merged[sym] = TrackedValue(
                symbol=sym,
                value=float(spec["value"]),
                unit=spec.get("unit", ""),
                provenance="default",
                description=f"{spec.get('note', sym)} [{profile_label} profile]",
            )
        return merged
