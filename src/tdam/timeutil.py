"""Time helpers. All internal times are integer epoch seconds (UTC)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


def parse_iso(text: str) -> int:
    """Parse an ISO-8601 timestamp (a trailing Z means UTC) to epoch seconds."""
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp())


def iso(epoch_s: int) -> str:
    return datetime.fromtimestamp(int(epoch_s), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ceil_div(a, b):
    """Ceiling division that works for ints and numpy integer arrays."""
    return -(-a // b)


@dataclass(frozen=True)
class Timeline:
    """Bin b covers event times in (t0 + (b-1)*s, t0 + b*s]; bin 0 ends at t0.

    A record stamped exactly at decision time t therefore sits in bin t, which
    matches the contract "event time in (t - W, t]".
    """

    t0: int
    n_bins: int
    bin_seconds: int = 60

    def bin_of(self, epoch_s):
        return ceil_div(epoch_s - self.t0, self.bin_seconds)

    def time_of(self, b):
        return self.t0 + b * self.bin_seconds

    @property
    def end(self) -> int:
        return self.time_of(self.n_bins - 1)
