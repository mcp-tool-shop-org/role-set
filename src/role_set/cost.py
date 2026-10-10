"""Cost before spend: the maintainer sees pod hours and dollars before anything is launched."""

from __future__ import annotations

from dataclasses import dataclass

# Secure-cloud rates the Publisher reported on 2026-10-10. Live prices are read by offrig at launch;
# these are for the estimate only.
PROFILES = {
    "two-card": {"rate_per_hour": 4.20, "tokens_per_second": 1500.0, "load_minutes": 12.0},
    "frontier": {"rate_per_hour": 8.36, "tokens_per_second": 2300.0, "load_minutes": 22.0},
}


@dataclass(frozen=True)
class Estimate:
    slots: int
    calls: float
    tokens: float
    pod_hours: float
    dollars: float

    def as_dict(self) -> dict:
        return {
            "slots": self.slots,
            "calls": round(self.calls, 1),
            "tokens": round(self.tokens),
            "pod_hours": round(self.pod_hours, 2),
            "dollars": round(self.dollars, 2),
        }


def estimate(
    slots: int,
    profile: str,
    *,
    attempts: float = 1.5,
    writer_tokens: int = 3500,
    critic_tokens: int = 1200,
    critic_on_pod: bool = False,
) -> Estimate:
    """slots x attempts x tokens per call, over the profile's throughput, plus the load time."""

    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; known: {sorted(PROFILES)}")
    if slots <= 0 or attempts <= 0:
        raise ValueError("slots and attempts must be positive")
    p = PROFILES[profile]
    calls = slots * attempts
    per_call = writer_tokens + (critic_tokens if critic_on_pod else 0)
    tokens = calls * per_call
    hours = tokens / p["tokens_per_second"] / 3600 + p["load_minutes"] / 60
    return Estimate(slots=slots, calls=calls, tokens=tokens, pod_hours=hours, dollars=hours * p["rate_per_hour"])
