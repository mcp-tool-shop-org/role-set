import pytest

from role_set.cost import PROFILES, estimate


def test_estimate_scales_with_slots_and_attempts():
    one = estimate(40, "two-card")
    two = estimate(80, "two-card")
    assert two.tokens == 2 * one.tokens
    assert two.pod_hours > one.pod_hours
    assert one.dollars == pytest.approx(one.pod_hours * PROFILES["two-card"]["rate_per_hour"])
    more = estimate(40, "two-card", attempts=3.0)
    assert more.calls == 120 and more.tokens == 2 * one.tokens


def test_critic_on_pod_adds_its_tokens_and_dict_rounds():
    a = estimate(20, "frontier")
    b = estimate(20, "frontier", critic_on_pod=True)
    assert b.tokens > a.tokens
    d = b.as_dict()
    assert set(d) == {"slots", "calls", "tokens", "pod_hours", "dollars"}
    assert d["slots"] == 20


def test_estimate_refuses_bad_inputs():
    with pytest.raises(ValueError, match="unknown profile"):
        estimate(10, "mystery")
    with pytest.raises(ValueError, match="positive"):
        estimate(0, "frontier")
