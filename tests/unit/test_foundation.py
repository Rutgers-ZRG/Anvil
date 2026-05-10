"""Unit tests for anvil.ml.foundation registry — no model load."""

from __future__ import annotations

import pytest

from anvil.ml.foundation import (
    ALLEGRO_OAM_L,
    FoundationConfig,
    FoundationError,
    FoundationRegistry,
)


def test_pre_registered() -> None:
    assert "allegro-oam-l-foundation" in FoundationRegistry.names()
    cfg = FoundationRegistry.get("allegro-oam-l-foundation")
    assert cfg is ALLEGRO_OAM_L


def test_path_for() -> None:
    cfg = FoundationRegistry.get("allegro-oam-l-foundation")
    p = cfg.path_for("amareln")
    assert p.endswith("allegro-oam-l-foundation.nequip.pth")
    assert p == cfg.path_for("amarel3")  # same path on both


def test_unknown_cluster() -> None:
    cfg = FoundationRegistry.get("allegro-oam-l-foundation")
    with pytest.raises(ValueError, match="no checkpoint registered"):
        cfg.path_for("bogus_cluster")


def test_unknown_foundation() -> None:
    with pytest.raises(FoundationError, match="Unknown foundation"):
        FoundationRegistry.get("nonexistent")


def test_register_custom() -> None:
    custom = FoundationConfig(
        name="custom-test",
        checkpoint_paths={"amareln": "/tmp/no_such.pth"},
    )
    FoundationRegistry.register(custom)
    try:
        assert "custom-test" in FoundationRegistry.names()
        assert FoundationRegistry.get("custom-test") is custom
    finally:
        # Clean up so other tests don't see it
        FoundationRegistry._registry.pop("custom-test", None)


def test_make_calc_missing_path(tmp_path) -> None:
    """make_calc should raise FoundationError when the file is absent."""
    custom = FoundationConfig(
        name="ghost",
        checkpoint_paths={"amareln": str(tmp_path / "no_such.pth")},
    )
    FoundationRegistry.register(custom)
    try:
        with pytest.raises(FoundationError, match="not found"):
            FoundationRegistry.make_calc("ghost", cluster="amareln")
    finally:
        FoundationRegistry._registry.pop("ghost", None)


def test_element_coverage_check(tmp_path) -> None:
    """If supported_elements is set, missing elements raise."""
    fake_path = tmp_path / "fake.pth"
    fake_path.write_bytes(b"junk")  # exists check passes; load will fail later

    custom = FoundationConfig(
        name="elem-restricted",
        checkpoint_paths={"amareln": str(fake_path)},
        supported_elements=frozenset({14, 6}),  # Si, C only
    )
    FoundationRegistry.register(custom)
    try:
        with pytest.raises(FoundationError, match="does not cover"):
            FoundationRegistry.make_calc(
                "elem-restricted", cluster="amareln", elements=["Fe"],
            )
    finally:
        FoundationRegistry._registry.pop("elem-restricted", None)
