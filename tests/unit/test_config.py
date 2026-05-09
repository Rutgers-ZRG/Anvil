"""Unit tests for anvil.config."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from anvil.config import ConfigError, load_config, to_canonical_json


SAMPLE_YAML = """\
system: si
elements: [Si]
seeds:
  from_local:
    - data/si_seeds/si_diamond.vasp
dft:
  functional: pbe
  encut: 600
regime:
  pressures_gpa: [0, 5, 10]
  temperatures_k: [300, 1000]
budget:
  max_dft_calls: 200
  max_walltime_h: 12
"""


def test_load_minimal(tmp_path: Path) -> None:
    p = tmp_path / "si.yaml"
    p.write_text(SAMPLE_YAML)
    cfg = load_config(p)
    assert cfg.system == "si"
    assert cfg.elements == ["Si"]
    assert cfg.dft.encut == 600
    assert cfg.regime.pressures_gpa == [0, 5, 10]
    # Defaults present
    assert cfg.training.ensemble_size == 3
    assert cfg.reform_relax.mode == "stability"


def test_missing_required(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("system: si\n")
    with pytest.raises(ConfigError, match="Missing required"):
        load_config(p)


def test_unknown_key(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(SAMPLE_YAML + "\nbogus_section: 42\n")
    with pytest.raises(ConfigError, match="Unknown keys"):
        load_config(p)


def test_seeds_xor(tmp_path: Path) -> None:
    bad = SAMPLE_YAML + textwrap.dedent("""
    """).join([])
    # Replace seeds block: both from_mp and from_local
    text = SAMPLE_YAML.replace(
        "seeds:\n  from_local:\n    - data/si_seeds/si_diamond.vasp",
        "seeds:\n  from_mp: [mp-149]\n  from_local: [foo.vasp]",
    )
    p = tmp_path / "bad.yaml"
    p.write_text(text)
    with pytest.raises(ConfigError, match="cannot specify both"):
        load_config(p)


def test_invalid_mode(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(SAMPLE_YAML + "\nreform_relax:\n  mode: bogus\n")
    with pytest.raises(ConfigError, match="reform_relax.mode"):
        load_config(p)


def test_canonical_json_stable(tmp_path: Path) -> None:
    p = tmp_path / "si.yaml"
    p.write_text(SAMPLE_YAML)
    cfg = load_config(p)
    j1 = to_canonical_json(cfg)
    j2 = to_canonical_json(cfg)
    assert j1 == j2  # deterministic
    assert '"system":' in j1 or '"system"' in j1
