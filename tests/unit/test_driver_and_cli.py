"""The loop driver and the CLI commands that expose it."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from anvil.cli import main
from anvil.driver import DriverOptions, pending_job_ids, run_loop, step
from anvil.state import AnvilState


class FakeCkpt:
    def __init__(self, state, round=0, jobs=None):
        self.state = state
        self.round = round
        self.dft_calls_used = 0
        self.pending_job_ids = jobs if jobs is not None else []


class FakeOrch:
    """Records which transition methods the driver calls, in order."""

    def __init__(self, state=AnvilState.INIT, round=0, jobs=None):
        self.ckpt = FakeCkpt(state, round, jobs)
        self.run_dir = Path("/tmp/fake_run")
        self.calls: list[str] = []

    def _record(self, name, next_state=None, **_):
        self.calls.append(name)
        if next_state is not None:
            self.ckpt.state = next_state

    def bootstrap(self, **kw):
        self._record("bootstrap", AnvilState.BOOTSTRAP_DFT_QUEUED)

    def bootstrap_dft_done(self):
        self._record("bootstrap_dft_done", AnvilState.BOOTSTRAP_DFT_DONE)

    def val_dft_done(self):
        self._record("val_dft_done", AnvilState.VAL_DFT_DONE)

    def bootstrap_train(self, **kw):
        self._record("bootstrap_train", AnvilState.BOOTSTRAP_TRAIN)

    def bootstrap_trained(self):
        self._record("bootstrap_trained", AnvilState.BOOTSTRAP_TRAINED)

    def round_validated(self, **kw):
        self._record("round_validated", AnvilState.ROUND_CANDIDATES)

    def round_candidates(self, **kw):
        self._record("round_candidates", AnvilState.ROUND_ACQUIRED)

    def round_dft_queued(self, **kw):
        self._record("round_dft_queued", AnvilState.ROUND_DFT_QUEUED)

    def round_dft_done(self):
        self._record("round_dft_done", AnvilState.ROUND_DFT_DONE)

    def round_train(self, **kw):
        self._record("round_train", AnvilState.ROUND_TRAIN)

    def round_trained(self):
        self._record("round_trained", AnvilState.ROUND_TRAINED)

    def round_validated_and_decide(self, **kw):
        self._record("round_validated_and_decide", AnvilState.FINAL_COMPILE)


def _opts(**kw):
    kw.setdefault("log", lambda _m: None)
    kw.setdefault("poll_seconds", 0.0)
    return DriverOptions(**kw)


def test_driver_walks_the_whole_state_machine():
    orch = FakeOrch()
    final = run_loop(orch, _opts())
    assert final is AnvilState.FINAL_COMPILE       # stops: nothing implements it
    assert orch.calls == [
        "bootstrap", "bootstrap_dft_done", "val_dft_done", "bootstrap_train",
        "bootstrap_trained", "round_validated", "round_candidates",
        "round_dft_queued", "round_dft_done", "round_train", "round_trained",
        "round_validated_and_decide",
    ]


def test_driver_stops_at_max_rounds():
    orch = FakeOrch(state=AnvilState.ROUND_CANDIDATES, round=3)
    run_loop(orch, _opts(max_rounds=3))
    assert orch.calls == []                        # round 3 >= max_rounds


def test_driver_stops_in_terminal_state():
    orch = FakeOrch(state=AnvilState.DONE)
    assert run_loop(orch, _opts()) is AnvilState.DONE
    assert orch.calls == []


def test_driver_waits_for_slurm_before_collecting(monkeypatch):
    """A queued state must poll Slurm before reading job output."""
    seen: list[list[str]] = []
    states = iter([{"1": "RUNNING", "2": "COMPLETED"}, {"1": "COMPLETED", "2": "COMPLETED"}])

    def fake_poll(ids, ssh_host=None):
        seen.append(list(ids))
        return next(states)

    monkeypatch.setattr("anvil.hpc.slurm.poll", fake_poll)
    orch = FakeOrch(state=AnvilState.BOOTSTRAP_DFT_QUEUED, jobs=["1", "2", "DONE", ""])
    step(orch, _opts())
    assert seen == [["1", "2"], ["1", "2"]]        # placeholders filtered out
    assert orch.calls == ["bootstrap_dft_done", "val_dft_done"]


def test_pending_job_ids_filters_placeholders():
    orch = FakeOrch(jobs=["5", "DONE", "", "6"])
    assert pending_job_ids(orch) == ["5", "6"]


def test_wait_times_out_rather_than_hanging(monkeypatch):
    from anvil.driver import wait_for_jobs

    monkeypatch.setattr("anvil.hpc.slurm.poll", lambda ids, ssh_host=None: {"1": "RUNNING"})
    orch = FakeOrch(state=AnvilState.ROUND_DFT_QUEUED, jobs=["1"])
    with pytest.raises(TimeoutError, match="still live"):
        wait_for_jobs(orch, _opts(max_wait_hours=0.0))


# --------------------------------------------------------------------- CLI


def test_cli_exposes_the_new_commands():
    out = CliRunner().invoke(main, ["--help"]).output
    for cmd in ("submit", "collect", "daemon", "cancel", "status"):
        assert cmd in out


def test_submit_accepts_skip_pools():
    out = CliRunner().invoke(main, ["submit", "--help"]).output
    assert "--skip-pools" in out


def test_collect_refuses_at_the_wrong_state(monkeypatch, tmp_path):
    class Stub:
        ckpt = FakeCkpt(AnvilState.BOOTSTRAP_TRAINED)
        run_dir = tmp_path

    monkeypatch.setattr("anvil.orchestrator.Orchestrator.resume",
                        classmethod(lambda cls, d: Stub()))
    monkeypatch.setattr("anvil.cli._resolve_run_dir", lambda r: tmp_path)
    res = CliRunner().invoke(main, ["collect", str(tmp_path)])
    assert res.exit_code != 0
    assert "Nothing to collect" in res.output


def test_cancel_reports_when_there_is_nothing_to_cancel(monkeypatch, tmp_path):
    class Stub:
        ckpt = FakeCkpt(AnvilState.ROUND_DFT_QUEUED, jobs=["DONE", ""])
        run_dir = tmp_path

    monkeypatch.setattr("anvil.orchestrator.Orchestrator.resume",
                        classmethod(lambda cls, d: Stub()))
    monkeypatch.setattr("anvil.cli._resolve_run_dir", lambda r: tmp_path)
    res = CliRunner().invoke(main, ["cancel", str(tmp_path)])
    assert res.exit_code == 0
    assert "No pending jobs" in res.output
