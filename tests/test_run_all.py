# tests/test_run_all.py
"""
Tests for the streaming stack launcher's readiness polling and shutdown
order, using fake processes (nothing is started).
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from streaming import run_all


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeProc:
    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def test_wait_until_returns_when_check_passes():
    clock, calls = FakeClock(), iter([False, False, True])
    waited = run_all.wait_until("svc", lambda: next(calls), timeout=10,
                                proc=FakeProc(), poll=0.5, clock=clock,
                                sleep=clock.sleep)
    assert waited == 1.0


def test_wait_until_times_out():
    clock = FakeClock()
    with pytest.raises(run_all.StartupError, match="not ready after 3s"):
        run_all.wait_until("svc", lambda: False, timeout=3, poll=1,
                           clock=clock, sleep=clock.sleep)


def test_wait_until_fails_fast_when_the_process_exits():
    clock = FakeClock()
    with pytest.raises(run_all.StartupError, match="exited with code 2"):
        run_all.wait_until("svc", lambda: False, timeout=600,
                           proc=FakeProc(returncode=2), clock=clock,
                           sleep=clock.sleep)
    assert clock.now == 0.0


def test_consumer_online_ignores_offline_and_stale_status():
    since = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    after = (since + timedelta(seconds=5)).isoformat()
    before = (since - timedelta(hours=1)).isoformat()
    assert run_all.consumer_online({"state": "online", "ts": after}, since)
    assert not run_all.consumer_online({"state": "online", "ts": before},
                                       since)  # left over from a past run
    assert not run_all.consumer_online({"state": "offline", "ts": after},
                                       since)
    assert not run_all.consumer_online(None, since)
    assert not run_all.consumer_online({"state": "online"}, since)


class FakeService:
    log = []

    def __init__(self, name, cmd):
        self.name, self.cmd = name, cmd
        self.proc = FakeProc()

    def start(self, env):
        FakeService.log.append(("start", self.name))

    def stop(self, grace=None):
        FakeService.log.append(("stop", self.name))

    def kill(self):
        FakeService.log.append(("kill", self.name))

    def exited(self):
        return self.proc.returncode is not None


@pytest.fixture
def launcher(monkeypatch):
    FakeService.log = []
    runs = []

    def run(cmd, **kwargs):
        runs.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    launcher = run_all.Launcher(service_factory=FakeService, run=run)
    launcher.runs = runs
    return launcher


def test_services_stop_in_reverse_start_order(launcher):
    for name in ("broker", "consumer", "gateway", "dashboard"):
        launcher.start(name, ["cmd"], lambda: True, timeout=1)
    launcher.stop_all()
    stops = [name for action, name in FakeService.log if action == "stop"]
    assert stops == ["dashboard", "gateway", "consumer", "broker"]


def test_broker_is_stopped_only_if_run_all_started_it(launcher, monkeypatch):
    monkeypatch.setattr(run_all, "broker_ready", lambda: True)
    launcher.start_broker()  # `docker compose ps` output empty: not running
    assert ["docker", "compose", "up", "-d", "mosquitto"] in launcher.runs
    launcher.stop_all()
    assert launcher.runs[-1] == ["docker", "compose", "stop", "mosquitto"]


def test_running_broker_is_left_running(launcher, monkeypatch):
    monkeypatch.setattr(run_all, "broker_ready", lambda: True)
    launcher.run = lambda cmd, **kw: (
        launcher.runs.append(cmd) or
        SimpleNamespace(returncode=0, stdout="abc123\n", stderr=""))
    launcher.start_broker()
    launcher.stop_all()
    assert not any(cmd[:3] == ["docker", "compose", "up"] or
                   cmd[:3] == ["docker", "compose", "stop"]
                   for cmd in launcher.runs)


def test_running_mlflow_is_not_started(launcher, monkeypatch):
    monkeypatch.setattr(run_all, "mlflow_ready", lambda: True)
    launcher.start_mlflow()
    assert launcher.services == []


def test_second_ctrl_c_kills_remaining_services(launcher):
    for name in ("consumer", "gateway"):
        launcher.start(name, ["cmd"], lambda: True, timeout=1)

    def interrupted(grace=None):
        raise KeyboardInterrupt
    launcher.services[1].stop = interrupted
    launcher.stop_all()
    kills = [name for action, name in FakeService.log if action == "kill"]
    assert kills == ["gateway", "consumer"]


def test_supervise_reports_a_service_that_exits(launcher):
    launcher.start("gateway", ["cmd"], lambda: True, timeout=1)
    launcher.services[0].proc.returncode = 3
    assert launcher.supervise(poll=0) == 1


def test_child_env_is_unbuffered_and_has_src_on_path():
    env = run_all.child_env()
    assert env["PYTHONUNBUFFERED"] == "1"
    first = env["PYTHONPATH"].split(os.pathsep)[0]
    assert Path(first) == Path("src").resolve()
