"""
Streaming Stack Launcher
========================

Starts the streaming simulation's long-running services in dependency
order, waiting for each to report ready before starting the next:

    broker     docker compose up -d mosquitto   ready: MQTT CONNACK
    mlflow     mlflow ui (only if not running)  ready: GET /health == OK
    consumer   src/streaming/consumer.py        ready: retained "online"
                                                status newer than its start
    gateway    uvicorn src.app.app_api:app      ready: /stream/status says
                                                the broker is connected
    dashboard  streamlit run app_dashboard.py   ready: /_stcore/health

The producer stays a manual step; the command is printed once everything
is up.

Output: every service's stdout/stderr is shown here, prefixed with its
name, and also written to logs/<service>.log (overwritten on each run).

Shutdown: one Ctrl+C stops everything this script started, in reverse
order (Ctrl+Break to each process group, then a forced kill after
STOP_GRACE seconds; the stateless dashboard is killed directly). A second
Ctrl+C kills what is left at once. A broker
or MLflow server that was already running is left running. If any service
exits on its own, everything else is stopped too.

Usage (from the repo root, Docker Desktop running):
    PYTHONPATH=src python src/streaming/run_all.py
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt

from streaming import config
from streaming.producer import read_consumer_status

HOST = "127.0.0.1"
MLFLOW_PORT, API_PORT, DASHBOARD_PORT = 5000, 8000, 8501
MLFLOW_URL = f"http://{HOST}:{MLFLOW_PORT}"
API_URL = f"http://{HOST}:{API_PORT}"
DASHBOARD_URL = f"http://{HOST}:{DASHBOARD_PORT}"
LOG_DIR = Path("logs")
STOP_GRACE = 10.0
POLL = 0.5
PROGRESS_EVERY = 15.0
PRODUCER_CMD = ("PYTHONPATH=src python src/streaming/producer.py "
                "--interval 0.2")

COLORS = {"run_all": "1", "broker": "35", "mlflow": "34", "consumer": "33",
          "gateway": "36", "dashboard": "32"}


class StartupError(RuntimeError):
    pass


class Output:
    """Serialises prefixed lines from every service onto one stream."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.color = self.stream.isatty()
        self.lock = threading.Lock()

    def line(self, name, text):
        label = f"[{name:<9}]"
        if self.color:
            label = f"\033[{COLORS.get(name, '0')}m{label}\033[0m"
        with self.lock:
            self.stream.write(f"{label} {text}\n")
            self.stream.flush()


OUT = Output()


def say(text):
    OUT.line("run_all", text)


# --- readiness checks ------------------------------------------------------


def http_text(url, timeout=1.0):
    """Response body, or None if the request fails."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except OSError:
        return None


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, port)) == 0


def broker_ready():
    """True once the broker answers an MQTT CONNECT (not just TCP)."""
    connected = threading.Event()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="")
    client.on_connect = (lambda c, u, f, rc, p:
                         None if rc.is_failure else connected.set())
    try:
        client.connect(config.BROKER_HOST, config.BROKER_PORT, keepalive=10)
    except OSError:
        return False
    client.loop_start()
    try:
        return connected.wait(1.0)
    finally:
        client.disconnect()
        client.loop_stop()


def mlflow_ready():
    return (http_text(f"{MLFLOW_URL}/health") or "").strip() == "OK"


def consumer_online(status, since):
    """True if `status` is an "online" report published after `since`."""
    if not status or status.get("state") != "online":
        return False
    try:
        return datetime.fromisoformat(status["ts"]) >= since
    except (KeyError, TypeError, ValueError):
        return False


def gateway_ready():
    body = http_text(f"{API_URL}/stream/status")
    try:
        return bool(body) and json.loads(body)["broker_connected"] is True
    except (ValueError, KeyError):
        return False


def dashboard_ready():
    return (http_text(f"{DASHBOARD_URL}/_stcore/health") or "").strip() \
        == "ok"


def wait_until(name, check, timeout, proc=None, poll=POLL,
               clock=time.monotonic, sleep=time.sleep):
    """
    Polls check() until it returns True. Raises StartupError if `proc`
    exits first or `timeout` seconds pass. Returns the seconds waited.
    """
    start = clock()
    next_progress = start + PROGRESS_EVERY
    while True:
        if proc is not None and proc.poll() is not None:
            raise StartupError(
                f"{name} exited with code {proc.returncode} while starting;"
                f" see {LOG_DIR / (name + '.log')}")
        if check():
            return clock() - start
        now = clock()
        if now - start >= timeout:
            raise StartupError(f"{name} not ready after {timeout:.0f}s")
        if now >= next_progress:
            say(f"still waiting for {name} ({now - start:.0f}s)")
            next_progress = now + PROGRESS_EVERY
        sleep(poll)


# --- processes -------------------------------------------------------------


def child_env():
    env = dict(os.environ)
    # Pipes are block-buffered; without this, child logs arrive in bursts.
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"  # MLflow prints emoji
    # conda's MKL-linked numpy loads the Intel Fortran runtime, whose own
    # Ctrl+Break handler aborts the process ("forrtl: error (200)") before
    # Python's handler can run the consumer's cleanup.
    env["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
    src = str(Path("src").resolve())
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (src, env.get("PYTHONPATH")) if p)
    return env


class Service:
    """A child process whose output is prefixed and copied to a log file."""

    def __init__(self, name, cmd, output=OUT):
        self.name = name
        self.cmd = cmd
        self.output = output
        self.proc = None
        self.pump = None
        self.what = name       # used in stop messages
        self.graceful = True   # False: kill at once, no Ctrl+Break first

    def start(self, env):
        LOG_DIR.mkdir(exist_ok=True)
        log_file = open(LOG_DIR / f"{self.name}.log", "w", encoding="utf-8")
        kwargs = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                  if os.name == "nt" else {"start_new_session": True})
        self.proc = subprocess.Popen(
            self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=env, **kwargs)
        self.pump = threading.Thread(target=self._copy_output,
                                     args=(log_file,), daemon=True)
        self.pump.start()

    def _copy_output(self, log_file):
        with log_file:
            for raw in iter(self.proc.stdout.readline, b""):
                text = raw.decode("utf-8", "replace").rstrip("\r\n")
                log_file.write(text + "\n")
                log_file.flush()
                self.output.line(self.name, text)

    def exited(self):
        return self.proc is not None and self.proc.poll() is not None

    def _wait(self, timeout):
        """Like proc.wait, but in short steps so Ctrl+C stays responsive."""
        deadline = time.monotonic() + timeout
        while self.proc.poll() is None:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.2)
        return True

    def stop(self, grace=STOP_GRACE):
        if self.proc is None or self.exited():
            return
        say(f"stopping {self.what}")
        if not self.graceful:
            self.kill()
            self._wait(5)
            return
        try:
            if os.name == "nt":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(self.proc.pid, signal.SIGINT)
        except OSError:
            pass
        if not self._wait(grace):
            say(f"{self.name} did not stop within {grace:.0f}s; killing it")
            self.kill()
            self._wait(5)
        if self.pump is not None:
            self.pump.join(timeout=2)

    def kill(self):
        if self.proc is None or self.exited():
            return
        if os.name == "nt":  # /T: streamlit and mlflow spawn children
            subprocess.run(["taskkill", "/T", "/F", "/PID",
                            str(self.proc.pid)], capture_output=True)
        else:
            os.killpg(self.proc.pid, signal.SIGKILL)


class Launcher:
    """Starts services in order and stops only what it started."""

    def __init__(self, service_factory=Service, run=subprocess.run):
        self.service_factory = service_factory
        self.run = run
        self.env = child_env()
        self.services = []
        self.broker_started = False

    def start(self, name, cmd, ready, timeout, poll=POLL, what=None,
              graceful=True):
        say(f"starting {what or name}: {' '.join(cmd)}")
        svc = self.service_factory(name, cmd)
        svc.what, svc.graceful = what or name, graceful
        svc.start(self.env)
        self.services.append(svc)
        waited = wait_until(name, ready, timeout, svc.proc, poll=poll)
        say(f"{name} ready ({waited:.1f}s)")
        return svc

    def start_broker(self):
        running = self.run(
            ["docker", "compose", "ps", "--status", "running", "-q",
             "mosquitto"], capture_output=True, text=True).stdout.strip()
        if running:
            say("broker already running; it will be left running")
        else:
            say("starting broker: docker compose up -d mosquitto")
            result = self.run(["docker", "compose", "up", "-d", "mosquitto"],
                              capture_output=True, text=True)
            if result.returncode != 0:
                raise StartupError(
                    f"docker compose up failed: {result.stderr.strip()}")
            self.broker_started = True
        # Follows the container's log; stopping this leaves the broker up.
        self.start("broker", ["docker", "compose", "logs", "-f", "--tail",
                              "5", "mosquitto"], broker_ready, 60,
                   what="broker log stream")

    def start_mlflow(self):
        if mlflow_ready():
            say(f"MLflow already running at {MLFLOW_URL}; it will be left "
                "running")
            return
        if port_in_use(MLFLOW_PORT):
            raise StartupError(f"port {MLFLOW_PORT} is in use by something "
                               "that is not answering as MLflow")
        self.start("mlflow", [
            sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", "sqlite:///mlflow.db",
            "--host", HOST, "--port", str(MLFLOW_PORT)], mlflow_ready, 120)

    def start_consumer(self):
        since = datetime.now(timezone.utc)
        self.start("consumer",
                   [sys.executable, "src/streaming/consumer.py"],
                   lambda: consumer_online(read_consumer_status(1.0), since),
                   600, poll=0)  # read_consumer_status already waits 1s

    def start_gateway(self):
        self.start("gateway", [
            sys.executable, "-m", "uvicorn", "src.app.app_api:app",
            "--host", HOST, "--port", str(API_PORT)], gateway_ready, 60)

    def start_dashboard(self):
        self.start("dashboard", [
            sys.executable, "-m", "streamlit", "run",
            "src/app/app_dashboard.py", "--server.headless", "true",
            "--server.port", str(DASHBOARD_PORT),
            "--browser.gatherUsageStats", "false"], dashboard_ready, 120,
            graceful=False)  # stateless, and ignores Ctrl+Break for >10s

    def supervise(self, poll=POLL):
        """Blocks until a service exits on its own; returns exit code 1."""
        while True:
            for svc in self.services:
                if svc.exited():
                    say(f"{svc.name} exited unexpectedly (code "
                        f"{svc.proc.returncode}); see "
                        f"{LOG_DIR / (svc.name + '.log')}. Shutting down.")
                    return 1
            time.sleep(poll)

    def stop_all(self):
        """Reverse start order; a second Ctrl+C kills everything left."""
        try:
            for svc in reversed(self.services):
                svc.stop()
            if self.broker_started:
                say("stopping broker: docker compose stop mosquitto")
                self.run(["docker", "compose", "stop", "mosquitto"],
                         capture_output=True, text=True)
        except KeyboardInterrupt:
            say("second Ctrl+C: killing remaining processes")
            for svc in reversed(self.services):
                svc.kill()
        say("all services started by run_all are stopped")


def preflight():
    if not Path("src/streaming/run_all.py").exists():
        raise StartupError("run from the repo root")
    if shutil.which("docker") is None:
        raise StartupError("docker not found; install Docker Desktop")
    if subprocess.run(["docker", "info"], capture_output=True).returncode:
        raise StartupError("Docker is not running; start Docker Desktop")
    for name, port in (("gateway", API_PORT), ("dashboard", DASHBOARD_PORT)):
        if port_in_use(port):
            raise StartupError(
                f"port {port} ({name}) is already in use; stop the running "
                f"{name} first")


def banner():
    say("all services ready")
    say(f"  dashboard:  {DASHBOARD_URL}  (Live tab)")
    say(f"  live page:  {API_URL}/live")
    say(f"  API docs:   {API_URL}/docs   status: {API_URL}/stream/status")
    say(f"  MLflow:     {MLFLOW_URL}")
    say(f"  logs:       {LOG_DIR.resolve()}")
    say(f"start a replay in another terminal: {PRODUCER_CMD}")
    say("press Ctrl+C once to stop everything")


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    if os.name == "nt":
        os.system("")  # enables ANSI colours in the classic console
        # Ctrl+Break would otherwise kill run_all at once and orphan its
        # children (they run in their own process groups).
        signal.signal(signal.SIGBREAK, _interrupt)
    launcher = Launcher()
    code = 0
    try:
        preflight()
        launcher.start_broker()
        launcher.start_mlflow()
        launcher.start_consumer()
        launcher.start_gateway()
        launcher.start_dashboard()
        banner()
        code = launcher.supervise()
    except KeyboardInterrupt:
        say("Ctrl+C: stopping everything run_all started")
    except StartupError as e:
        say(f"startup failed: {e}")
        code = 1
    finally:
        launcher.stop_all()
    sys.exit(code)


if __name__ == "__main__":
    main()
