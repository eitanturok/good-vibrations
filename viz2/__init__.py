"""Launch the viz2 dashboard in a detached tmux session, once.

Kept import-light: src/run.py imports ensure_viz at startup, and pulling in app/data there
would drag torch and fastapi into the training process for a dashboard it only spawns.
"""

import os, shlex, shutil, socket, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tmux(*args):
    return subprocess.run(["tmux", *args], capture_output=True, text=True)


def _serving(host, port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def ensure_viz(experiment_dir, port=8504, host="127.0.0.1", runs_dir=None, enabled=True):
    """Start viz2 in tmux if it isn't already up on this experiment. Returns the session, or None."""
    if not enabled or os.environ.get("MODAL_TASK_ID") or shutil.which("tmux") is None: return None

    session = f"viz2-{port}"
    cmd = " ".join(shlex.quote(a) for a in [
        sys.executable, "-m", "viz2",
        "--experiment", str(Path(experiment_dir).resolve()),
        "--runs", str(runs_dir or REPO_ROOT / "runs"),
        "--port", str(port), "--host", host])

    # Reuse only if the live server was started with this exact command -- otherwise it is
    # serving a different experiment and would quietly show the wrong dataset.
    if _serving(host, port) and cmd in _tmux("list-panes", "-t", session, "-F", "#{pane_start_command}").stdout:
        print(f"[viz2] already serving http://{host}:{port} -- reusing")
        return session

    _tmux("kill-session", "-t", session)  # a dead session, or a live one on another experiment
    for _ in range(25):  # kill-session returns before the listener is gone
        if not _serving(host, port): break
        time.sleep(0.2)

    # PYTHONPATH must go in the command, not env=: tmux only applies our environment when it
    # has to start a server. The trailing read keeps the pane open so a crash stays readable.
    wrapped = f"PYTHONPATH={shlex.quote(str(REPO_ROOT))} {cmd}; echo '[viz2] exited'; read _"
    if _tmux("new-session", "-d", "-s", session, "-c", str(REPO_ROOT), wrapped).returncode != 0:
        print("[viz2] could not start; continuing without it")  # never take down a training job
        return None

    print(f"[viz2] http://{host}:{port}  (tmux attach -t {session})")
    return session
