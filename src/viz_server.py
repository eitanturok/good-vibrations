"""Launch the viz2 dashboard in a detached tmux session, once."""

import os, shlex, shutil, socket, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_viz(experiment_dir, port=8504, host="127.0.0.1", runs_dir=None, enabled=True):
    """Start viz2 in tmux if it isn't already up. Returns the session name, or None."""
    if not enabled or os.environ.get("MODAL_TASK_ID") or shutil.which("tmux") is None: return None

    # The port is the source of truth, not the session: a session can outlive a viz2 that died
    # (e.g. on EADDRINUSE), and a hand-started viz2 can hold the port with no session at all.
    session = f"viz2-{port}"
    with socket.socket() as s:
        s.settimeout(0.5)
        if s.connect_ex((host, port)) == 0:
            print(f"[viz2] already serving http://{host}:{port} -- reusing")
            return session

    # Port is free, so any same-named session is a dead one; clear it or new-session will fail.
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)

    inner = " ".join(shlex.quote(a) for a in [
        sys.executable, "-m", "viz2",
        "--experiment", str(Path(experiment_dir).resolve()),
        "--runs", str(runs_dir or REPO_ROOT / "runs"),
        "--port", str(port), "--host", host])
    # PYTHONPATH must go in the command, not env=: tmux only applies our environment when it has to
    # start a server. The trailing read keeps the pane open so a crash is visible in `tmux attach`.
    cmd = f"PYTHONPATH={shlex.quote(str(REPO_ROOT))} {inner}; echo '[viz2] exited'; read _"
    try:
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(REPO_ROOT), cmd], check=True)
    except Exception as e:  # a broken dashboard must never take down a training job
        print(f"[viz2] could not start ({e}); continuing without it")
        return None

    print(f"[viz2] http://{host}:{port}  (tmux attach -t {session})")
    return session
