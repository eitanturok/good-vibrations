import argparse, hashlib, subprocess
from pathlib import Path

from utils.helpers import Timing


def hash_dir(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(path).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def send_to_remote(local_path: Path, remote_host: str, remote_path: str, verbose: bool = True):
    local_path = Path(local_path)
    remote_parent = str(Path(remote_path).parent.as_posix())

    local_hash_path = local_path.parent / f"{local_path.name}.hash.txt"
    local_hash_path.write_text(hash_dir(local_path))

    with Timing("scp: ", enter=f"Sending {local_path} to {remote_host}:{remote_path} ...", enabled=verbose):
        # inherit stdout/stderr (not captured) so scp's own progress meter streams live; scp -r
        # can't create a missing remote parent dir, so mkdir -p it over ssh first. hash.txt is
        # sent last so a remote hash.txt only appears once the transfer actually completed.
        subprocess.run(["ssh", remote_host, f"mkdir -p {remote_parent}"], check=True)
        subprocess.run(["scp", "-r", str(local_path), f"{remote_host}:{remote_path}"], check=True)
        subprocess.run(["scp", str(local_hash_path), f"{remote_host}:{remote_parent}/hash.txt"], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("local_path", type=Path)
    parser.add_argument("remote_host")
    parser.add_argument("remote_path")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    send_to_remote(args.local_path, args.remote_host, args.remote_path, verbose=not args.quiet)
