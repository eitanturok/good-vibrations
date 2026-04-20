import argparse
import shlex
import subprocess
import time


REMOTE_HOST = "ethantu@mcluster11.wisdom.weizmann.ac.il"
REMOTE_REPO = "mark_sheinin_lab/code/eitan/good-vibrations"
REMOTE_UV_INSTALL = "curl -LsSf https://astral.sh/uv/install.sh | sh"
REMOTE_UV = "/home/ethantu/.local/bin/uv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run laser-vibrations backfill on mcluster11.")
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--old-repo-id", default="eturok-weizmann/vibrations")
    parser.add_argument("--new-repo-id", default="eturok-weizmann/laser-vibrations")
    parser.add_argument("--source-data-root", default="/net/mraid20/ifs/wisdom/groups/mark_sheinin_lab/DATA")
    parser.add_argument("--left", type=float, default=0.15)
    parser.add_argument("--right", type=float, default=0.67)
    parser.add_argument("--up", type=float, default=0.08)
    parser.add_argument("--down", type=float, default=0.7)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--lowcut", type=float, default=50.0)
    parser.add_argument("--highcut", type=float, default=1000.0)
    parser.add_argument("--filter-order", type=int, default=5)
    parser.add_argument("--hann-applied", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-freq", type=float, default=50.0)
    parser.add_argument("--max-freq", type=float, default=1000.0)
    parser.add_argument("--fft-audio-laser-idx", type=int, default=50)
    parser.add_argument("--fft-audio-xy-idx", type=int, default=0)
    parser.add_argument("--fft-audio-out-sr", type=int, default=22050)
    parser.add_argument("--create-pr", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    remote_args = [
        REMOTE_UV,
        "run",
        "--no-project",
        "--with", "numpy",
        "--with", "scipy",
        "--with", "opencv-python-headless",
        "--with", "datasets",
        "--with", "huggingface_hub",
        "--with", "pillow",
        "python", "scripts/backfill_laser_vibrations.py",
        "--sample-id", str(args.sample_id),
        "--old-repo-id", args.old_repo_id,
        "--new-repo-id", args.new_repo_id,
        "--source-data-root", args.source_data_root,
        "--left", str(args.left),
        "--right", str(args.right),
        "--up", str(args.up),
        "--down", str(args.down),
        "--lowcut", str(args.lowcut),
        "--highcut", str(args.highcut),
        "--filter-order", str(args.filter_order),
        "--min-freq", str(args.min_freq),
        "--max-freq", str(args.max_freq),
        "--fft-audio-laser-idx", str(args.fft_audio_laser_idx),
        "--fft-audio-xy-idx", str(args.fft_audio_xy_idx),
        "--fft-audio-out-sr", str(args.fft_audio_out_sr),
        "--auto-discover-source",
        "--create-pr" if args.create_pr else "--no-create-pr",
        "--hann-applied" if args.hann_applied else "--no-hann-applied",
    ]
    if args.prompt is not None:
        remote_args.extend(["--prompt", args.prompt])

    remote_cmd = (
        f"{REMOTE_UV_INSTALL} && "
        f"cd {shlex.quote(REMOTE_REPO)} && "
        f"{' '.join(shlex.quote(a) for a in remote_args)}"
    )

    t0 = time.perf_counter()
    subprocess.run(["ssh", REMOTE_HOST, remote_cmd], check=True)
    print(f"[timing] remote backfill total: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
