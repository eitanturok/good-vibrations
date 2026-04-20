import argparse
import textwrap
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "eturok-weizmann/laser-vibrations"


def stage(label, fn):
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    print(f"[timing] {label}: {dt:.2f}s")
    return result


def build_readme() -> str:
    return textwrap.dedent(
        """\
        ---
        pretty_name: Laser Vibrations
        ---

        # Laser Vibrations

        This dataset is the next-generation pipeline dataset for the Good Vibrations project.

        It is designed to store:
- sample metadata in parquet shards under `data/`
- shared audio assets under `audio/`
- per-sample pipeline artifacts under `samples/sample_xxxxxx/`

        Target sample assets include:
- `mask.npz`
- `speckle_vibrations_raw.npy`
- `speckle_vibrations.mp4`
- `speckle_shifts.npz`
- `speckle_shifts_clean.npz`
- `speckle_shifts_fft.npz`
- `speckle_shifts_fft_audio.wav`
- `manifest.json`

        This repository is being initialized incrementally. The first goal is to validate the schema and viewer behavior with one dummy sample before backfilling historical data.
        """
    )


def build_skeleton(root: Path) -> None:
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    (root / "samples").mkdir(parents=True, exist_ok=True)

    (root / "README.md").write_text(build_readme(), encoding="utf-8")
    (root / "data" / ".gitkeep").write_text("", encoding="utf-8")
    (root / "audio" / ".gitkeep").write_text("", encoding="utf-8")
    (root / "samples" / ".gitkeep").write_text("", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the laser-vibrations dataset scaffold.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--create-pr", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    api = HfApi()

    with TemporaryDirectory(prefix="laser-vibrations-init-") as tmp:
        root = Path(tmp)
        stage("build_skeleton", lambda: build_skeleton(root))
        stage(
            "upload_folder",
            lambda: api.upload_folder(
                folder_path=str(root),
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message="Initialize laser-vibrations dataset scaffold",
                create_pr=args.create_pr,
            ),
        )

    print(f"[done] initialized dataset scaffold: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
