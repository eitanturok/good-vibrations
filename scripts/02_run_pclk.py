import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from watch import watch
from status import claim, finish

REMOTE_HOST     = 'mcluster11'
REMOTE_CODE_DIR = 'mark_sheinin_lab/code'
MAX_JOBS        = 5


def to_remote_path(windows_path):
    """Convert a Windows path (Q:\foo\bar) to a remote Linux path (foo/bar)."""
    return '/'.join(Path(windows_path).parts[1:])


def build_process(shared_dir):
    @watch(shared_dir)
    def process(sample_path):
        result = claim(sample_path, "run_pclk", prerequisite="move_data")
        if result == "waiting": return False
        out_log = sample_path / "out.log"
        if result == "claimed" and not out_log.exists():
            n_jobs = int(subprocess.run(
                ['ssh', REMOTE_HOST, 'squeue -u $USER -h | wc -l'],
                capture_output=True, text=True, check=True
            ).stdout.strip())
            if n_jobs >= MAX_JOBS:
                print(f"Queue full ({n_jobs} jobs), will retry {sample_path.name} later...")
                return False
            print(f"Submitting pclk job for {sample_path.name}...")
            remote_path = '~/' + to_remote_path(sample_path)
            try:
                subprocess.run([
                    'ssh', REMOTE_HOST,
                    f'cd {REMOTE_CODE_DIR} && ./run_pclk.sh {remote_path}'
                ], check=True, timeout=5)
            except subprocess.TimeoutExpired:
                pass  # job was submitted; run_pclk.sh just never exits due to a bash subshell bug
        if not out_log.exists():
            return False
        last_line = out_log.read_text().strip().splitlines()[-1]
        if last_line != "Done.":
            return False
        finish(sample_path, "run_pclk")
        return True
    return process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-dir',  required=True, type=Path)
    parser.add_argument('--shared-dir', required=True, type=Path)
    args = parser.parse_args()

    process = build_process(args.shared_dir)
    process()


if __name__ == '__main__':
    main()
