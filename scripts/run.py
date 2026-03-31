"""
Launch move_data, run_pclk, and upload all in the background.

Usage:
    python run.py --experiment-name experiment-10
"""

import argparse
import subprocess
import threading
from pathlib import Path


LOCAL_BASE  = Path(r'C:\Users\eitanturok\experiments')
SHARED_BASE = Path(r'Q:\mark_sheinin_lab\DATA')
REMOTE_HOST     = 'mcluster11'
REMOTE_CODE_DIR = 'mark_sheinin_lab/code'
REMOTE_DATA_DIR = 'mark_sheinin_lab/DATA'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', default='experiment-12')
    parser.add_argument('--delete',           action='store_true', default=False, help='Delete local data after moving to shared dir')
    parser.add_argument('--hf-dataset',      default='eturok-weizmann/vibrations')
    parser.add_argument('--left',            type=float, default=0.15)
    parser.add_argument('--right',           type=float, default=0.67)
    parser.add_argument('--up',              type=float, default=0.08)
    parser.add_argument('--down',            type=float, default=0.7)
    args = parser.parse_args()

    local_dir  = LOCAL_BASE  / args.experiment_name
    shared_dir = SHARED_BASE / args.experiment_name

    local_dir.mkdir(parents=True, exist_ok=True)
    shared_dir.mkdir(parents=True, exist_ok=True)

    remote_dir = f'~/{REMOTE_DATA_DIR}/{args.experiment_name}'

    def run_pclk():
        subprocess.run([
            'ssh', REMOTE_HOST,
            f'cd {REMOTE_CODE_DIR} && ./run_pclk_eitan.sh {remote_dir}'
        ])

    t = threading.Thread(target=run_pclk, daemon=True)
    t.start()
    print(f"Started pclk job submission thread for {remote_dir}")

    scripts  = Path(__file__).parent
    root_dir = str(scripts.parent)
    subprocess.Popen([
        'wt',
        'new-tab', '--startingDirectory', root_dir, '--title', 'move_data', 'uv', 'run', 'python', str(scripts / '01_move_data.py'), str(local_dir), str(shared_dir), *(['--delete'] if args.delete else []),
        ';',
        'new-tab', '--startingDirectory', root_dir, '--title', 'squeue',    'ssh', '-t', REMOTE_HOST, 'watch -n 2 squeue -u $USER',
        ';',
        'new-tab', '--startingDirectory', root_dir, '--title', 'upload',    'uv', 'run', 'python', str(scripts / '03_upload.py'),    '--shared-dir', str(shared_dir), '--hf-dataset', args.hf_dataset, '--left', str(args.left), '--right', str(args.right), '--up', str(args.up), '--down', str(args.down),
    ])

    t.join()


if __name__ == '__main__':
    main()
