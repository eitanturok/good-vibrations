"""
Launch move_data, run_pclk, and upload all in the background.

Usage:
    python run.py --experiment-name experiment-10
"""

import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from _01_move_data import build_process as build_move
from _02_run_pclk  import run_pclk
from _03_upload    import build_process as build_upload


LOCAL_BASE  = Path(r'C:\Users\eitanturok\experiments')
SHARED_BASE = Path(r'Q:\mark_sheinin_lab\DATA')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', default='experiment-10')
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

    threads = [
        threading.Thread(target=run_pclk,    args=(shared_dir,),                                                       daemon=True),
        threading.Thread(target=build_move(  local_dir, shared_dir),                                                   daemon=True),
        threading.Thread(target=build_upload(shared_dir, args.hf_dataset, args.left, args.right, args.up, args.down), daemon=True),
    ]

    for t in threads: t.start()
    for t in threads: t.join()


if __name__ == '__main__':
    main()
