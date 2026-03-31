import argparse
import subprocess
from pathlib import Path

REMOTE_HOST     = 'mcluster11'
REMOTE_CODE_DIR = 'mark_sheinin_lab/code'


def run_pclk(shared_dir):
    subprocess.run([
        'ssh', REMOTE_HOST,
        f'cd {REMOTE_CODE_DIR} && mkdir -p {shared_dir} && ./run_pclk.sh {shared_dir}'
    ], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-dir',  required=True, type=Path, help='Local dir with raw experiment results')
    parser.add_argument('--shared-dir', required=True, type=Path, help='Mounted shared dir accessible by both local machine and mcluster11')
    args = parser.parse_args()

    run_pclk(args.shared_dir)


if __name__ == '__main__':
    main()
