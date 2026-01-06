import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 1. Setup

    Launch with `marimo edit --watch src/08_explore.py`
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from pathlib import Path
    import sys, types

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    return Path, np, pd, sys, types


@app.cell
def _(sys, types):
    # Patch for pickle compatibility
    # if 'recover_core_lib' not in sys.modules:
    print('adding recover_core_lib to sys')
    fake = types.ModuleType('recover_core_lib')
    fake.compute_CAM2_translations_v3_cupy = lambda *a, **k: None
    sys.modules['recover_core_lib'] = fake
    return


@app.cell
def _(np):
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 2. Load Data
    """)
    return


@app.cell
def _(Path):
    BASE_DIR = Path('data/experiment_01/')
    exp_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir()])
    exp_dirs
    return (exp_dirs,)


@app.cell
def _(Path, np):
    def load_experiment(exp_dir: Path, idx:int) -> dict:
        """Load recovery and metadata from an experiment directory."""
        recovery = np.load(exp_dir / 'RECOVERY.npz', allow_pickle=True)
        # run_opt = recovery['run_opt'].item()

        object = str(exp_dir).split('/')[-1].split('_')[0]
        position = int(str(exp_dir).split('/')[-1].split('_')[1][3:])
    
        metadata = {
            'name': f'{object}-pos{position}-{idx}',
            'object': object,
            'position': position,
            # 'fps': run_opt['cam_params']['camera_FPS'],
            'fps': 5_000,
            'path': exp_dir
        }
        raw_shifts = recovery['all_shifts']
        return metadata, raw_shifts
    return (load_experiment,)


@app.cell
def _(exp_dirs, load_experiment):
    experiments = [load_experiment(d, idx) for idx, d in enumerate(exp_dirs)]
    raw_shifts = {exp[0]['name']: exp[1] for exp in experiments}
    raw_shifts
    return experiments, raw_shifts


@app.cell
def _(experiments, pd):
    df = pd.DataFrame([exp[0] for exp in experiments])
    df
    return (df,)


@app.cell
def _(df, np, raw_shifts):
    timesteps = {name: np.arange(raw_shift.shape[1]) / df[df['name'] == name]['fps'] for name, raw_shift in raw_shifts.items()}
    timesteps
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
