import argparse

import wandb

def get_parser():
    parser = argparse.ArgumentParser(description="Compare time/train between two logged steps of a wandb run.")
    parser.add_argument("run_name",         type=str, help="wandb run name (also used as the run id, per WandBLogger's id=run_name).")
    parser.add_argument("--entity",         type=str, default="eturok")
    parser.add_argument("--project",        type=str, default="better-tsa")
    parser.add_argument("--start-step",     type=int, default=16, help="Skip early steps, since they're inflated by the torch.compile warmup cost.")
    parser.add_argument("--end-step",       type=int, default=300)
    return parser

def time_train_at_or_before(history, step):
    """Return (actual_step, time/train) for the last logged row with _step <= step."""
    rows = [row for row in history if row["_step"] <= step and "time/train" in row]
    if not rows: return None
    row = max(rows, key=lambda r: r["_step"])
    return row["_step"], row["time/train"]

def batches_per_epoch(history):
    """Infer batches/epoch as 1 + the max time/batch_in_epoch logged (it's a 0-indexed counter)."""
    values = [row["time/batch_in_epoch"] for row in history if "time/batch_in_epoch" in row]
    if not values: return None
    return max(values) + 1

def resolve_run(api, entity, project, run_name):
    """run_name may be the wandb run id or its display name; try id first, then search by name."""
    try:
        return api.run(f"{entity}/{project}/{run_name}")
    except wandb.errors.CommError:
        matches = [r for r in api.runs(f"{entity}/{project}") if r.name == run_name]
        if not matches: raise
        if len(matches) > 1: raise RuntimeError(f"Multiple runs named {run_name!r}: {[r.id for r in matches]}")
        return matches[0]

def main():
    args = get_parser().parse_args()

    api = wandb.Api()
    run = resolve_run(api, args.entity, args.project, args.run_name)
    history = list(run.scan_history(keys=["_step", "time/train", "time/batch_in_epoch"]))

    start = time_train_at_or_before(history, args.start_step)
    end = time_train_at_or_before(history, args.end_step)

    if start is None or end is None:
        raise RuntimeError(f"Could not find time/train near step {args.start_step} and/or {args.end_step} for run {args.run_name!r}.")

    start_step, start_time = start
    end_step, end_time = end
    delta_time = end_time - start_time
    delta_steps = end_step - start_step

    delta_time_sec = delta_time * 60  # time/train is logged in minutes (RuntimeEstimator(time_unit="minutes"))

    print(f"run: {args.run_name}")
    print(f"start: step={start_step} time/train={start_time:.4f} min")
    print(f"end:   step={end_step} time/train={end_time:.4f} min")
    print(f"delta: steps={delta_steps} time/train={delta_time_sec:.4f} sec ({delta_time_sec / delta_steps:.4f} sec/step)" if delta_steps else "delta: steps=0")

    bpe = batches_per_epoch(history)
    if bpe and delta_steps:
        delta_epochs = delta_steps / bpe
        print(f"batches/epoch: {bpe}")
        print(f"delta: epochs={delta_epochs:.4f} time/train={delta_time_sec:.4f} sec ({delta_time_sec / delta_epochs:.4f} sec/epoch)")
    else:
        print("batches/epoch: could not infer (no time/batch_in_epoch logged)")

if __name__ == "__main__":
    main()
