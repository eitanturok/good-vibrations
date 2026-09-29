"""Change the `layout` (and optionally other metadata fields) of every sample at a range of positions.

The layout lives in exactly one place: the {"layout": ...} line of samples/<id>/metadata.jsonl.
Everything downstream reads it from there (dataset.collect_samples -> meta["layout"], e.g. the
empty-box reference picks EMPTY_BOX_LAYOUT samples), and hash_samples hashes the full metadata,
so any MDS dir built from the old metadata is invalidated automatically and rebuilt on the next run.

Only the targeted lines are rewritten; every other byte (incl. CRLF) is kept. Positions are resolved
through positions.jsonl, and each sample's own {"position_id": ...} must agree.

    # dry run (default): shows what would change
    python scripts/set_layout.py experiments/X --positions 110-150 300 --layout grid-2
    # also change other fields at the same time (value parsed as JSON, else kept as a string)
    python scripts/set_layout.py experiments/X --positions 110-150 --layout grid-2 \
        --set description="Two cubes, grid 2" --set n_objects=2 --apply
    python scripts/set_layout.py experiments/X --undo experiments/X/relayout/<stamp>

--apply first copies every metadata.jsonl it will touch to experiments/X/relayout/<stamp>/, so
--undo can restore them byte-for-byte.
"""
import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


def rd(p: Path) -> str:
    with open(p, newline="") as fh:  # the recorder writes CRLF on Windows; keep every byte as-is
        return fh.read()


def wr(p: Path, text: str):
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        fh.write(text)
    shutil.copystat(p, tmp)
    os.replace(tmp, p)  # atomic, and breaks any hard link instead of editing a shared inode


def parse_ranges(specs: list[str]) -> set[int]:
    out = set()
    for s in specs:
        for part in s.split(","):
            lo, _, hi = part.partition("-")
            out |= set(range(int(lo), int(hi or lo) + 1))
    return out


def parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def read_positions(exp: Path) -> dict[int, list[str]]:
    rows = {}
    for line in rd(exp / "positions.jsonl").splitlines():
        if line.strip():
            ((k, v),) = json.loads(line).items()
            rows[int(k)] = v
    return rows


def rewrite(text: str, updates: dict) -> tuple[str, dict]:
    """Swap the single-key line of each field in `updates`. Returns (text, {key: old value})."""
    lines, old = text.splitlines(keepends=True), {}
    for i, line in enumerate(lines):
        if not line.strip(): continue
        d = json.loads(line)
        if len(d) == 1 and (k := next(iter(d))) in updates:
            assert k not in old, f"{k!r} appears twice"
            assert json.dumps(d) in line, line
            old[k] = d[k]
            lines[i] = line.replace(json.dumps(d), json.dumps({k: updates[k]}), 1)
    if missing := updates.keys() - old.keys():
        raise KeyError(f"no line for {sorted(missing)}")
    return "".join(lines), old


def plan(exp: Path, positions: set[int], updates: dict) -> list[tuple[Path, str, dict]]:
    pos = read_positions(exp)
    if unknown := sorted(positions - pos.keys()):
        raise SystemExit(f"positions not in positions.jsonl: {unknown[:20]}{' ...' if len(unknown) > 20 else ''}")
    out, no_dir, problems = [], [], []
    for p in sorted(positions):
        for sid in pos[p]:
            meta = exp / "samples" / sid / "metadata.jsonl"
            if not meta.exists():
                no_dir.append(sid); continue
            text = rd(meta)
            d = {k: v for l in text.splitlines() if l.strip() for k, v in json.loads(l).items()}
            if d.get("position_id") != p:
                problems.append(f"{sid}: metadata position_id={d.get('position_id')} but positions.jsonl says {p}"); continue
            try:
                new, old = rewrite(text, updates)
            except (KeyError, AssertionError) as e:
                problems.append(f"{sid}: {e}"); continue
            out.append((meta, new, old))
    if problems:
        raise SystemExit("PLAN FAILED:\n  " + "\n  ".join(problems[:50]))
    if no_dir: print(f"  {len(no_dir)} ids in positions.jsonl without a sample dir (skipped): {no_dir[:10]}")
    return out


def summarize(rows, updates):
    changed = [r for r in rows if any(r[2][k] != v for k, v in updates.items())]
    print(f"{len(rows)} samples at the given positions, {len(changed)} would change")
    for k, v in updates.items():
        before = {}
        for _, _, old in rows: before[json.dumps(old[k])] = before.get(json.dumps(old[k]), 0) + 1
        print(f"  {k}: {', '.join(f'{b} (x{n})' for b, n in sorted(before.items()))}  ->  {json.dumps(v)}")
    return changed


def apply(exp: Path, rows, changed):
    j = exp / "relayout" / datetime.now().strftime("%Y%m%d-%H%M%S")
    j.mkdir(parents=True)
    for meta, _, _ in changed:  # journal first: originals of everything we rewrite
        dst = j / meta.parent.name / "metadata.jsonl"
        dst.parent.mkdir()
        shutil.copy2(meta, dst)
    for meta, new, _ in changed:
        wr(meta, new)
    bad = [m for m, new, _ in changed if rd(m) != new]
    print(f"rewrote {len(changed) - len(bad)}/{len(changed)} metadata.jsonl; originals in {j}")
    if bad: raise SystemExit(f"VERIFY FAILED: {bad[:10]}")


def undo(exp: Path, j: Path):
    n = 0
    for orig in sorted(j.glob("*/metadata.jsonl")):
        wr(exp / "samples" / orig.parent.name / "metadata.jsonl", rd(orig)); n += 1
    print(f"restored {n} metadata.jsonl from {j}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", type=Path, help="experiment dir (holds samples/ and positions.jsonl)")
    ap.add_argument("--positions", nargs="+", help="position ids / inclusive ranges: 110-150 200 300,305-310")
    ap.add_argument("--layout", help="new layout value")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="other metadata fields to change too")
    ap.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    ap.add_argument("--undo", type=Path, metavar="JOURNAL_DIR", help="restore from a relayout/<stamp> dir")
    a = ap.parse_args()
    if a.undo:
        undo(a.exp, a.undo); raise SystemExit
    if not a.positions: ap.error("--positions is required")
    updates = ({"layout": a.layout} if a.layout is not None else {}) | \
              {k: parse_value(v) for k, _, v in (s.partition("=") for s in a.set)}
    if not updates: ap.error("nothing to change: pass --layout and/or --set")
    rows = plan(a.exp, parse_ranges(a.positions), updates)
    changed = summarize(rows, updates)
    if a.apply and changed: apply(a.exp, rows, changed)
    elif not a.apply: print("dry run; pass --apply to write")
