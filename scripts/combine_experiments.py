"""Combine experiment parts (e.g. a recording split across drives) into one experiment dir.

Later parts' sample ids / position ids are shifted so they continue right after the previous
part's (part 2 of gastronorm_four_objs restarted at sample 000016 / position 4, colliding with
part 1). Ids live in exactly three places, all rewritten here:
  1. samples/<sample_id>/            -- the dir name
  2. samples/*/metadata.jsonl        -- the {"sample_id": ...} and {"position_id": ...} lines
  3. positions.jsonl                 -- {"<position_id>": ["<sample_id>", ...]}
Every other file is hard-linked (no extra disk; the parts stay intact) or copied with --copy.
failed_samples.jsonl is a watcher log of Windows paths that don't map to part samples
(e.g. part 2 logs 000001-000015 from a deleted attempt), so it is concatenated verbatim.
`experiment_dir` in metadata is left as-is (it records where each sample was recorded).
An id_map.jsonl {"part", "old_sample_id", "new_sample_id", "old_position_id", "new_position_id"}
is written to the output dir.

    python scripts/combine_experiments.py combine experiments/A_part_1 experiments/A_part_2 --out experiments/A
    python scripts/combine_experiments.py verify  experiments/A_part_1 experiments/A_part_2 --out experiments/A
"""
import argparse
import json
import os
import shutil
from pathlib import Path


# the parts were recorded on Windows (CRLF); newline="" keeps every byte as-is
def rd(p: Path) -> str:
    with open(p, newline="") as fh:
        return fh.read()


def wr(p: Path, text: str):
    with open(p, "w", newline="") as fh:
        fh.write(text)


def read_positions(part: Path) -> list[tuple[int, list[str]]]:
    rows = []
    for line in rd(part / "positions.jsonl").splitlines():
        if line.strip():
            ((k, v),) = json.loads(line).items()
            rows.append((int(k), v))
    return rows


def plan(parts: list[Path]) -> list[dict]:
    """One row per sample id named in positions.jsonl or present as a dir, with its new ids."""
    rows, next_sample, next_position = [], 1, 1
    for i, part in enumerate(parts):
        positions = read_positions(part)
        s2p = {s: p for p, ss in positions for s in ss}
        dirs = {d.name for d in (part / "samples").iterdir() if d.is_dir()}
        assert dirs <= s2p.keys(), f"{part}: sample dirs missing from positions.jsonl: {sorted(dirs - s2p.keys())[:10]}"
        # shift so this part's smallest id lands on the next free id (keeps gaps inside a part)
        s_off = next_sample - min(int(s) for s in s2p)
        p_off = next_position - min(p for p, _ in positions)
        for s, p in sorted(s2p.items()):
            rows.append(dict(part=i + 1, src=part, old_sample_id=s, new_sample_id=f"{int(s) + s_off:06d}",
                             old_position_id=p, new_position_id=p + p_off, has_dir=s in dirs))
        next_sample = max(int(r["new_sample_id"]) for r in rows) + 1
        next_position = max(r["new_position_id"] for r in rows) + 1
    new_ids = [r["new_sample_id"] for r in rows]
    assert len(new_ids) == len(set(new_ids)), "new sample ids collide"
    return rows


def rewrite_metadata(text: str, r: dict) -> str:
    """Replace only the sample_id / position_id lines; every other byte is kept."""
    out, seen = [], set()
    for line in text.splitlines(keepends=True):
        d = json.loads(line) if line.strip() else {}
        if list(d) == ["sample_id"]:
            assert d["sample_id"] == r["old_sample_id"], (r, line)
            assert json.dumps(d) in line, line
            line = line.replace(json.dumps(d), json.dumps({"sample_id": r["new_sample_id"]}), 1)
            seen.add("sample_id")
        elif list(d) == ["position_id"]:
            assert d["position_id"] == r["old_position_id"], (r, line)
            assert json.dumps(d) in line, line
            line = line.replace(json.dumps(d), json.dumps({"position_id": r["new_position_id"]}), 1)
            seen.add("position_id")
        out.append(line)
    assert seen == {"sample_id", "position_id"}, (r, seen)
    return "".join(out)


def combine(parts: list[Path], out: Path, copy: bool):
    assert not out.exists(), f"{out} already exists"
    rows = plan(parts)
    missing = [(r["part"], r["old_sample_id"]) for r in rows if not r["has_dir"]]
    print(f"{len(rows)} samples, {len(missing)} named in positions.jsonl without a dir: {missing[:20]}")
    link = shutil.copy2 if copy else os.link
    (out / "samples").mkdir(parents=True)
    for n, r in enumerate(rows):
        if not r["has_dir"]:
            continue
        src, dst = r["src"] / "samples" / r["old_sample_id"], out / "samples" / r["new_sample_id"]
        for f in sorted(src.rglob("*")):
            g = dst / f.relative_to(src)
            if f.is_dir():
                g.mkdir(parents=True, exist_ok=True)
            elif f.name == "metadata.jsonl" and f.parent == src:
                g.parent.mkdir(parents=True, exist_ok=True)
                wr(g, rewrite_metadata(rd(f), r))
                shutil.copystat(f, g)
            else:
                g.parent.mkdir(parents=True, exist_ok=True)
                link(f, g)
        if n % 500 == 0:
            print(f"  {n}/{len(rows)} {r['src'].name}/{r['old_sample_id']} -> {r['new_sample_id']}", flush=True)
    # positions.jsonl in original order (parts in order), with new ids
    by_pos: dict[tuple, list[str]] = {}
    for r in rows:
        by_pos.setdefault((r["part"], r["old_position_id"]), []).append(r["new_sample_id"])
    new_pos = {(r["part"], r["old_position_id"]): r["new_position_id"] for r in rows}
    new_sid = {(r["part"], r["old_sample_id"]): r["new_sample_id"] for r in rows}
    eol = "\r\n" if rd(parts[0] / "positions.jsonl").endswith("\r\n") else "\n"
    with open(out / "positions.jsonl", "w", newline="") as fh:
        for i, part in enumerate(parts):
            for p, ss in read_positions(part):
                new_ss = [new_sid[(i + 1, s)] for s in ss]
                assert new_ss == by_pos[(i + 1, p)]
                fh.write(json.dumps({str(new_pos[(i + 1, p)]): new_ss}) + eol)
    with open(out / "failed_samples.jsonl", "w", newline="") as fh:
        for part in parts:
            if (part / "failed_samples.jsonl").exists():
                text = rd(part / "failed_samples.jsonl")
                fh.write(text if text.endswith("\n") or not text else text + eol)
    with open(out / "id_map.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps({"part": r["src"].name, **{k: r[k] for k in
                     ("old_sample_id", "new_sample_id", "old_position_id", "new_position_id")}}) + "\n")
    print(f"wrote {out}")


def verify(parts: list[Path], out: Path) -> bool:
    """Every part file has a counterpart in out: byte-identical, except metadata.jsonl (only the two
    id lines differ, and they hold the new ids) -- and out has nothing else."""
    rows, errors, n_files = plan(parts), [], 0
    expected = {"positions.jsonl", "failed_samples.jsonl", "id_map.jsonl", "COMBINE_NOTES.md"}
    for r in rows:
        if not r["has_dir"]:
            continue
        src, dst = r["src"] / "samples" / r["old_sample_id"], out / "samples" / r["new_sample_id"]
        for f in sorted(p for p in src.rglob("*") if p.is_file()):
            rel = f.relative_to(src)
            g = dst / rel
            expected.add(str(g.relative_to(out)))
            n_files += 1
            if not g.is_file():
                errors.append(f"missing {g}")
            elif f.name == "metadata.jsonl" and f.parent == src:
                a, b = rd(f).split("\n"), rd(g).split("\n")  # keeps any \r, so CRLF loss is caught
                diff = [(x, y) for x, y in zip(a, b) if x != y]
                want = [(x, x.replace(o, n, 1)) for x in a for o, n in
                        [(json.dumps({"sample_id": r["old_sample_id"]}), json.dumps({"sample_id": r["new_sample_id"]})),
                         (json.dumps({"position_id": r["old_position_id"]}), json.dumps({"position_id": r["new_position_id"]}))]
                        if x.rstrip("\r") == o and o != n]
                if len(a) != len(b) or diff != want:
                    errors.append(f"metadata {src} -> {g}: {diff} != {want}")
            elif not (os.path.samefile(f, g) or f.read_bytes() == g.read_bytes()):
                errors.append(f"content differs {f} vs {g}")
    extra = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()} - expected
    if extra:
        errors.append(f"{len(extra)} unexpected files in out: {sorted(extra)[:10]}")
    # positions.jsonl: same shape as the parts' concatenation, with ids mapped
    m = {(r["part"], r["old_sample_id"]): r["new_sample_id"] for r in rows}
    mp = {(r["part"], r["old_position_id"]): r["new_position_id"] for r in rows}
    want_pos = [(mp[(i + 1, p)], [m[(i + 1, s)] for s in ss]) for i, part in enumerate(parts) for p, ss in read_positions(part)]
    eol = "\r\n" if rd(parts[0] / "positions.jsonl").endswith("\r\n") else "\n"
    if rd(out / "positions.jsonl") != "".join(json.dumps({str(p): ss}) + eol for p, ss in want_pos):
        errors.append("positions.jsonl mismatch")
    got_ids = [s for _, ss in read_positions(out) for s in ss]
    if len(got_ids) != len(set(got_ids)) or len({p for p, _ in read_positions(out)}) != len(want_pos):
        errors.append("duplicate ids in combined positions.jsonl")
    fails = "".join(rd(p / "failed_samples.jsonl") for p in parts if (p / "failed_samples.jsonl").exists())
    if rd(out / "failed_samples.jsonl") != fails:
        errors.append("failed_samples.jsonl mismatch")
    for part in parts:
        print(f"  {part.name}: {sum(r['src'] == part and r['has_dir'] for r in rows)} sample dirs, "
              f"{len(read_positions(part))} positions")
    print(f"  {out.name}: {len(list((out / 'samples').iterdir()))} sample dirs, {len(read_positions(out))} positions, "
          f"{n_files} files checked")
    for e in errors[:50]:
        print("  ERROR", e)
    print("VERIFY OK" if not errors else f"VERIFY FAILED ({len(errors)} errors)")
    return not errors


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["combine", "verify"])
    ap.add_argument("parts", nargs="+", type=Path, help="part dirs, in recording order")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--copy", action="store_true", help="copy files instead of hard-linking")
    a = ap.parse_args()
    if a.cmd == "combine":
        combine(a.parts, a.out, a.copy)
    raise SystemExit(0 if verify(a.parts, a.out) else 1)
