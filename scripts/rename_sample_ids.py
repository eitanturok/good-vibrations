"""Rename an experiment's sample ids from a running counter ("000545") to "{position_id}-{speaker}"
("110-1"), IN PLACE (dir renames keep every file's inode, so no sample data is copied or rewritten).

Ids live in these places, all handled here:
  1. samples/<sample_id>/        -- the dir name (renamed)
  2. samples/*/metadata.jsonl    -- the {"sample_id": ...} line (only that line is rewritten; CRLF kept)
  3. positions.jsonl             -- {"<position_id>": [sample ids]} (same line endings)
  4. failed_samples.jsonl        -- the `samples\\<id>\\` path segment in every string field of an entry
  5. id_map.jsonl (if present)   -- gains a "renamed_sample_id" field per row

The speaker for an id comes from its metadata. An id that positions.jsonl names but that has no dir
(e.g. a crashed sample) gets it from its slot: the k-th id of a position was recorded by
speakers[k] -- `plan` checks that rule holds for every sample that does have metadata.

failed_samples.jsonl entries whose id is not a sample of this experiment are left unchanged and
listed (see UNMAPPABLE_FAILED for 2026_09_27_gastronorm_four_objs).

    python scripts/rename_sample_ids.py plan   experiments/X      # read-only: checks + preview
    python scripts/rename_sample_ids.py apply  experiments/X      # writes rename/ journal, renames, verifies
    python scripts/rename_sample_ids.py verify experiments/X      # checks against the journal
    python scripts/rename_sample_ids.py undo   experiments/X      # reverses apply using the journal

apply first writes experiments/X/rename/ (the mapping, every sample file's inode/size, metadata
hashes, and copies of the original positions.jsonl / failed_samples.jsonl / id_map.jsonl), so it can
be verified and undone. It is resumable: rerunning after a crash picks up where it stopped.
"""
import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

# failed_samples.jsonl paths that name ids which are NOT samples of the experiment. For
# 2026_09_27_gastronorm_four_objs: an earlier attempt on E: recorded positions 1-3 as 000001-000015,
# then was deleted (which is why part 2 started at 000016); its post-processing failures
# ("FileNotFoundError ... E:\eturok\2026_09_27_gastronorm_four_objs\samples\0000NN") are in the log.
# Renaming them by number would relabel them as part 1's unrelated samples 1-15.
UNMAPPABLE_FAILED = {
    "2026_09_27_gastronorm_four_objs": lambda d, sid: (
        d["item"].startswith("E:\\eturok\\2026_09_27_gastronorm_four_objs\\samples\\") and int(sid) <= 15),
}
FAILED_ID = re.compile(r"(\\samples\\)(\d{6})(\\)")  # ...\samples\000534\vibration\... (Windows paths)
FILES = ("positions.jsonl", "failed_samples.jsonl", "id_map.jsonl")


def rd(p: Path) -> str:
    with open(p, newline="") as fh:  # the recorder writes CRLF on Windows; keep every byte as-is
        return fh.read()


def wr(p: Path, text: str):
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        fh.write(text)
    os.replace(tmp, p)  # atomic: a crash leaves the old or the new file, never half of one


def eol_of(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def read_positions(exp: Path) -> list[tuple[int, list[str]]]:
    rows = []
    for line in rd(exp / "positions.jsonl").splitlines():
        if line.strip():
            ((k, v),) = json.loads(line).items()
            rows.append((int(k), v))
    return rows


def load_meta(p: Path) -> dict:
    return {k: v for line in rd(p).splitlines() if line.strip() for k, v in json.loads(line).items()}


def new_id(position_id: int, speaker: int, pos_width: int) -> str:
    return f"{position_id:0{pos_width}d}-{speaker}"


def plan(exp: Path, pos_width: int, verbose: bool = True) -> list[dict]:
    """[{old, new, position_id, speaker, has_dir}] for every id in positions.jsonl. Raises on any
    inconsistency, so apply never starts on data this script doesn't fully understand."""
    positions = read_positions(exp)
    dirs = {d.name for d in (exp / "samples").iterdir() if d.is_dir()}
    named = [s for _, ss in positions for s in ss]
    problems = []
    if len(named) != len(set(named)): problems.append("positions.jsonl names an id twice")
    if len({p for p, _ in positions}) != len(positions): problems.append("positions.jsonl has a position twice")
    if dirs - set(named): problems.append(f"sample dirs not in positions.jsonl: {sorted(dirs - set(named))[:10]}")

    rows, slot_rule_ok = [], 0
    for position_id, ids in positions:
        metas = {s: load_meta(exp / "samples" / s / "metadata.jsonl") for s in ids if s in dirs}
        speakers_lists = {tuple(m["speakers"]) for m in metas.values()}
        if len(speakers_lists) != 1:
            problems.append(f"position {position_id}: samples disagree on speakers list (or none have metadata): {speakers_lists}")
            continue
        (speakers,) = speakers_lists
        if len(ids) != len(speakers):
            problems.append(f"position {position_id}: {len(ids)} ids but speakers={list(speakers)}")
            continue
        for k, s in enumerate(ids):
            m = metas.get(s)
            if m is not None:
                if m.get("sample_id") != s: problems.append(f"{s}: metadata sample_id={m.get('sample_id')!r}")
                if m.get("position_id") != position_id: problems.append(f"{s}: metadata position_id={m.get('position_id')} != {position_id}")
                if m.get("speaker") != speakers[k]: problems.append(f"{s}: speaker={m.get('speaker')} but slot {k} of {list(speakers)} says {speakers[k]}")
                else: slot_rule_ok += 1
            speaker = m["speaker"] if m is not None else speakers[k]
            rows.append(dict(old=s, new=new_id(position_id, speaker, pos_width), position_id=position_id,
                             speaker=speaker, has_dir=s in dirs))

    dup = [k for k, v in Counter(r["new"] for r in rows).items() if v > 1]
    if dup: problems.append(f"new ids collide (a speaker recorded twice at one position?): {dup[:10]}")
    clash = {r["new"] for r in rows} & dirs
    if clash: problems.append(f"new ids already exist as dirs: {sorted(clash)[:10]}")
    m = {r["old"]: r["new"] for r in rows}
    if (exp / "failed_samples.jsonl").exists():
        _, unmapped = new_failed_text(rd(exp / "failed_samples.jsonl"), m, exp.name, strict=False)
        bad = [u for u in unmapped if u[2] != "unmappable"]
        if bad: problems.append(f"failed_samples.jsonl ids that are neither samples nor known-unmappable: {bad[:10]}")
    if problems:
        raise SystemExit("PLAN FAILED:\n  " + "\n  ".join(problems[:50]))

    if verbose:
        n_dir = sum(r["has_dir"] for r in rows)
        print(f"{exp.name}: {len(positions)} positions, {len(rows)} ids ({n_dir} dirs, {len(rows) - n_dir} without a dir)")
        print(f"  slot rule (k-th id of a position = speakers[k]) held for all {slot_rule_ok} samples with metadata")
        for r in rows[:3] + [r for r in rows if not r["has_dir"]][:5] + rows[-2:]:
            print(f"  {r['old']} -> {r['new']}" + ("" if r["has_dir"] else "  (no dir; speaker from slot)"))
        if (exp / "failed_samples.jsonl").exists():
            text = rd(exp / "failed_samples.jsonl")
            n = sum(1 for l in text.splitlines() if l.strip())
            print(f"  failed_samples.jsonl: {n - len(unmapped)} of {n} entries renamed; left as-is (not samples of "
                  f"this experiment): lines {[u[0] for u in unmapped]}")
    return rows


def rewrite_sample_id(text: str, old: str, new: str) -> str:
    """Swap only the {"sample_id": old} line; every other byte (incl. CRLF) is kept."""
    lines = text.splitlines(keepends=True)
    hits = [i for i, l in enumerate(lines) if l.strip() and list(json.loads(l)) == ["sample_id"]]
    assert len(hits) == 1, f"expected one sample_id line, found {len(hits)}"
    i = hits[0]
    cur = json.loads(lines[i])["sample_id"]
    if cur == new: return text  # already done (resumed apply)
    assert cur == old and json.dumps({"sample_id": old}) in lines[i], (old, lines[i])
    lines[i] = lines[i].replace(json.dumps({"sample_id": old}), json.dumps({"sample_id": new}), 1)
    return "".join(lines)


def new_positions_text(original: str, m: dict[str, str]) -> str:
    eol = eol_of(original)
    out = []
    for line in original.splitlines():
        if line.strip():
            ((k, v),) = json.loads(line).items()
            out.append(json.dumps({k: [m[s] for s in v]}) + eol)
    return "".join(out)


def new_failed_text(original: str, m: dict[str, str], exp_name: str, strict: bool = True) -> tuple[str, list]:
    """Each entry's `\\samples\\<id>\\` segments (in item, error, traceback, ...) -> the new id.
    Returns (text, [(line_no, id, reason)] for entries left unchanged)."""
    skip = UNMAPPABLE_FAILED.get(exp_name, lambda d, sid: False)
    eol, out, unmapped = eol_of(original), [], []
    for n, line in enumerate(original.splitlines(), 1):
        if not line.strip(): continue
        d = json.loads(line)
        ids = {g for v in d.values() if isinstance(v, str) for g in (x[1] for x in FAILED_ID.findall(v))}
        if len(ids) != 1: raise SystemExit(f"failed_samples.jsonl line {n}: expected one sample id, found {ids}")
        (sid,) = ids
        if skip(d, sid):
            unmapped.append((n, sid, "unmappable")); out.append(line + eol); continue
        if sid not in m:
            if strict: raise SystemExit(f"failed_samples.jsonl line {n}: {sid} is not a sample of this experiment")
            unmapped.append((n, sid, "unknown")); out.append(line + eol); continue
        d = {k: FAILED_ID.sub(lambda g: g[1] + m[g[2]] + g[3], v) if isinstance(v, str) else v for k, v in d.items()}
        out.append(json.dumps(d) + eol)
    return "".join(out), unmapped


def new_id_map_text(original: str, m: dict[str, str]) -> str:
    eol = eol_of(original)
    out = []
    for line in original.splitlines():
        if line.strip():
            r = json.loads(line)
            out.append(json.dumps({**r, "renamed_sample_id": m[r["new_sample_id"]]}) + eol)
    return "".join(out)


def rewrite_index_files(exp: Path, j: Path, m: dict[str, str]) -> dict[str, str]:
    """What positions.jsonl / failed_samples.jsonl / id_map.jsonl should contain after the rename."""
    want = {"positions.jsonl": new_positions_text(rd(j / "positions.jsonl"), m)}
    if (j / "failed_samples.jsonl").exists():
        want["failed_samples.jsonl"] = new_failed_text(rd(j / "failed_samples.jsonl"), m, exp.name)[0]
    if (j / "id_map.jsonl").exists():
        want["id_map.jsonl"] = new_id_map_text(rd(j / "id_map.jsonl"), m)
    return want


def apply(exp: Path, pos_width: int):
    j = exp / "rename"
    if (j / "map.jsonl").exists():
        rows = [json.loads(l) for l in rd(j / "map.jsonl").splitlines()]
        print(f"resuming from {j}/map.jsonl")
    else:
        rows = plan(exp, pos_width)
        j.mkdir()
        # journal first: the originals of everything that gets rewritten, and every file's inode
        for name in FILES:
            if (exp / name).exists(): wr(j / name, rd(exp / name))
        manifest = []
        for r in rows:
            if not r["has_dir"]: continue
            d = exp / "samples" / r["old"]
            for f in sorted(p for p in d.rglob("*") if p.is_file()):
                st = f.stat()
                e = {"old": r["old"], "rel": str(f.relative_to(d)), "ino": st.st_ino, "size": st.st_size}
                if e["rel"] == "metadata.jsonl": e["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
                manifest.append(json.dumps(e))
        wr(j / "manifest_before.jsonl", "\n".join(manifest) + "\n")
        wr(j / "map.jsonl", "".join(json.dumps(r) + "\n" for r in rows))  # last: its presence = journal complete
        print(f"journal written to {j}: {len(rows)} ids, {len(manifest)} files")

    m = {r["old"]: r["new"] for r in rows}
    for n, r in enumerate(rows):
        if not r["has_dir"]: continue
        src, dst = exp / "samples" / r["old"], exp / "samples" / r["new"]
        # metadata first (inside the old dir), then the rename -- both steps are safe to repeat
        d = src if src.exists() else dst
        meta = d / "metadata.jsonl"
        wr(meta, rewrite_sample_id(rd(meta), r["old"], r["new"]))
        if src.exists():
            assert not dst.exists(), f"{dst} already exists"
            os.rename(src, dst)
        if n % 1000 == 0: print(f"  {n}/{len(rows)} {r['old']} -> {r['new']}", flush=True)
    for name, text in rewrite_index_files(exp, j, m).items():
        wr(exp / name, text)
    print("applied")


def verify(exp: Path) -> bool:
    """Every file from before is at its new path with the SAME inode and size (a rename never
    touches contents); metadata.jsonl, with its sample_id line swapped back, hashes to the original;
    no other files appeared under samples/; the index files are exactly the originals with ids mapped."""
    j = exp / "rename"
    rows = [json.loads(l) for l in rd(j / "map.jsonl").splitlines()]
    m = {r["old"]: r["new"] for r in rows}
    errors, seen, n = [], set(), 0
    for line in rd(j / "manifest_before.jsonl").splitlines():
        e = json.loads(line)
        f = exp / "samples" / m[e["old"]] / e["rel"]
        seen.add(str(f.relative_to(exp)))
        n += 1
        if not f.is_file():
            errors.append(f"missing {f}")
        elif "sha256" in e:  # metadata.jsonl: rewritten (new inode), so check content instead
            text = rd(f)
            if json.dumps({"sample_id": m[e["old"]]}) not in text:
                errors.append(f"{f}: sample_id is not {m[e['old']]!r}")
            elif hashlib.sha256(rewrite_sample_id(text, m[e["old"]], e["old"]).encode()).hexdigest() != e["sha256"]:
                errors.append(f"{f}: differs from the original beyond the sample_id line")
        elif (f.stat().st_ino, f.stat().st_size) != (e["ino"], e["size"]):
            errors.append(f"changed {f}: not the same file as before")
    extra = {str(p.relative_to(exp)) for p in (exp / "samples").rglob("*") if p.is_file()} - seen
    if extra: errors.append(f"{len(extra)} new files under samples/: {sorted(extra)[:10]}")
    stale = [r["old"] for r in rows if r["has_dir"] and (exp / "samples" / r["old"]).exists()]
    if stale: errors.append(f"old dirs still present: {stale[:10]}")
    for name, text in rewrite_index_files(exp, j, m).items():
        if rd(exp / name) != text: errors.append(f"{name} != original with ids mapped")
    print(f"  {sum(r['has_dir'] for r in rows)} dirs, {n} files checked")
    for e in errors[:50]: print("  ERROR", e)
    print("VERIFY OK" if not errors else f"VERIFY FAILED ({len(errors)} errors)")
    return not errors


def undo(exp: Path):
    j = exp / "rename"
    rows = [json.loads(l) for l in rd(j / "map.jsonl").splitlines()]
    for r in rows:
        if not r["has_dir"]: continue
        src, dst = exp / "samples" / r["new"], exp / "samples" / r["old"]
        d = src if src.exists() else dst
        meta = d / "metadata.jsonl"
        wr(meta, rewrite_sample_id(rd(meta), r["new"], r["old"]))
        if src.exists():
            assert not dst.exists(), f"{dst} already exists"
            os.rename(src, dst)
    for name in FILES:
        if (j / name).exists(): wr(exp / name, rd(j / name))
    print(f"undone; journal left at {j} (delete it once you've checked)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "apply", "verify", "undo"])
    ap.add_argument("exp", type=Path, help="experiment dir (holds samples/ and positions.jsonl)")
    ap.add_argument("--pos-width", type=int, default=1, help="zero-pad position ids to this width (1 = no padding: \"110-1\")")
    a = ap.parse_args()
    if a.cmd == "plan": plan(a.exp, a.pos_width)
    elif a.cmd == "apply":
        apply(a.exp, a.pos_width)
        raise SystemExit(0 if verify(a.exp) else 1)
    elif a.cmd == "verify": raise SystemExit(0 if verify(a.exp) else 1)
    else: undo(a.exp)
