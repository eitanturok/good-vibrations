"""Makes `data.*` importable from record/post_process.py the same way record.ipynb's own
cell 1 does (inserting src/ onto sys.path) -- pytest, run from the repo root, doesn't do
this automatically."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
