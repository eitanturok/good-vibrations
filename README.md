# Good Vibe-Rations

![good vibes](assets/good%20vibes.png)

On windows do
```bash
uv venv
.venv\Scripts\activate
uv pip install -r pyproject.toml --extra lab
uv pip install "../../../Program Files/Euresys/eGrabber/python/egrabber-26.01.0.5-py2.py3-none-any.whl"
```
Then run files directly in the venv with `python foo.py` or via `uv` with `uv run --no-sync foo.py`. The egrabber files only exists on my windows machine and so we cannot add it directly to uv.lock or pyproject.toml because then this will fail on mac/linux. So we add `--no-syc` to rpevent egrabber from being written into uv.lock.

On mac/windows with uv normally.

`uv run modal run -d src/model.py`
