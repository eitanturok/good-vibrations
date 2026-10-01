"""The GUI cell's routes, against the real recording cells with fake hardware: every button is one
notebook function, and the hardware can't be changed mid-position."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from record2.tests.fakes import cell, load

FORM = {"position": {"speakers": [1, 3], "box": "gastronorm", "crop": [0.1, 0.9, 0.1, 0.9], "objects": {"red-cube": 1},
                     "prompts": {"red-cube": "Red cube"}, "layout": "one-cube", "description": ""},
        "preview": {"speaker": 3, "laser": 5, "channel": 0, "use_pc": True}, "save": True, "vibrate": True}


@pytest.fixture
def gui(tmp_path):
    import soundfile
    ns = load(tmp_path)
    ns.update(sf=soundfile, REPO_DIR=Path(__file__).resolve().parents[2])
    exec(cell("app = FastAPI()"), ns)
    ns["client"] = TestClient(ns["app"])
    return ns


def finish(ns):  # wait for the background run and its saves
    ns["RECORD_POOL"].submit(lambda: None).result()
    ns["SAVE_POOL"].shutdown(wait=True)


def test_page_catalog_and_state(gui):
    c = gui["client"]
    assert "<title>record</title>" in c.get("/").text and c.get("/static/app.js").status_code == 200
    assert c.get("/static/../record.ipynb").status_code == 404
    catalog = c.get("/catalog").json()
    assert catalog["position"]["box"] == "gastronorm"
    assert catalog["roi_defaults"] == {"n_rows": 10, "n_cols": 10, "roi_width": 80, "roi_height": 32}  # the ROI fields before any grid
    state = c.get("/state").json()
    assert state["recording"] is False and state["next_position_id"] == 1 and len(state["laser"]["roi"]["rois"]) == 100
    assert c.get(f"/state?since={state['records'][-1]['seq'] if state['records'] else 0}").json()["records"] == []


def test_run_records_a_position_and_its_panels(gui):
    c = gui["client"]
    assert c.post("/run", json=FORM).status_code == 200
    finish(gui)
    state = c.get("/state").json()
    assert state["position_id"] == "000001" and set(state["panels"].values()) == {"ready"} and state["counts"]["experiment"] == 1
    assert c.get("/panel/smask.png").headers["content-type"] == "image/png"
    from io import BytesIO
    from PIL import Image
    assert Image.open(BytesIO(c.get("/panel/shifts.png?w=900&h=250").content)).size == (900, 250)  # fills the browser's box
    assert c.get("/preview.wav").headers["content-type"] == "audio/wav"
    assert any(r.get("stage") == "vibrate" and r["label"] == "000001-3" for r in state["records"])  # the timeline


def test_a_bad_form_is_a_400_and_records_nothing(gui):
    form = {**FORM, "position": {**FORM["position"], "prompts": {}}}
    r = gui["client"].post("/run", json=form)
    assert r.status_code == 400 and "prompt" in r.json()["detail"]


def test_no_hardware_changes_while_recording(gui):
    c = gui["client"]
    with gui["RECORDING"]:
        assert c.post("/run", json=FORM).status_code == 409
        assert c.post("/camera", json={"cam": "laser", "exposure": 50}).status_code == 409
        assert c.post("/calibrate").status_code == 409
    assert c.post("/camera", json={"cam": "laser", "exposure": 50}).status_code == 200
    assert c.get("/state").json()["laser"]["exposure"] == 50


def test_calibrate_then_set_rois(gui):
    c = gui["client"]
    assert c.post("/calibrate").status_code == 200
    assert c.get("/state").json()["laser"]["roi"] is None  # wide-open: the whole sensor to click on
    rois = {"rows": [200, 600], "crop": [400, 1500], "cols": [500, 900, 1300], "roi_width": 76, "roi_height": 32}
    assert c.post("/rois", json=rois).status_code == 200
    laser = c.get("/state").json()["laser"]
    assert len(laser["roi"]["rois"]) == 6 and laser["size"] == [1104, 64]  # columns 400-1504 (16 px grid), only the 2 ROI rows
    assert c.post("/rois", json={**rois, "roi_height": 30}).status_code == 400  # the camera's multiple-of-4 rule


def test_delete_the_last_position(gui):
    c = gui["client"]
    c.post("/run", json=FORM)
    finish(gui)
    assert c.post("/delete", json={"position_id": "000001"}).json() == ["000001-1", "000001-3"]
    assert c.get("/state").json()["counts"]["experiment"] == 0


def test_panels_say_not_run_yet_then_failed(gui):
    """Each panel says what it's waiting for: nothing run yet, still computing, or failed (never "loading" forever)."""
    from concurrent.futures import Future
    c = gui["client"]
    assert set(c.get("/state").json()["panels"].values()) == {"none"}
    failed = Future()
    failed.set_exception(RuntimeError("SAM3 timed out"))
    gui["LAST"].update(position_id="000001", seg=failed, preview=Future())
    panels = c.get("/state").json()["panels"]
    assert (panels["smask"], panels["coverage"], panels["shifts"]) == ("failed", "failed", "loading")


def test_coverage_follows_the_layout(gui):
    """Any layout's coverage is drawn from its saved positions, with no run needed; an unseen layout is a 404."""
    c = gui["client"]
    assert c.get("/panel/coverage.png?layout=one-cube").status_code == 404
    c.post("/run", json=FORM)
    finish(gui)
    assert c.get("/panel/coverage.png?layout=one-cube").headers["content-type"] == "image/png"
    assert c.get("/panel/coverage.png?layout=two-cubes").status_code == 404
