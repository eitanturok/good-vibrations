import time

from record.utils import Task, log


def test_task_result_after_join():
    t = Task(lambda a, b: a + b, 2, 3)
    t.join()
    assert t.result == 5
    assert t.exception is None


def test_task_exception_is_captured_not_swallowed():
    def boom():
        raise ValueError("kaboom")

    t = Task(boom)
    t.join()
    assert t.result is None
    assert isinstance(t.exception, ValueError)
    assert str(t.exception) == "kaboom"


def test_task_kwargs_supported():
    t = Task(lambda a, b=0: a + b, 2, b=10)
    t.join()
    assert t.result == 12


def test_task_timing_attributes_populated():
    t = Task(lambda: time.sleep(0.01))
    t.join()
    assert t.thread_id is not None
    assert t.end_time is not None
    assert t.end_time >= t.launch_time


def test_log_prints(capsys):
    # the GUI shows per-sample stages instead; messages go to the notebook output only
    log(object(), "hello world")
    assert "hello world" in capsys.readouterr().out


def test_task_failure_is_reported(capsys):
    """Real bug: plot_smask crashed on every run and the smask panel just never appeared --
    nothing joins a plot task, so a stored-only exception was invisible."""
    def plot_smask():
        raise AttributeError("module 'matplotlib.cm' has no attribute 'get_cmap'")

    Task(plot_smask).join()
    err = capsys.readouterr().err
    assert "plot_smask" in err and "get_cmap" in err
