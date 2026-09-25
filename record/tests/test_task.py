import queue
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


class _FakeExperimentConfig:
    def __init__(self):
        self.log_queue = queue.Queue()


def test_log_prints_and_enqueues(capsys):
    ec = _FakeExperimentConfig()
    log(ec, "hello world")
    captured = capsys.readouterr()
    assert "hello world" in captured.out
    assert ec.log_queue.get_nowait() == "hello world"
