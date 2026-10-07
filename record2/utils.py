"""Hardware-handle helpers shared by the camera classes in record.ipynb."""


def close_previous_instance(cls):
    """Close cls's live instance (cls._current), if any. Call first in __init__, before opening a new
    handle; set cls._current = self last, once construction succeeded. At most one open handle."""
    if cls._current is not None:
        cls._current.close()
        cls._current = None


def stop_then_close(resource):
    """stop() before close(): stop() tells the device to stop acquiring (releasing feature locks tied
    to "acquisition active"); closing first leaves the camera acquiring, and the next handle then
    fails with "feature is locked"."""
    resource.stop()
    resource.close()
