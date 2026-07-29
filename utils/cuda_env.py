# On Windows, cupy needs the CUDA DLLs from the pip-installed nvidia-* wheels
# (no system CUDA toolkit). Register them before any cupy import: ctypes resolves
# via add_dll_directory, but nvrtc loads its builtins DLL through PATH. CUDA_PATH
# must also point at nvidia-cuda-runtime's dir so cupy's NVRTC compiler can find
# CUDA headers (e.g. cuda_fp16.h), even though no nvcc/toolkit is installed.
#
# This must run before the FIRST `import cupy` anywhere in the process: cupy
# memoizes its CUDA root lookup at first call (cupy._environment._cuda_path),
# so setting CUDA_PATH after cupy has already been imported has no effect.
import os, sys, glob, sysconfig

if sys.platform == "win32":
    _nvidia_dir = os.path.join(sysconfig.get_paths()["purelib"], "nvidia")
    for _d in glob.glob(os.path.join(_nvidia_dir, "*", "bin")):
        os.add_dll_directory(_d)
        os.environ["PATH"] = _d + os.pathsep + os.environ["PATH"]
    _cuda_runtime_dir = os.path.join(_nvidia_dir, "cuda_runtime")
    if "CUDA_PATH" not in os.environ and os.path.isdir(os.path.join(_cuda_runtime_dir, "include")):
        os.environ["CUDA_PATH"] = _cuda_runtime_dir
