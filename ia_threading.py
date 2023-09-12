import gc
import inspect
import threading
from functools import wraps

import torch

from ia_check_versions import ia_check_versions

model_access_sem = threading.Semaphore(1)


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if ia_check_versions.torch_mps_is_available:
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def clear_cache():
    gc.collect()
    torch_gc()


def post_clear_cache(sem):
    with sem:
        gc.collect()
        torch_gc()


def async_post_clear_cache():
    thread = threading.Thread(target=post_clear_cache, args=(model_access_sem,))
    thread.start()


def clear_cache_decorator(func):
    @wraps(func)
    def yield_wrapper(*args, **kwargs):
        clear_cache()
        yield from func(*args, **kwargs)
        clear_cache()

    @wraps(func)
    def wrapper(*args, **kwargs):
        clear_cache()
        res = func(*args, **kwargs)
        clear_cache()
        return res

    if inspect.isgeneratorfunction(func):
        return yield_wrapper
    else:
        return wrapper
