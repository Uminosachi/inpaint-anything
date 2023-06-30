import torch
import gc
import threading

model_access_sem = threading.Semaphore(1)

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def clear_cache():
    gc.collect()
    torch_gc()

def post_clear_cache(sem):
    with sem:
        gc.collect()
        torch_gc()

def sleep_clear_cache():
    thread = threading.Thread(target=post_clear_cache, args=(model_access_sem,))
    thread.start()
