from time import time
import logging
import torch
from colorlog import ColoredFormatter


def sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time()


Log = logging.getLogger()
Log.time = time
Log.sync_time = sync_time

# Set default
Log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Use colorlog
formatstring = "[%(cyan)s%(asctime)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] %(message)s"
datefmt = "%m/%d %H:%M:%S"
ch.setFormatter(ColoredFormatter(formatstring, datefmt=datefmt))

Log.addHandler(ch)
# Log.info("Init-Logger")


def timer(sync_cuda=False, mem=False, loop=1):
    """
    Args:
        func: function
        sync_cuda: bool, whether to synchronize cuda
        mem: bool, whether to log memory
    """
    has_cuda = torch.cuda.is_available()

    def decorator(func):
        def wrapper(*args, **kwargs):
            if mem and has_cuda:
                start_mem = torch.cuda.memory_allocated() / 1024**2
            if sync_cuda and has_cuda:
                torch.cuda.synchronize()

            start = Log.time()
            for _ in range(loop):
                result = func(*args, **kwargs)

            if sync_cuda and has_cuda:
                torch.cuda.synchronize()
            if loop == 1:
                message = f"{func.__name__} took {Log.time() - start:.3f} s."
            else:
                message = f"{func.__name__} took {((Log.time() - start))/loop:.3f} s. (loop={loop})"

            if mem and has_cuda:
                end_mem = torch.cuda.memory_allocated() / 1024**2
                end_max_mem = torch.cuda.max_memory_allocated() / 1024**2
                message += f" Start_Mem {start_mem:.1f} Max {end_max_mem:.1f} MB"
            Log.info(message)

            return result

        return wrapper

    return decorator


def timed(fn):
    """example usage: timed(lambda: model(inp))"""
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        return result, start.elapsed_time(end) / 1000
    else:
        start = time()
        result = fn()
        return result, time() - start
