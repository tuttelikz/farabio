import warnings
from functools import wraps


def _warn(*args, **kwargs):
    warnings.warn(*args, **kwargs)


def rank_zero_only(fn):

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


rank_zero_warn = rank_zero_only(_warn)
