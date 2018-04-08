from time import time

def timeit(func):
    def _func(*args, **kwargs):
        t1 = time()
        ret = func(*args, **kwargs)
        t2 = time()
        print func, t2 - t1
        return ret
    return _func

def print_io(func):
    def _func(*args, **kwargs):
        ret = func(*args, **kwargs)
        print locals()
        return ret
    return _func