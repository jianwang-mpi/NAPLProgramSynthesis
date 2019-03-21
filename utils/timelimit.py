import functools
import threading
import ctypes
import inspect
class TimeOutException(Exception):
    pass


def time_limit(timeout=0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def _async_raise(tid, exctype):
                """raises the exception, performs cleanup if needed"""
                tid = ctypes.c_long(tid)
                if not inspect.isclass(exctype):
                    exctype = type(exctype)
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
                if res == 0:
                    raise ValueError("invalid thread id")
                elif res != 1:
                    # """if it returns a number greater than one, you're in trouble,
                    # and you should call it again with exc=NULL to revert the effect"""
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
                    raise SystemError("PyThreadState_SetAsyncExc failed")

            class TimeLimit(threading.Thread):
                def __init__(self):
                    super(TimeLimit, self).__init__()
                    self.error = None
                    self.result = None

                def run(self):
                    try:
                        self.error = None
                        self.result = func(*args, **kwargs)
                    except Exception as e:
                        self.error = e
                        self.result = None

                def stop(self):
                    try:
                        _async_raise(self.ident, SystemExit)
                    except Exception:
                        pass

            t = TimeLimit()
            t.setDaemon(True)
            t.start()

            if timeout > 0:
                t.join(timeout)
            else:
                t.join()

            if t.isAlive():
                t.stop()
                emsg = "function(%s) execute timeout after %d second" % (func.__name__, timeout)
                raise TimeOutException(emsg)

            if t.error is not None:
                raise t.error

            return t.result

        return wrapper

    return decorator

