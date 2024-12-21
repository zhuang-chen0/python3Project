import threading
import time
import ctypes

class TimedThread(threading.Thread):
    def __init__(self, target, timeout=None, *args, **kwargs):
        super().__init__(target=target, *args, **kwargs)
        self.timeout = timeout
        self._timer = None
        self._lock = threading.Lock()
        self._stopped = False

    def _get_tid(self):
        if hasattr(self, "_thread_id"):
            return self._thread_id
        for tid, tobj in threading._active.items():
            if tobj is self:
                self._thread_id = tid
                return tid
        raise RuntimeError("Could not determine thread ID")

    def stop(self):
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            thread_id = self._get_tid()
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread_id),
                ctypes.py_object(SystemExit)
            )
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
                raise SystemError("PyThreadState_SetAsyncExc failed")

    def start(self):
        super().start()
        if self.timeout:
            self._timer = threading.Timer(self.timeout, self.stop)
            self._timer.start()

    def join(self, timeout=None):
        super().join(timeout)
        if self._timer:
            self._timer.cancel()

# 使用示例
def long_task():
    try:
        while True:
            print("Working...")
            time.sleep(1)
    except SystemExit:
        print("Task stopped")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # 创建一个3秒超时的线程
    thread = TimedThread(target=long_task, timeout=3)
    thread.start()
    print("Thread finished")
    while thread.is_alive():
        print("Main Working...")
        time.sleep(2)
    print("Main Thread finished")