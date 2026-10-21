import copy
import time


class TimingStruct:
    def __init__(self):
        # CPU time of the thread that ran the timer
        self.start = None
        self.end = None
        self.elapsed = None
        # wall-clock time
        self.wall_start = None
        self.wall_end = None
        self.wall_elapsed = None


def _add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b


# times named sections in CPU time of the calling thread (time.thread_time) and in wall-clock time (time.perf_counter).
# A timer has to be started and stopped in the same thread.
class Stopwatch:
    def __init__(self):
        self.timers = {}

    def start_timer(self, name) -> None:
        time_struc = TimingStruct()
        time_struc.wall_start = time.perf_counter()
        time_struc.start = time.thread_time()
        self.timers[name] = time_struc

    def stop_timer(self, name: str) -> None:
        cur_struct = self.timers[name]
        cur_struct.end = time.thread_time()
        cur_struct.wall_end = time.perf_counter()
        cur_struct.elapsed = cur_struct.end - cur_struct.start
        cur_struct.wall_elapsed = cur_struct.wall_end - cur_struct.wall_start

    def merge(self, other: 'Stopwatch') -> None:
        for name, ts in other.timers.items():
            if name in self.timers:
                cur_struct = self.timers[name]
                cur_struct.elapsed = _add(cur_struct.elapsed, ts.elapsed)
                cur_struct.wall_elapsed = _add(getattr(cur_struct, 'wall_elapsed', None),
                                               getattr(ts, 'wall_elapsed', None))
            else:
                self.timers[name] = copy.copy(ts)

    # CPU time of the timer in seconds
    def get_time(self, name):
        return self.timers[name].elapsed

    # wall-clock time of the timer in seconds
    def get_wall_time(self, name):
        return getattr(self.timers[name], 'wall_elapsed', None)
