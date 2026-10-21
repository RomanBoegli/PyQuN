"""The Stopwatch measures CPU and wall time in any thread and process, and merging tolerates unstopped timers."""

import subprocess
import sys
import threading
from pathlib import Path

from synthetic import model_set

from RaQuN_Lab.strategies.RaQuN.RaQuN import VanillaRaQuN
from RaQuN_Lab.utils.Stopwatch import Stopwatch


def _busy():
    return sum(i * i for i in range(200_000))


def test_stopwatch_in_a_thread():
    results = {}

    def work():
        s = Stopwatch()
        s.start_timer("t")
        _busy()
        s.stop_timer("t")
        results["times"] = (s.get_time("t"), s.get_wall_time("t"))
        # the regression: every timer raised TypeError, so RaQuN could not run at all
        results["matching"], _ = VanillaRaQuN("thread").timed_match(model_set())

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    cpu, wall = results["times"]
    assert cpu > 0 and wall > 0
    assert len(results["matching"].get_matches()) > 0


def test_stopwatch_in_a_subprocess():
    script = (
        "from RaQuN_Lab.utils.Stopwatch import Stopwatch\n"
        "s = Stopwatch(); s.start_timer('t'); sum(i * i for i in range(200_000)); s.stop_timer('t')\n"
        "print(s.get_time('t'), s.get_wall_time('t'))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).parent).stdout
    cpu, wall = map(float, out.split())
    assert cpu > 0 and wall > 0


def test_merge_adds_and_tolerates_unstopped_timers():
    a, b = Stopwatch(), Stopwatch()
    a.start_timer("t")  # never stopped: elapsed stays None
    b.start_timer("t")
    _busy()
    b.stop_timer("t")
    b.start_timer("only_b")
    b.stop_timer("only_b")
    a.merge(b)
    assert a.get_time("t") == b.get_time("t")
    a.merge(b)
    assert a.get_time("t") == 2 * b.get_time("t")
    # merged timers are copies: adding to a does not change b
    assert a.timers["only_b"] is not b.timers["only_b"]
