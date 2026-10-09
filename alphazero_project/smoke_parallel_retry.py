"""Verify that one failed game is retried without discarding other games."""

from contextlib import redirect_stdout
from io import StringIO
import sys

import parallel_selfplay


class FakeProcess:
    outcomes = iter((-11, 0, 0))
    next_pid = 100

    def __init__(self, _command):
        self.exit_code = next(self.outcomes)
        self.pid = self.next_pid
        FakeProcess.next_pid += 1

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.exit_code = -15

    def wait(self, timeout=None):
        return self.exit_code

    def kill(self):
        self.exit_code = -9


parallel_selfplay.subprocess.Popen = FakeProcess
parallel_selfplay.time.sleep = lambda _seconds: None
sys.argv = [
    "parallel_selfplay.py",
    "--games",
    "2",
    "--workers",
    "2",
    "--retries",
    "2",
    "--retry-delay",
    "0",
]
output = StringIO()
with redirect_stdout(output):
    exit_code = parallel_selfplay.main()

text = output.getvalue()
assert exit_code == 0, text
assert "failed (exit=-11); retry 1/2" in text, text
assert "retry 1/2 started" in text, text
assert "All 2 parallel self-play games finished" in text, text
print("PARALLEL_RETRY_OK")
