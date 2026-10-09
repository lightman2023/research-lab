from __future__ import annotations

import argparse
from collections import deque
import signal
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel v2 self-play launcher")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--max-plies", type=int, default=512)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.msgpack")
    parser.add_argument("--output-dir", type=str, default="selfplay_data")
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    pending = deque((number, 0) for number in range(1, args.games + 1))
    active: list[tuple[int, int, subprocess.Popen[bytes]]] = []
    stopping = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True
        for _, _, process in active:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while (pending or active) and not stopping:
        while pending and len(active) < args.workers:
            game_number, attempt = pending.popleft()
            command = [
                sys.executable,
                "-u",
                "selfplay.py",
                "--games",
                "1",
                "--simulations",
                str(args.simulations),
                "--max-plies",
                str(args.max_plies),
                "--checkpoint",
                args.checkpoint,
                "--output-dir",
                args.output_dir,
            ]
            process = subprocess.Popen(command)
            print(
                f"Parallel game {game_number}/{args.games} "
                f"attempt {attempt + 1}/{args.retries + 1} started (pid={process.pid})",
                flush=True,
            )
            active.append((game_number, attempt, process))
            if pending:
                time.sleep(1.0)

        remaining = []
        for game_number, attempt, process in active:
            exit_code = process.poll()
            if exit_code is None:
                remaining.append((game_number, attempt, process))
            elif exit_code == 0:
                print(f"Parallel game {game_number}/{args.games} finished", flush=True)
            elif attempt < args.retries:
                print(f"Parallel game {game_number} failed; retrying", flush=True)
                pending.append((game_number, attempt + 1))
            else:
                print(f"Parallel game {game_number} failed permanently", flush=True)
                stopping = True
        active = remaining
        if active and not stopping:
            time.sleep(0.2)

    if stopping:
        for _, _, process in active:
            if process.poll() is None:
                process.terminate()
        return 1
    print(f"All {args.games} parallel self-play games finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
