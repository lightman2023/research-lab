"""Run independent self-play games concurrently without importing JAX in the parent."""

from __future__ import annotations

import argparse
from collections import deque
import signal
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel AlphaZero self-play launcher")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    args = parser.parse_args()
    if args.games < 1 or args.workers < 1 or args.retries < 0:
        parser.error("games and workers must be at least 1; retries must be non-negative")

    pending = deque((game_number, 0) for game_number in range(1, args.games + 1))
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
            ]
            process = subprocess.Popen(command)
            if attempt:
                print(
                    f"Parallel game {game_number}/{args.games} retry "
                    f"{attempt}/{args.retries} started (pid={process.pid})",
                    flush=True,
                )
            else:
                print(
                    f"Parallel game {game_number}/{args.games} started (pid={process.pid})",
                    flush=True,
                )
            active.append((game_number, attempt, process))
            # GPU初期化を完全に同時実行させず、ドライバ負荷を少し分散する。
            if pending:
                time.sleep(1.0)

        still_active: list[tuple[int, int, subprocess.Popen[bytes]]] = []
        for game_number, attempt, process in active:
            exit_code = process.poll()
            if exit_code is None:
                still_active.append((game_number, attempt, process))
            elif exit_code != 0:
                if attempt < args.retries:
                    next_attempt = attempt + 1
                    print(
                        f"Parallel game {game_number}/{args.games} failed "
                        f"(exit={exit_code}); retry {next_attempt}/{args.retries} "
                        f"after {args.retry_delay:g}s",
                        flush=True,
                    )
                    time.sleep(max(0.0, args.retry_delay))
                    pending.append((game_number, next_attempt))
                else:
                    print(
                        f"Parallel game {game_number}/{args.games} failed "
                        f"after {args.retries} retries (exit={exit_code})",
                        flush=True,
                    )
                    stopping = True
            else:
                print(f"Parallel game {game_number}/{args.games} finished", flush=True)
        active = still_active
        if active and not stopping:
            time.sleep(0.2)

    if stopping:
        for _, _, process in active:
            if process.poll() is None:
                process.terminate()
        for _, _, process in active:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        return 1
    print(f"All {args.games} parallel self-play games finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
