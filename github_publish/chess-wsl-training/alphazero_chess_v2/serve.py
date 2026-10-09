"""Persistent JSON-lines bridge for the Windows v2 chess GUI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess

from azchess.checkpoint import load_variables
from azchess.mcts import MCTS


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=50)
    args = parser.parse_args()
    model, variables = load_variables(args.checkpoint)
    agent = MCTS(model, variables, simulations=args.simulations)
    emit({"status": "ready", "checkpoint": str(args.checkpoint)})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "quit":
                emit({"status": "bye"})
                return
            board = chess.Board()
            for uci in request.get("moves", []):
                board.push_uci(uci)
            move = agent.choose_move(
                board, temperature=float(request.get("temperature", 0.0))
            )
            emit({"move": move.uci()})
        except Exception as error:
            emit({"error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
