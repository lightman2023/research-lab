from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess

from azchess.checkpoint import load_variables
from azchess.mcts import MCTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a small AlphaZero-style chess model")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.msgpack"))
    parser.add_argument("--color", choices=("white", "black", "random"), default="white")
    parser.add_argument("--simulations", type=int, default=50)
    args = parser.parse_args()

    model, variables = load_variables(args.checkpoint)
    agent = MCTS(model, variables, simulations=args.simulations)
    board = chess.Board()
    color = random.choice(("white", "black")) if args.color == "random" else args.color
    human = chess.WHITE if color == "white" else chess.BLACK
    print(f"You are {color}. Enter moves such as e2e4. Type quit to stop.")

    while not board.is_game_over(claim_draw=True):
        print("\n", board, "\n", sep="")
        if board.turn == human:
            text = input("Your move: ").strip().lower()
            if text in {"quit", "exit"}:
                return
            try:
                move = chess.Move.from_uci(text)
            except ValueError:
                print("Use UCI format, for example e2e4.")
                continue
            if move not in board.legal_moves:
                print("Illegal move.")
                continue
        else:
            print("AlphaZero-style AI is thinking...")
            move = agent.choose_move(board, temperature=0.25)
            print("AI move:", move.uci())
        board.push(move)
    print(board)
    print("Result:", board.result(claim_draw=True))


if __name__ == "__main__":
    main()

