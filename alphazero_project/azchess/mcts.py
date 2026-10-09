from __future__ import annotations

import math
import random

import chess
import jax
import jax.numpy as jnp
import numpy as np

from .game import ACTION_SIZE, encode_board, legal_action_map, terminal_value


class Node:
    def __init__(self, prior: float = 0.0) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[chess.Move, Node] = {}

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class MCTS:
    def __init__(
        self,
        model,
        variables,
        simulations: int = 50,
        cpuct: float = 1.5,
        root_noise: bool = False,
    ) -> None:
        self.model = model
        self.variables = variables
        self.simulations = simulations
        self.cpuct = cpuct
        self.root_noise = root_noise
        self._predict = jax.jit(lambda variables, x: model.apply(variables, x, train=False))

    def evaluate(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        logits, value = self._predict(self.variables, jnp.asarray(encode_board(board)[None, ...]))
        logits = np.asarray(logits[0])
        action_map = legal_action_map(board)
        indices = np.fromiter(action_map.keys(), dtype=np.int32)
        legal_logits = logits[indices]
        legal_logits -= legal_logits.max()
        probabilities = np.exp(legal_logits)
        probabilities /= probabilities.sum()
        priors = {action_map[index]: float(p) for index, p in zip(indices, probabilities)}
        return priors, float(value[0])

    def expand(self, node: Node, board: chess.Board) -> float:
        terminal = terminal_value(board)
        if terminal is not None:
            return terminal
        priors, value = self.evaluate(board)
        node.children = {move: Node(prior) for move, prior in priors.items()}
        return value

    def select_child(self, node: Node) -> tuple[chess.Move, Node]:
        scale = math.sqrt(max(1, node.visit_count))
        return max(
            node.children.items(),
            key=lambda item: -item[1].value
            + self.cpuct * item[1].prior * scale / (1 + item[1].visit_count),
        )

    def search(self, board: chess.Board) -> Node:
        root = Node()
        self.expand(root, board)
        if self.root_noise and root.children:
            noise = np.random.dirichlet([0.3] * len(root.children))
            for child, random_prior in zip(root.children.values(), noise):
                child.prior = 0.75 * child.prior + 0.25 * float(random_prior)
        for _ in range(self.simulations):
            # 三fold repetitionを探索中にも判定できるよう、棋譜履歴を保持する。
            simulation_board = board.copy(stack=True)
            node = root
            path = [node]
            while node.children:
                move, node = self.select_child(node)
                simulation_board.push(move)
                path.append(node)
                if node.visit_count == 0:
                    break
            value = self.expand(node, simulation_board)
            for visited in reversed(path):
                visited.visit_count += 1
                visited.value_sum += value
                value = -value
        return root

    def policy(self, board: chess.Board, temperature: float = 1.0):
        root = self.search(board)
        moves = list(root.children)
        visits = np.asarray([root.children[move].visit_count for move in moves], dtype=np.float64)
        # 方策教師そのものから反復癖を弱める。同一局面への2回目の到達は
        # 他に選択肢がある限り外し、自分の直前手の単純な逆戻りも強く減点する。
        repetition_weights = np.ones_like(visits)
        previous_own_move = board.move_stack[-2] if len(board.move_stack) >= 2 else None
        for index, move in enumerate(moves):
            candidate = board.copy(stack=True)
            candidate.push(move)
            if candidate.is_repetition(2):
                repetition_weights[index] = 0.0
            elif (
                previous_own_move is not None
                and move.from_square == previous_own_move.to_square
                and move.to_square == previous_own_move.from_square
            ):
                repetition_weights[index] = 0.01
        adjusted_visits = visits * repetition_weights
        if adjusted_visits.sum() > 0:
            visits = adjusted_visits
        if temperature <= 1e-6:
            probabilities = np.zeros_like(visits)
            probabilities[int(visits.argmax())] = 1.0
        else:
            scaled = np.where(
                visits > 0,
                np.power(np.maximum(visits, 0.0), 1.0 / temperature),
                0.0,
            )
            if scaled.sum() <= 0:
                scaled = np.ones_like(visits)
            probabilities = scaled / scaled.sum()
        dense = np.zeros(ACTION_SIZE, dtype=np.float32)
        move_to_index = {move: index for index, move in legal_action_map(board).items()}
        for move, probability in zip(moves, probabilities):
            dense[move_to_index[move]] = probability
        return moves, probabilities, dense

    def choose_move(self, board: chess.Board, temperature: float = 0.0) -> chess.Move:
        moves, probabilities, _ = self.policy(board, temperature)
        return random.choices(moves, weights=probabilities, k=1)[0]
