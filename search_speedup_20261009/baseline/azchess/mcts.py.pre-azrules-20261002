from __future__ import annotations

import math
import random

import chess
import jax
import jax.numpy as jnp
import numpy as np

from .game import encode_board, legal_action_map, move_to_action, terminal_value


class Node:
    def __init__(self, prior: float = 0.0) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        # None represents the optional action of claiming a draw.
        self.children: dict[chess.Move | None, Node] = {}

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class MCTS:
    def __init__(
        self,
        model,
        variables,
        simulations: int = 50,
        c_init: float = 1.25,
        c_base: float = 19652.0,
        root_noise: bool = False,
    ) -> None:
        self.model = model
        self.variables = variables
        self.simulations = simulations
        self.c_init = c_init
        self.c_base = c_base
        self.root_noise = root_noise
        self._predict = jax.jit(lambda variables, x: model.apply(variables, x, train=False))

    def evaluate(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        logits, value = self._predict(self.variables, jnp.asarray(encode_board(board)[None]))
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
        if board.can_claim_draw():
            # The network has no policy logit for a draw claim. Give it the
            # same prior as one move in a uniform policy, then renormalize.
            claim_prior = 1.0 / (len(priors) + 1)
            node.children = {
                move: Node(prior * (1.0 - claim_prior))
                for move, prior in priors.items()
            }
            node.children[None] = Node(claim_prior)
        else:
            node.children = {move: Node(prior) for move, prior in priors.items()}
        return value

    def select_child(self, node: Node) -> tuple[chess.Move | None, Node]:
        visits = max(1, node.visit_count)
        exploration = math.log((visits + self.c_base + 1.0) / self.c_base) + self.c_init
        scale = exploration * math.sqrt(visits)
        return max(
            node.children.items(),
            key=lambda item: -item[1].value
            + item[1].prior * scale / (1 + item[1].visit_count),
        )

    def search(self, board: chess.Board) -> Node:
        root = Node()
        self.expand(root, board)
        if self.root_noise and root.children:
            noise = np.random.dirichlet([0.3] * len(root.children))
            for child, random_prior in zip(root.children.values(), noise):
                child.prior = 0.75 * child.prior + 0.25 * float(random_prior)
        for _ in range(self.simulations):
            simulation_board = board.copy(stack=True)
            node = root
            path = [node]
            claimed = False
            while node.children:
                move, node = self.select_child(node)
                path.append(node)
                if move is None:
                    claimed = True
                    break
                simulation_board.push(move)
                if node.visit_count == 0:
                    break
            value = 0.0 if claimed else self.expand(node, simulation_board)
            for visited in reversed(path):
                visited.visit_count += 1
                visited.value_sum += value
                value = -value
        return root

    def policy_and_target(self, board: chess.Board, temperature: float = 1.0):
        root = self.search(board)
        moves = [move for move in root.children if move is not None]
        visits = np.asarray([root.children[move].visit_count for move in moves], dtype=np.float64)
        if visits.sum() <= 0:
            target = np.ones_like(visits) / len(visits)
        else:
            target = visits / visits.sum()
        if temperature <= 1e-6:
            probabilities = np.zeros_like(visits)
            probabilities[int(visits.argmax())] = 1.0
        else:
            scaled = np.power(np.maximum(visits, 0.0), 1.0 / temperature)
            if scaled.sum() <= 0:
                scaled = np.ones_like(visits)
            probabilities = scaled / scaled.sum()
        indices = np.asarray([move_to_action(board, move) for move in moves], dtype=np.uint16)
        return moves, probabilities, indices, target

    def decision_and_target(self, board: chess.Board, temperature: float = 1.0):
        """Sample a move or draw claim; train policy on legal-move visits."""
        root = self.search(board)
        actions = list(root.children)
        visits = np.asarray([root.children[action].visit_count for action in actions], dtype=np.float64)
        if temperature <= 1e-6:
            probabilities = np.zeros_like(visits)
            probabilities[int(visits.argmax())] = 1.0
        else:
            scaled = np.power(np.maximum(visits, 0.0), 1.0 / temperature)
            probabilities = (
                scaled / scaled.sum()
                if scaled.sum() > 0
                else np.ones_like(visits) / len(visits)
            )
        selected = actions[int(np.random.choice(len(actions), p=probabilities))]
        moves = [action for action in actions if action is not None]
        move_visits = np.asarray([root.children[move].visit_count for move in moves], dtype=np.float64)
        target = (
            move_visits / move_visits.sum()
            if move_visits.sum() > 0
            else np.ones_like(move_visits) / len(move_visits)
        )
        indices = np.asarray([move_to_action(board, move) for move in moves], dtype=np.uint16)
        return selected, indices, target

    def choose_action(self, board: chess.Board, temperature: float = 0.0) -> chess.Move | None:
        action, _, _ = self.decision_and_target(board, temperature)
        return action

    def policy(self, board: chess.Board, temperature: float = 1.0):
        moves, probabilities, indices, _ = self.policy_and_target(board, temperature)
        return moves, probabilities, indices

    def choose_move(self, board: chess.Board, temperature: float = 0.0) -> chess.Move:
        moves, probabilities, _ = self.policy(board, temperature)
        return random.choices(moves, weights=probabilities, k=1)[0]
