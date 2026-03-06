from .o_class import Othello
from random import randint
from typing import Callable


def get_random_player(game: Othello):
    def play_random():
        plays = game.available_plays()
        return plays[randint(0, len(plays) - 1)]

    return play_random


def get_random_player_with_rules(game: Othello):
    def play_almost_random():
        plays = game.available_plays()
        n = game.n - 1
        for play in plays:
            if play in [(0, 0), (0, n), (n, 0), (n, n)]:
                return play
        return plays[randint(0, len(plays) - 1)]

    return play_almost_random


def get_min_max_basic(game: Othello, depth: int):
    assert depth > 0

    def eval(game: Othello, player: int):
        return game.score() * player

    solv = Solver(eval)
    return lambda: solv.min_max(depth, game, game.get_player())[1]


def get_min_max_corners(game: Othello, depth: int):
    assert depth > 0

    def eval(game: Othello, player: int):
        base = game.score() * player
        for x in range(-1, 2, 2):
            for y in range(-1, 2, 2):
                base += game.board[x][y] * game.n
        return base

    solv = Solver(eval)
    return lambda: solv.min_max(depth, game, game.get_player())[1]


class Solver:
    def __init__(self, evaluation: Callable[[Othello, int], int]):
        self.evaluation = evaluation

    def min_max(
        self, depth: int, game: Othello, player: int
    ) -> tuple[int, tuple[int, int]]:
        if depth == 0 or game.check_end():
            return -self.evaluation(game, player), (-1, -1)

        plays = game.available_plays()
        best = None
        best_play = None
        for x, y in plays:
            new_game = game.copy()
            new_game.play(x, y)
            score, _ = self.min_max(depth - 1, new_game, -player)
            if not best or score > best:
                best = score
                best_play = (x, y)
        assert best_play is not None and best is not None
        return -best, best_play
