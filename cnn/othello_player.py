from .main import ConvolutionNN
from .othello.o_class import Othello

from typing import Callable, cast, Dict, Any, Self
import numpy as np
from tqdm import tqdm


class OthelloPlayer:
    def __init__(
        self,
        game: Othello,
        depth: int,
        width: int,
        kernel_size: int = 3,
        padding: int = -1,
        stride: int = 1,
        seed: int = -1,
        player_position: int = -1,
    ) -> None:
        self.game = game
        self.cnn = ConvolutionNN(
            depth, width, kernel_size, padding, stride, seed
        )
        self.states = []
        self.moves = []
        self.player_position = player_position

    def play_once(self) -> tuple[int, int]:
        input = self.get_input_vector()
        output = self.cnn.forward_pass(input)
        logits = output.flatten()
        legal_logits = self.legality_mask(logits)
        move = legal_logits.argmax()
        move = cast(
            int, move
        )  # Doesn't do anything, pure type signaling to IDE
        return x_y_from_move(move, self.game.n)

    def play_no_training(self):
        return lambda: self.play_once

    def play_once_training(self) -> tuple[int, int]:
        input = self.get_input_vector()
        output = self.cnn.forward_pass(input)
        logits = output.flatten()
        legal_logits = self.legality_mask(logits)
        probs = softmax(legal_logits)
        move = np.random.choice(len(probs), p=probs)
        self.moves.append(move)
        self.states.append(input.copy())

        move = cast(
            int, move
        )  # Doesn't do anything, pure type signaling to IDE
        return x_y_from_move(move, self.game.n)

    def train(
        self, spar_par: None | Callable, nb_games: int, learning_rate: float, display=False
    ) -> int:
        # For now CNN can only be p1 because I hardcoded the sign of score
        if spar_par is None:
            raise RuntimeError("Not implemented yet")

        player = lambda: self.play_once_training()
        nb_win = 0
        for _ in tqdm(range(nb_games)):
            self.wipe()
            if self.player_position == -1:
                winner, score = self.game.game(player, spar_par)
            else:
                winner, score = self.game.game(spar_par, player)
            n = self.game.n
            score = self.player_position * score
            normalized_score = score / (n * n)
            if winner == self.player_position:
                nb_win += 1
            for i in range(len(self.moves)):
                move = self.moves[i]
                input = self.states[i]
                output = self.cnn.forward_pass(input)
                logits = output.flatten()
                legal_logits = self.legality_mask(logits, input)
                if (legal_logits == -np.inf).all():
                    print("ohoh")
                probs = softmax(legal_logits)
                gradient_loss = -normalized_score * (
                    one_hot(self.game.n**2, move) - probs
                )
                gradient_loss = np.array(
                    [
                        [[gradient_loss[n * i + j]] for j in range(n)]
                        for i in range(n)
                    ]
                )
                self.cnn.backward_pass(gradient_loss)
            self.cnn.update(learning_rate)
            if display:
                print(score)
        return nb_win

    def legality_mask(
        self, logits: np.ndarray, input: np.ndarray | None = None
    ) -> np.ndarray:
        board = None
        if input is not None:
            n = self.game.n
            board = reconstruct_board(input, n, self.player_position)

        logits = logits.copy()
        n = self.game.n
        for i in range(n):
            for j in range(n):
                if not self.game.check_validity_play(
                    i, j, board, self.player_position
                ):
                    logits[n * i + j] = -np.inf
        return logits

    def get_input_vector(
        self,
    ) -> np.ndarray:
        board = self.game.board
        return np.array(
            [
                [
                    [j == self.player_position, j == -self.player_position]
                    for j in i
                ]
                for i in board
            ]
        )

    def wipe(self):
        self.states = []
        self.moves = []
        self.cnn.reset_gradients()

    def get_state_dict(self) -> Dict[str, Any]:
        return self.cnn.get_state_dict()
    
    def load_state_dict(self, state_dict : Dict[str,Any]) -> None:
        self.cnn = self.cnn.load_state_dict(state_dict)

    @classmethod
    def create_from_state_dict(
        cls, state_dict : Dict[str,Any], game : Othello
    ) -> Self :
        nb_layer = state_dict["meta/nb_layer"]
        kernel_size = state_dict["meta/kernel_size"]
        nb_channel = state_dict["meta/nb_channel"]
        padding = state_dict["meta/padding"]
        stride = state_dict["meta/stride"]
        obj = cls(
            game, nb_layer, nb_channel, kernel_size, padding, stride
            )
        obj.load_state_dict(state_dict)
        return obj


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum()


def one_hot(n: int, k: int) -> np.ndarray:
    vector = np.zeros(n)
    vector[k] = 1.0
    return vector


def x_y_from_move(move: int, n: int):
    x_move = int(move // n)
    y_move = int(move % n)
    return x_move, y_move


def reconstruct_board(
    input_vector: np.ndarray, n: int, player_position: int
) -> list[list[int]]:
    board = [
        [
            player_position if input_vector[i][j][0] 
            else - player_position * input_vector[i][j][1]
            for j in range(n)
        ]
        for i in range(n)
    ]
    return board
