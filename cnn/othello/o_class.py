from typing import Callable, Self


class InvalidPlay(Exception):
    pass


class Exit(Exception):
    pass


class Othello:
    def __init__(self, n: int, display: bool = False):
        assert n % 2 == 0
        self.n = n
        self.board = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(2):
            for j in range(2):
                self.board[n // 2 - 1 + i][n // 2 - 1 + j] = -1 + 2 * (i ^ j)
        self.active_player = -1
        self.display = display

    def __str__(self) -> str:
        map_symbols = {0: "-", 1: "O", -1: "X"}
        s = f"\n{map_symbols[self.active_player]}  "
        for i in range(self.n):
            s += f"{i}  "
        s += "y"
        for i in range(self.n):
            s += f"\n{i}  "
            for j in range(self.n):
                s += map_symbols[self.board[i][j]] + "  "
        s += "\nx"
        return s

    def get_player(self):
        return self.active_player

    def copy(self) -> Self:
        copy = Othello(self.n)
        for i in range(self.n):
            for j in range(self.n):
                copy.board[i][j] = self.board[i][j]
        copy.active_player = self.active_player
        return copy

    def wipe(self):
        n = self.n
        self.board = [[0 for _ in range(n)] for _ in range(n)]
        self.active_player = -1
        for i in range(2):
            for j in range(2):
                self.board[n // 2 - 1 + i][n // 2 - 1 + j] = -1 + 2 * (i ^ j)

    def check_line(
        self,
        x: int,
        y: int,
        dx: int,
        dy: int,
        board: list | None = None,
        active_player: int | None = None,
    ) -> bool:
        """
        Check if there is a allied token at the end of this line
        """
        if not board:
            board = self.board
        if not active_player:
            active_player = self.active_player
        if dx == dy and dx == 0:
            return False
        k = 1
        nx = x + k * dx
        ny = y + k * dy
        while self.check_inboard(nx, ny) and board[nx][ny] == -active_player:
            k += 1
            nx = x + k * dx
            ny = y + k * dy
        return (
            self.check_inboard(nx, ny)
            and board[nx][ny] == active_player
            and k > 1
        )

    def check_validity_play(
        self,
        x: int,
        y: int,
        board: list | None = None,
        active_player: int | None = None,
    ) -> bool:
        if not board:
            board = self.board
        if not active_player:
            active_player = self.active_player

        if board[x][y] != 0:
            return False
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (
                    self.check_inboard(x + i, y + j)
                    and board[x + i][y + j] == -active_player
                    and self.check_line(x, y, i, j, board, active_player)
                ):
                    return True
        return False

    def check_inboard(self, x: int, y: int) -> bool:
        return 0 <= x < self.n and 0 <= y < self.n

    def play(self, x: int, y: int) -> None:
        if not self.check_validity_play(x, y):
            raise InvalidPlay
        self.board[x][y] = self.active_player
        for i in range(-1, 2):
            for j in range(-1, 2):
                k = 1
                nx = x + k * i
                ny = y + k * j
                if self.check_line(x, y, i, j):
                    while (
                        self.check_inboard(nx, ny)
                        and self.board[nx][ny] == -self.active_player
                    ):
                        self.board[nx][ny] = self.active_player
                        k += 1
                        nx = x + k * i
                        ny = y + k * j

        self.active_player *= -1
        if self.display:
            print(self)

    def available_plays(
        self, board: list | None = None, active_player: int | None = None,
    ) -> list[tuple[int, int]]:
        if not board:
            board = self.board
        if not active_player:
            active_player = self.active_player
        plays = []
        for i in range(self.n):
            for j in range(self.n):
                if self.check_validity_play(i, j, board, active_player):
                    plays.append((i, j))
        return plays

    def score(self) -> int:
        score = 0
        for i in range(self.n):
            for j in range(self.n):
                score += self.board[i][j]
        return score

    def check_end(self, already_passed=False) -> bool:
        if self.available_plays() != []:
            return False
        else:
            self.active_player *= -1
            if not already_passed:
                return self.check_end(True)
            return True

    def human_input(self) -> tuple[int, int]:
        try:
            play = input("Enter your play with this format x y :")
            if play == "exit":
                raise Exit
            x, y = map(int, play.split())
        except Exit:
            print(f"The game ended with a score of {self.score()}")
        except Exception:
            x, y = self.human_input()
        return x, y

    def game(
        self,
        f1: Callable[[], tuple[int, int]] | None = None,
        f2: Callable[[], tuple[int, int]] | None = None,
        start_over: bool = True,
    ) -> tuple[int, int]:
        """
        Plays a game of Othello

        :param self:
        :param f1: Decision function for X
        :type f1: Callable[[], tuple[int, int]] | None
        :param f2: Decision function for X
        :type f2: Callable[[], tuple[int, int]] | None
        :param start_over: Whether or not we start a new game
        :type start_over: bool
        :return: The winner : 1 if O won and -1 if X won and the relative score
        :rtype: tuple[int, int]
        """
        if start_over:
            self.wipe()
        if not f1:
            self.display = True
            f1 = self.human_input
        if not f2:
            self.display = True
            f2 = self.human_input
        players = {-1: f1, 1: f2}
        while not self.check_end():
            x, y = players[self.active_player]()
            try:
                self.play(x, y)
            except InvalidPlay:
                if players[self.active_player] == self.human_input:
                    print("Illegal play")
                else:
                    raise InvalidPlay
        score = self.score()
        if self.display:
            if score > 0:
                print(f"O wins ! With a score of {score}")
            elif score < 0:
                print(f"X wins ! With a score of {score}")
            else:
                print("A draw O.o !!!")
        return ((score > 0) - (score < 0), score)
