# %%
from tqdm import tqdm
from o_class import Othello
from solver import (
    get_random_player,
    get_random_player_with_rules,
    get_min_max_basic,
    get_min_max_corners,
)

# %%
game = Othello(8, True)
print(game)
# %%
game.display = False
p1 = get_min_max_basic(game, 5)
p2 = get_min_max_corners(game, 7)
# %%
print(game.game(p1, p2))
print(game.game(p2, p1))
# %%
mM3 = get_min_max_basic(game, 3)
p1 = get_min_max_basic(game, 4)
p2 = get_min_max_corners(game, 3)

game.display = False
nb_game = 20


def simulate_games(nb_game, p1, p2):
    nb_win_X = 0
    nb_win_O = 0
    avg_score_X = 0
    avg_score_O = 0
    for i in tqdm(range(nb_game)):
        winner, score = game.game(p1, p2)
        if winner > 0:
            nb_win_O += 1
            avg_score_O = (avg_score_O * i + score) / (i + 1)
        elif winner < 0:
            nb_win_X += 1
            avg_score_X = (avg_score_X * i + -score) / (i + 1)
    print(f"X won {nb_win_X} game with an average score of {avg_score_X}")
    print(f"O won {nb_win_O} game with an average score of {avg_score_O}")


# %%
