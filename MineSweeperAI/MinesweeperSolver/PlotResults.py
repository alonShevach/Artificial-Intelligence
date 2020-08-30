from SAT_AI import SAT_AI
from CSP_AI import CSP_AI
from MineSweeperGame import Minesweeper
import numpy as np
from pandas import DataFrame
from plotnine import *
import time


def get_rates_for_ai(ai_type, no_of_iterations, difficulty: str, p1, p2) -> tuple:
    assert difficulty in ['beginner', 'medium', 'expert']
    if difficulty == "beginner":
        HEIGHT, WIDTH, NO_OF_MINES = 9, 9, 10
    elif difficulty == "medium":
        HEIGHT, WIDTH, NO_OF_MINES = 16, 16, 40
    else:
        HEIGHT, WIDTH, NO_OF_MINES = 16, 30, 99
    won_counter = 0
    runtime = 0
    for i in range(no_of_iterations):
        start = time.time()
        print('({}/{})'.format(str(i + 1), str(no_of_iterations)))
        expert_board = Minesweeper(HEIGHT, WIDTH, NO_OF_MINES)
        if ai_type == SAT_AI:
            ai = ai_type(height=HEIGHT, width=WIDTH, no_of_mines_in_board=NO_OF_MINES, prob_mode=(p1, p2))
        else:
            assert ai_type == CSP_AI
            ai = ai_type(height=HEIGHT, width=WIDTH, no_of_mines_in_board=NO_OF_MINES,
                         use_most_constrained=(p1 == 1), use_probability=(p2 == 1))
        print("NEXT GAME")
        revealed = set()
        lost = False
        won = False
        while not lost:
            flags = ai.mines.copy()
            if flags == expert_board.mines:
                print("WON")
                won = True
                break
            move = ai.make_safe_move()
            if move is None:
                if ai_type == 'SAT_AI':
                    move = ai.make_random_move()
                else:
                    move = ai.make_random_move()
                if move is None:
                    lost = True
            if move:
                if expert_board.is_mine(move):
                    lost = True
                else:
                    nearby = expert_board.nearby_mines(move)
                    revealed.add(move)
                    ai.extend_knowledge_base(move, nearby)
            runtime += time.time() - start
        if won:
            won_counter += 1
    print(str(ai_type), (won_counter / no_of_iterations), (runtime / no_of_iterations))
    return (won_counter / no_of_iterations), (runtime / no_of_iterations)


def ai_success_rates(no_of_itertions, difficulty) -> DataFrame:
    # SAT
    s1, t1 = get_rates_for_ai(SAT_AI, no_of_itertions, difficulty, 1, 1)
    s2, t2 = get_rates_for_ai(SAT_AI, no_of_itertions, difficulty, 1, 0)
    s3, t3 = get_rates_for_ai(SAT_AI, no_of_itertions, difficulty, 0, 1)
    # CSP
    s4, t4 = get_rates_for_ai(CSP_AI, no_of_itertions, difficulty, 1, 1)
    s5, t5 = get_rates_for_ai(CSP_AI, no_of_itertions, difficulty, 1, 0)
    s6, t6 = get_rates_for_ai(CSP_AI, no_of_itertions, difficulty, 0, 1)

    _data = DataFrame({'type': np.array(['SAT ALL', 'SAT PDB', 'SAT AA', 'CSP ALL', 'CSP MC', 'CSP PROB']),
                       'percentage': np.array([s1, s2, s3, s4, s5, s6]),
                       'average time': np.array([t1, t2, t3, t4, t5, t6])})
    return _data


def plot_rates(data, no_of_games, difficulty: str) -> tuple:
    _plot_success = ggplot(aes(x="type", weight="percentage", fill="type"), data) + geom_bar() \
                    + labs(x="Agent Type", y="Percentage of Success",
                           title="Comparison of Success Rates of Agent Types Over {} Games ({} Difficulty)".format(
                               str(no_of_games), difficulty.capitalize()), fill="Agent Type")

    _plot_time = ggplot(aes(x="type", weight="average time", fill="type"), data) + geom_bar() \
                 + labs(x="Agent Type", y="Average runtime in seconds",
                        title="Comparison of Average Runtimes of Agent Types Over {} Games ({} Difficulty)".format(
                            str(no_of_games), difficulty.capitalize()), fill="Agent Type")
    return _plot_success, _plot_time


def test_and_plot_rates(no_of_games, difficulty: str):
    rates = ai_success_rates(no_of_games, difficulty)
    print(rates)
    plot_success, plot_time = plot_rates(rates, no_of_games, difficulty)
    plot_success.save("success_{}_{}.png".format(difficulty, str(no_of_games)))
    plot_time.save("runtime_{}_{}.png".format(difficulty, str(no_of_games)))


if __name__ == "__main__":
    print("Beginner")
    test_and_plot_rates(50, "beginner")
    print("Medium")
    test_and_plot_rates(30, "medium")
    print("Expert")
    test_and_plot_rates(25, "expert")
