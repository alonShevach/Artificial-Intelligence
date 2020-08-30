from itertools import combinations
from SAT_AI import SAT_AI as MinesweeperAI

import numpy as np


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)
        self.initial_board = self.board.copy()
        # Add mines randomly

    def get_initial_board(self):
        return self.initial_board

    def __str__(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def set_mines(self, mines):
        for i, j in mines:
            self.board[i][j] = True
        self.mines = mines

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1
        return count


def all_boards_iterator(height, width, no_of_mines):
    coords_list = [(i, j) for i in range(height) for j in range(width)]
    for mines_comb in combinations(coords_list, no_of_mines):
        new_board = Minesweeper(height, width, no_of_mines)
        new_board.set_mines(set(mines_comb))
        yield new_board


def append_to_dict(dict, key, value):
    if key not in dict:
        dict[key] = [value]
    else:
        dict[key].append(value)


def unify_dict_values(dict_a, dict_b):
    for k, v in dict_b.items():
        if k in dict_a:
            dict_a[k].extend(v)
        else:
            dict_a[k] = v


def gather_information_for_4x4():
    conf_moves = dict()
    conf_count = dict()
    NO_OF_GAMES_PER_MINES_COMB = 1000
    for no_of_mines in range(1, 16):
        for game in all_boards_iterator(4, 4, no_of_mines):
            for i in range(NO_OF_GAMES_PER_MINES_COMB):
                current_conf_moves = dict()
                print("NEXT GAME")
                ai = MinesweeperAI(height=4, width=4, no_of_mines_in_board=no_of_mines)
                revealed = set()
                lost = False
                won = False
                while not lost:
                    flags = ai.mines.copy()
                    current_configuration = ai.get_configuration()
                    if current_configuration in conf_count:
                        conf_count[current_configuration] += 1
                    else:
                        conf_count[current_configuration] = 1
                    if flags == game.mines:
                        print("WON")
                        won = True
                        break
                    move = ai.make_safe_move()
                    if move is None:
                        move = ai.make_random_move()
                        if move is None:
                            # print("No moves left to make.")
                            lost = True
                        else:
                            pass
                            # print("No known safe moves, AI making random move:", move)
                    else:
                        pass
                        # print("AI making safe move:", move)
                    # Make move and update AI knowledge
                    if move:
                        if game.is_mine(move):
                            lost = True
                        else:

                            append_to_dict(current_conf_moves, current_configuration, move)
                            nearby = game.nearby_mines(move)
                            revealed.add(move)
                            ai.extend_knowledge_base(move, nearby)
                if won:
                    unify_dict_values(conf_moves, current_conf_moves)

    for k, v in conf_moves.items():
        max_value = max(v, key=v.count)
        max_count = v.count(max_value)
        prob = max_count / conf_count[k]
        conf_moves[k] = (max_value, prob)
    np.save('PDB.npy', conf_moves)
    # read_dictionary = np.load('PDB.npy', allow_pickle=True).item()
    # print(read_dictionary.values())  # displays "world"


gather_information_for_4x4()
