import random
from pqdict import pqdict
import operator
from functools import reduce
import itertools
import copy
from util import dfs, ncr

SAMPLE_MAX_SIZE = 2 ** 16
OPT_SIZE = 25
FLAG = 0
OPEN = 1


class CSP_AI:
    """
    A class representing the CSP AI, solving the Minesweeper game using backtracking.
    """

    def __init__(self, height=8, width=8, no_of_mines_in_board=0, use_most_constrained=True, use_probability=True):
        # loading pattern db cache
        self._constrained_variables = pqdict(reverse=True)
        # Set initial height and width
        self.height = height
        self.width = width
        # Keep track of which cells have been clicked on
        self.opened_cells = set()
        self.open_information = dict()
        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.prev_state = []
        self.first_move = True
        self.no_of_mines = no_of_mines_in_board
        self.safes = set()
        self.surely_safe = set()
        self.linked_tiles = list()
        self.last_random = None
        self.disjoint_dict = dict()
        self._use_most_constrained = use_most_constrained
        self._open_count = dict()
        self.closed_cells = set()
        self.use_probability = use_probability
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in self.mines and (i, j) not in self.opened_cells:
                    self.closed_cells.add((i, j))

    def get_cell_neighbors(self, cell):
        """
        a function returning a list of neighbors of a given cell
        :param cell: a tuple of (x,y) to find his neighbors
        :return: list of neighbors
        """
        i, j = cell
        neighbors = []
        for row in range(i - 1, i + 2):
            for col in range(j - 1, j + 2):
                if (0 <= row < self.height) \
                        and (0 <= col < self.width) \
                        and (row, col) != cell:
                    neighbors.append((row, col))
        return neighbors

    def extend_knowledge_base(self, point, no_of_mines):
        """
        a function that adds knowledge according to the board, after opening a tile it gives the number that
        appears on the tile.
        :param point: the (x,y) we recently opened
        :param no_of_mines: the number appeared on the tile
        :return: None, updates the class.
        """
        if self._use_most_constrained and point in self._constrained_variables:
            del self._constrained_variables[point]
        self.open_information[point] = no_of_mines
        if no_of_mines == 0:
            for neighbor in self.get_cell_neighbors(point):
                if neighbor not in self.mines and neighbor not in self.opened_cells:
                    self.surely_safe.add(neighbor)
        elif self._use_most_constrained:
            for neighbor in self.get_cell_neighbors(point):
                if neighbor not in self.opened_cells and neighbor not in self.mines:
                    if neighbor in self._constrained_variables:
                        self._constrained_variables[neighbor] += 1
                    else:
                        self._constrained_variables[neighbor] = 1
        if point not in self.opened_cells:
            self.opened_cells.add(point)
            self.closed_cells.remove(point)
        if point in self.safes:
            self.safes.remove(point)
        if point in self.surely_safe:
            self.surely_safe.remove(point)
        self.first_move = False

    def is_complete_assignment(self, new_cell, opened_cells, flags):
        """
        a function that check if the current assignment of new_cell to the board creates a complete assignment,
        meaning the board rules are not violated
        :param new_cell: the new cell we added.
        :param opened_cells: the opened cells in the current board
        :param flags: the current flags in the current board
        :return: True if it is complete, false otherwise
        """
        for neighbor in self.get_cell_neighbors(new_cell):
            for cell in opened_cells:
                if cell == neighbor:
                    if cell in self.opened_cells and not self.check_mines_consistency(self.open_information[cell],
                                                                                      neighbor,
                                                                                      flags, opened_cells):
                        return False
        return True

    def check_mines_consistency(self, number_of_mines_around, cell, flags, opened_cells):
        """
        Checks for Each cell - if #mines around cell < #mines flagged - isn't complete
        Checks for each cell - if #mines around cell > #mines flagged - isn't complete
        :param cell: the cell to check consistency in
        :param number_of_mines_around: the number on the cell
        :param flags: the flags on thr board
        :param opened_cells: the opened cells on the board
        :return: True if the cell is consistent, false otherwise.
        """

        number_of_flags = 0
        opened_neighbors = 0
        cells_neighbors = self.get_cell_neighbors(cell)
        no_of_neighbors = len(cells_neighbors)
        for neighbor in cells_neighbors:
            if neighbor in flags:
                number_of_flags += 1
            if neighbor in opened_cells:
                opened_neighbors += 1
        if number_of_mines_around < number_of_flags or (
                no_of_neighbors - opened_neighbors + number_of_flags) < number_of_mines_around:
            return False  # this is bad
        return True

    def select_unassigned_variable(self, opened_cells=None, flags=None, tiles_to_check=None, recursion=False):
        """
        a function returning a safe move, if exists, otherwise returns None
        :param flags: the flags on thr board
        :param opened_cells: the opened cells on the board
        :param tiles_to_check: tiles to check if we have a safe move in
        :param recursion: True if part of the recursion, false otherwise.
        :return: (cell, operation) if the is a safe move, None, None otherwise.
        """
        if not self.closed_cells:
            return None, None
        self.linked_tiles = []
        to_open = set()
        to_flag = set()
        added_flags = False
        if opened_cells is None and flags is None:
            flags = self.mines
            opened_cells = self.opened_cells
        for cell in self.opened_cells:
            self.add_linked_tile(cell, opened_cells, flags)
        if tiles_to_check is not None:
            for tile in tiles_to_check:
                if tile not in opened_cells and tile not in flags and tile not in self.mines:
                    if self.check_for_safe_flag(tile, opened_cells, flags):
                        if recursion:
                            to_flag.add(tile)
                            flags.add(tile)
                        else:
                            self.mines.add(tile)
                            flags.add(tile)
                            self.closed_cells.remove(tile)
                            added_flags = True
                    elif self.check_for_safe_opening(tile, opened_cells, flags):
                        if recursion:
                            to_open.add(tile)
                        else:
                            self.safes.add(tile)
        else:
            for cell in opened_cells:
                for neighbor in self.get_cell_neighbors(cell):
                    if neighbor not in opened_cells and neighbor not in flags and neighbor not in self.mines:
                        if self.check_for_safe_flag(neighbor, opened_cells, flags):
                            self.mines.add(neighbor)
                            flags.add(neighbor)
                            self.closed_cells.remove(neighbor)
                            added_flags = True
                        elif self.check_for_safe_opening(neighbor, opened_cells, flags):
                            self.safes.add(neighbor)
        if recursion:
            return to_open, to_flag
        if self.safes:
            return self.safes.pop(), OPEN
        if added_flags:
            return self.select_unassigned_variable(opened_cells, flags, tiles_to_check, recursion)
        return None, None

    def add_linked_tile(self, cell, opened_cells, flags):
        """
        a function that finds and add the linked tiles to the linked tiles sets list.
        :param cell: an open cell to check linked neighbors
        :param opened_cells: the opened cells on the board
        :param flags: the flags on the board
        :return: None
        """
        no_of_actual_mines_around = self.open_information[cell]
        if no_of_actual_mines_around == 0:
            return
        no_of_flags_around = len(
            [neighbor_of_neighbor for neighbor_of_neighbor in self.get_cell_neighbors(cell) if
             neighbor_of_neighbor in flags])
        closed_around = [neighbor_of_neighbor for neighbor_of_neighbor in self.get_cell_neighbors(cell) if
                         neighbor_of_neighbor not in opened_cells and neighbor_of_neighbor not in flags
                         ]
        if no_of_actual_mines_around - no_of_flags_around == len(closed_around) - 1:
            self.linked_tiles.append(closed_around)

    def check_for_safe_flag(self, cell, opened_cells, flags):
        """
        a function that checks for safe flags on the board
        :param cell: a closed cell to check open neighbors
        :param opened_cells: the opened cells on the board
        :param flags: the flags on the board
        :return: True if this cell is safe flag, false otherwise
        """
        for neighbor in self.get_cell_neighbors(cell):
            if neighbor in self.opened_cells:
                no_of_actual_mines_around = self.open_information[neighbor]
                no_of_flags_around = len(
                    [neighbor_of_neighbor for neighbor_of_neighbor in self.get_cell_neighbors(neighbor) if
                     neighbor_of_neighbor in flags])
                no_of_closed_around = [neighbor_of_neighbor for neighbor_of_neighbor in
                                       self.get_cell_neighbors(neighbor) if
                                       neighbor_of_neighbor not in opened_cells and neighbor_of_neighbor not in flags]
                if len(no_of_closed_around) + no_of_flags_around == no_of_actual_mines_around:
                    return True
                for linked in self.linked_tiles:
                    if cell in linked:
                        continue
                    sure_flags = -1
                    for c in linked:
                        if c in no_of_closed_around:
                            sure_flags += 1
                    if sure_flags > 0:
                        if len(linked) == 2 and len(
                                no_of_closed_around) - 1 + no_of_flags_around == no_of_actual_mines_around:
                            return True
                        if len(no_of_closed_around) + no_of_flags_around == no_of_actual_mines_around - sure_flags:
                            return True
        return False

    def check_for_safe_opening(self, cell, opened_cells, flags):
        """
        a function that checks for safe open on the board
        :param cell: a closed cell to check open neighbors
        :param opened_cells: the opened cells on the board
        :param flags: the flags on the board
        :return: True if this cell is safe open, false otherwise
        """
        for neighbor in self.get_cell_neighbors(cell):
            if neighbor in self.opened_cells:
                no_of_actual_mines_around = self.open_information[neighbor]
                no_of_flags_around = len(
                    [neighbor_of_neighbor for neighbor_of_neighbor in self.get_cell_neighbors(neighbor) if
                     neighbor_of_neighbor in flags])
                if no_of_flags_around == no_of_actual_mines_around:
                    return True
                no_of_closed_around = [neighbor_of_neighbor for neighbor_of_neighbor in
                                       self.get_cell_neighbors(neighbor) if
                                       neighbor_of_neighbor not in opened_cells and neighbor_of_neighbor not in flags]
                for linked in self.linked_tiles:
                    if cell in linked:
                        continue
                    sure_flags = -1
                    for c in linked:
                        if c in no_of_closed_around:
                            sure_flags += 1
                    if sure_flags > 0 and no_of_flags_around == no_of_actual_mines_around - sure_flags:
                        return True
        return False

    def make_random_move(self):
        """
        a function that runs a probability move, or random move(if first)
        :return: cell if finds a move, None if not.
        """
        if self.first_move or len(self.mines) == self.no_of_mines or not self.use_probability:
            if len(self.closed_cells) == 0:
                return None
            move = random.choice(tuple(self.closed_cells))
            self.opened_cells.add(move)
            self.closed_cells.remove(move)
            self.first_move = False
            return move
        else:
            cell, oper = self.use_prob_heuristics()
            if self.last_random == cell:
                self.last_random = None
                move = self.simple_probability_heuristics()
                self.opened_cells.add(move)
                self.closed_cells.remove(move)
                return move
            if oper == OPEN:
                if cell == "close":
                    cell = self.closed_cells.pop()
                else:
                    self.closed_cells.remove(cell)
                self.opened_cells.add(cell)
                return cell
            else:
                self.last_random = cell
                for mine in cell:
                    if mine == "close":
                        self.mines.add(self.closed_cells.pop())
                        continue
                    self.mines.add(mine)
                    self.closed_cells.remove(mine)
                safe = self.make_safe_move()
                if safe is None:
                    return self.make_random_move()
                return safe

    def use_prob_heuristics(self):
        """
        A function that returns a probability choice of the next flag to put on the board.
        if the function finds a cell with probability of bomb of 0, it returns to open it,
        otherwise it returns the cell with the highest bomb probability.
        :return: tuple of :a list of cells to put flags in, FLAG
        or a tuple of: cell, OPEN
        """
        combinations = list()
        disjoint_sets = self.find_disjoint_sets()
        disjoint_sets.sort(key=len)
        combinations_dict = dict()
        closed = self.closed_cells.copy()
        cell, oper = self.find_combinations(closed, combinations, combinations_dict, disjoint_sets)
        if cell is not None:
            self.prev_state.pop(-1)
            return cell, oper
        total_combinations = self.update_combinations_dict(closed, combinations, combinations_dict)
        if not closed:
            del combinations_dict["close"]
        if closed and combinations_dict["close"] == 0:
            self.prev_state.pop(-1)
            self.safes = closed
            return self.safes.pop(), OPEN
        arg_max = max(combinations_dict.items(), key=operator.itemgetter(1))[0]
        arg_min = min(combinations_dict.items(), key=operator.itemgetter(1))[0]
        arg_min_prob = combinations_dict[arg_min] / total_combinations
        not_arg_max_prob = 1 - combinations_dict[arg_max] / total_combinations
        if not_arg_max_prob < arg_min_prob:
            if self.last_random and self.last_random is not None and arg_max == self.last_random.pop():
                self.prev_state.pop(-1)
                return arg_min, OPEN
            return [arg_max], FLAG
        else:
            self.prev_state.pop(-1)
            return arg_min, OPEN

    def find_combinations(self, closed, combinations, combinations_dict, disjoint_sets):
        """
        a function that finds all the combination of legal assignments to the disjoint sets separately.
        :param closed: the closed cells that we have on board, not including the disjoint_sets's cells.
        :param combinations: the list of combinations to add to
        :param combinations_dict: a dict of apperences of mines on cells.
        :param disjoint_sets: the disjoint sets.
        :return: (cell, open) or (list_of_cells, flag) if finds a safe move(probability 1 or 0)
        None, None if did not find a safe move.
        """
        for cell_set in disjoint_sets:
            if len(cell_set) > OPT_SIZE:
                return self.simple_probability_heuristics(), OPEN
            for cell in cell_set:
                combinations_dict[cell] = 0
                closed.remove(cell)
            mines_sets = list()
            self.probability_heuristic(self.mines.copy(), self.opened_cells.copy(), mines_sets, cell_set.copy())
            # If no mine possible, meaning we can open it safely
            if not mines_sets or len(mines_sets[0]) == 0:
                self.safes = cell_set
                return self.safes.pop(), OPEN
            intersect = reduce(set.intersection, mines_sets)
            # If a mine appeared in all possibilities, it is a sure mine.
            if len(intersect) != 0 and (self.last_random is None or intersect != self.last_random):
                return intersect, FLAG
            joint = cell_set - reduce(set.union, mines_sets)
            # if a cell did not appear in any mine set, he cant be a mine.
            if len(joint) != 0:
                self.safes = joint
                return self.safes.pop(), OPEN
            combinations.append(mines_sets.copy())
        return None, None

    def simple_probability_heuristics(self):
        """
        a function that calculates the probability for a bomb, but this function calculates the function's probability
        in a simple way, checking min((number - flags_around)/closed_around)
        :return: a cell to open.
        """
        real_num_dict = dict()
        for cell in self.disjoint_dict.keys():
            flag_num = 0
            for neighbor in self.get_cell_neighbors(cell):
                if neighbor in self.mines:
                    flag_num += 1
            real_num_dict[cell] = self.open_information[cell] - flag_num
        min_key = min(self.disjoint_dict, key=lambda k: (real_num_dict[k] / len(self.disjoint_dict[k])))
        c = self.disjoint_dict[min_key].pop()
        return c

    def update_combinations_dict(self, closed, combinations, combinations_dict):
        """
        a function that updates the amount of combination that each cell has mine in, according to the combinations list
        using cartesian multiplexing of the combinations of different disjoint sets,
        and updates the amount of combinations
        that each cell has a bomb.
        :param closed: the closed cells that we have on board, not including the disjoint_sets's cells.
        :param combinations: the list of combinations to add to
        :param combinations_dict: a dict of appearances of mines on cells
        :return: the number of total combinations possible
        """
        combinations_dict["close"] = 0
        counter = 0
        total_combinations = 0
        for element in itertools.product(*combinations):
            counter += 1
            if counter > SAMPLE_MAX_SIZE:
                return total_combinations
            mines_num = 0
            for item in element:
                mines_num += len(item)
            if mines_num + len(self.mines) > self.no_of_mines:
                continue
            elif mines_num + len(self.mines) == self.no_of_mines:
                for item in element:
                    for mine in item:
                        combinations_dict[mine] += 1
                total_combinations += 1
            else:
                mine_remaining = self.no_of_mines - mines_num - len(self.mines)
                ncr_open = ncr(len(closed), mine_remaining)
                ncr_closed = ncr(len(closed) - 1, mine_remaining - 1)
                for item in element:
                    for mine in item:
                        combinations_dict[mine] += ncr_open
                combinations_dict["close"] += ncr_closed
                total_combinations += ncr_open
        return total_combinations

    def probability_heuristic(self, flags, opened, mines_sets, disjoint_set, last_added=None):
        """
        a function that finds the legal assignments to the disjoint set.
        :param flags: Initialized with self.mines.copy
        :param opened: Initialized with self.opened.copy
        :param mines_sets: the current assignment of mines to the disjoint set
        :param disjoint_set: the current set we check legal assignments to
        :param last_added: the last cell added, on first run = None
        :return: None, but updates the mines_sets.
        """
        if last_added is not None:
            if not self.is_complete_assignment(last_added, opened, flags):
                return
            to_open, to_flag = self.select_unassigned_variable(opened, flags, disjoint_set, True)
            while to_open or to_flag:
                for o in to_open:
                    if not self.is_complete_assignment(o, opened, flags):
                        return
                    opened.add(o)
                    disjoint_set.remove(o)
                for f in to_flag:
                    if not self.is_complete_assignment(f, opened, flags):
                        return
                    flags.add(f)
                    disjoint_set.remove(f)
                to_open, to_flag = self.select_unassigned_variable(opened, flags, disjoint_set, True)
        if len(disjoint_set) == 0:
            if last_added is not None:
                if not self.is_complete_assignment(last_added, opened, flags):
                    return
            mine_comp = flags - self.mines
            if mine_comp not in mines_sets:
                mines_sets.append(flags - self.mines)
            return
        neighbor = disjoint_set.pop()
        if neighbor not in self.opened_cells.union(opened).union(flags).union(self.mines):
            flags.add(neighbor)
            self.probability_heuristic(flags.copy(), opened.copy(), mines_sets, disjoint_set.copy(), neighbor)
            flags.remove(neighbor)
            opened.add(neighbor)
            self.probability_heuristic(flags.copy(), opened.copy(), mines_sets, disjoint_set.copy(),
                                       neighbor)

    def find_disjoint_sets(self):
        """
        a function that finds disjoint tiles, A.K.A tiles that does not affect each other.
        :return: a list of disjoint closed cells sets.
        """
        graph = dict()
        disjoint_sets = []
        opens = []
        for open_cell in self.opened_cells:
            graph[open_cell] = set()
            opens.append(open_cell)
            for neighbor in self.get_cell_neighbors(open_cell):
                if neighbor not in self.opened_cells and neighbor not in self.mines:
                    graph[open_cell].add(neighbor)
            if not graph[open_cell]:
                graph.pop(open_cell)
                opens.remove(open_cell)
        self.disjoint_dict = copy.deepcopy(graph)
        while opens:
            disjoint_k, disjoint_v = dfs(graph, opens[0], set(), set())
            for k in disjoint_k:
                graph.pop(k)
                opens.remove(k)
            disjoint_sets.append(disjoint_v)
        return disjoint_sets

    def make_safe_move(self):
        """
        a function that gives a safe move to make on the board, if she does not find, she returnes a probability move.
        when a probability or random move occurs the saves the board, and then if inconsistency in the board appears,
        it returns to the first saved board and erases the last_board field.
        :return: tuple (x,y) cell to be opened.
        """
        mines = self.mines.copy()
        to_backtrack = False
        if self.first_move:
            return None
        if self.surely_safe:
            c = self.surely_safe.pop()
            self.opened_cells.add(c)
            self.closed_cells.remove(c)
            return c
        if self.safes:
            c = self.safes.pop()
            if self.is_complete_assignment(c, self.opened_cells, self.mines):
                self.opened_cells.add(c)
                self.closed_cells.remove(c)
                return c
            to_backtrack = True
        if self.prev_state:
            var, move = self.select_unassigned_variable(self.opened_cells, self.prev_state[0][0].copy())
            if var is None:
                var, move = self.select_unassigned_variable()
        else:
            if self._use_most_constrained:
                most_costrained = self._constrained_variables.keys()
                most_costrained_list = list(most_costrained)
                var, move = self.select_unassigned_variable(self.opened_cells, self.mines, most_costrained_list)
            else:
                var, move = self.select_unassigned_variable()
        if var is None:
            if len(self.mines) > len(mines):
                return self.make_safe_move()
            self.prev_state.append((self.mines.copy(), self.closed_cells.copy()))
            return None
        opened = self.opened_cells.copy()
        flags = self.mines.copy()
        if var in flags:
            to_backtrack = True
        if move == OPEN:
            opened.add(var)
        elif move == FLAG:
            flags.add(var)
        if not to_backtrack and self.is_complete_assignment(var, opened, flags):
            if move == FLAG:
                self.mines.add(var)
                self.closed_cells.remove(var)
                return self.make_safe_move()
            elif move == OPEN:
                self.opened_cells.add(var)
                self.closed_cells.remove(var)
                return var
        self.mines, self.closed_cells = self.prev_state[0]
        for op in self.opened_cells:
            if op in self.closed_cells:
                self.closed_cells.remove(op)
        self.prev_state = []
        self.safes = set()
        return self.make_safe_move()
