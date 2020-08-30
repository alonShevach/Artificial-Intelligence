from board import Board
from search import SearchProblem, Node, build_actions
import util
import math
import heapq
from pieces import Piece

#############
# Constants #
#############
FREE = -1
ONE_PLAYER = 1


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, ONE_PLAYER, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in
                state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)


class BlokusCornersProblem(SearchProblem):
    """
    This class represents the Blokus Corners Problem
    The main purpose is to cover three corners most effectively
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        self.board = Board(board_w, board_h, ONE_PLAYER, piece_list, starting_point)
        self.board_w = board_w
        self.board_h = board_h
        self.corners = [(0, 0), (0, board_w - 1),
                        (board_h - 1, 0), (board_h - 1, board_w - 1)]
        self.targets = self.corners.copy()
        self.piece_list = piece_list

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        :param state: the current state
        :return: True if and only if the state is a valid goal state
        """
        corners = [state.get_position(0, 0), state.get_position(0, -1),
                   state.get_position(-1, 0), state.get_position(-1, -1)]
        if -1 in corners:
            return False
        return True

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for
                move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        :param actions: list of actions
        :return: the total cost of a particular sequence of actions
        """
        sum = 0
        for act in actions:
            sum += act.piece.get_num_tiles()
        return sum


def blokus_corners_heuristic(state, problem):
    """
    The heuristic for the BlokusCornersProblem.
    It uses two other heuristic functions for a better result
    :param state: the current state
    :param problem: the current problem
    :return: the maximum level of two heuristic functions
    """
    h1 = discrete_space(state, problem)
    h2 = heuristic_smallest_dist(state, problem)
    h3 = corner_heuristics(state, problem)
    return max(h1, h2, h3)


class BlokusCoverProblem(SearchProblem):
    """
        This class represents the Blokus Cover Problem
        The main purpose is to cover three corners most effectively
        """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.targets = targets.copy()
        self.expanded = 0
        self.board = Board(board_w, board_h, ONE_PLAYER, piece_list, starting_point)
        self.board_w = board_w
        self.board_h = board_h
        self.piece_list = piece_list

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        :param state: the current state
        :return: True if and only if the state is a valid goal state
        """
        for target in self.targets:
            if state.get_position(target[1], target[0]) == FREE:
                return False
        return True

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for
                move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        cost = 0
        for act in actions:
            cost += act.piece.get_num_tiles()
        return cost


def blokus_cover_heuristic(state, problem):
    """
    The heuristic for the BlokusCoverProblem.
    It uses two other heuristic functions for a better result
    :param state: the current state
    :param problem: the current problem
    :return: the maximum level of two heuristic functions
    """
    h1 = discrete_space(state, problem)
    h2 = heuristic_smallest_dist(state, problem)
    return max(h1, h2)


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0),
                 targets=(0, 0)):
        self.board_h = board_h
        self.board_w = board_w
        self.expanded = 0
        self.targets = targets.copy()
        self.board = Board(board_w, board_h, ONE_PLAYER, piece_list, starting_point)
        self.piece_list = piece_list
        self.starting_point = starting_point

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in
                state.get_legal_moves(0)]

    def is_goal_state(self, state):
        """
        :param state: the current state
        :return: True if and only if the state is a valid goal state
        """
        for target in self.targets:
            if state.get_position(target[1], target[0]) == FREE:
                return False
        return True

    def solve(self):
        """
        This functions cover all the given locations
        This time we trade optimality for speed.
        :return: list of actions that covers all target locations on the board
        """
        root = self.board.__copy__()
        targets_to_cover = []
        actions = []
        for target in self.targets:  # targets left to cover
            target_dist = root.get_position(target[1], target[0])
            if target_dist == FREE:
                targets_to_cover.append(target)
        while targets_to_cover:
            min_tile, min_target = min_target_tile(root, self)  # find the nearest tile and target
            found = False
            for succ, act, cost in self.get_successors(root):
                to_continue = False
                for target in targets_to_cover:
                    # if successor has no legal moves to the target
                    if succ.get_position(target[1], target[0]) == FREE and not succ.check_tile_legal(0, target[1],
                                                                                                     target[0]):
                        to_continue = True
                        break
                if to_continue:
                    continue
                # if successor covers the target:
                if succ.get_position(min_target[1], min_target[0]) != FREE:
                    actions.append(act)
                    root = succ
                    found = True
                    break
            # if did not find the successors that covers the target, choose the nearest successor to a target
            if not found:
                for succ, act, cost in self.get_successors(root):
                    if ((min_target[0] <= act.x <= min_tile[0]) or (min_target[0] >= act.x >= min_tile[0])) and (
                            (min_target[1] <= act.y <= min_tile[1]) or (min_target[1] >= act.y >= min_tile[1])):
                        actions.append(act)
                        root = succ
                        found = True
                        break
            # if did not find a solution before, make the A* search with heuristic function that
            # does not consider the path cost
            if not found:
                try:
                    succ, act, cost = self.get_successors(root)[0]
                except:
                    s = BlokusCoverProblem(self.board_w, self.board_h, self.piece_list, self.starting_point,
                                           self.targets)
                    return closest_location_search(s, blokus_cover_heuristic)
                actions.append(act)
                root = succ
            targets_to_cover = []
            for target in self.targets:
                target_dist = root.get_position(target[1], target[0])
                if target_dist == -1:
                    targets_to_cover.append(target)
        return actions


class MiniContestSearch:
    """
        In this problem you have to cover all given positions on the board,
        but the objective is speed, not optimality.
        """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0),
                 targets=(0, 0)):
        self.board_h = board_h
        self.board_w = board_w
        self.expanded = 0
        self.targets = targets.copy()
        self.board = Board(board_w, board_h, ONE_PLAYER, piece_list, starting_point)
        self.piece_list = piece_list
        self.starting_point = starting_point

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in
                state.get_legal_moves(0)]

    def is_goal_state(self, state):
        """
        :param state: the current state
        :return: True if and only if the state is a valid goal state
        """
        for target in self.targets:
            if state.get_position(target[1], target[0]) == FREE:
                return False
        return True

    def solve(self):
        """
        This functions cover all the given locations
        :return: list of actions that covers all target locations on the board
        """
        closestLocSearch = ClosestLocationSearch(self.board_w, self.board_h, self.piece_list, self.starting_point,
                                                 self.targets)
        return closestLocSearch.solve()


####################
# Helper Functions #
####################
def closest_location_search(problem, heuristic):
    """
    An A* searc that uses the heuristic function that does not consider the path cost
    :param problem: ClosestLocationSearch
    :param heuristic: heuristic function
    :return: a list of actions to make to reach the goal
    """
    queue = util.PriorityQueue()  # Init the queue
    visited = set()
    actions = []
    root = Node(problem.get_start_state())
    h = heuristic(root.state, problem) - root.cost
    root.set_f(h)
    queue.push(root, root.f)
    while not queue.isEmpty():  # while queue is not empty
        node = queue.pop()
        while node.state in visited and not queue.isEmpty():  # remove the same states from queue
            # with worse priority
            node = queue.pop()
        if problem.is_goal_state(node.state):
            actions = build_actions(node)
            return actions
        visited.add(node.state)
        for succ, act, cost in problem.get_successors(node.state):  # Build successors
            if succ not in visited:
                new_node = Node(succ, act, node, cost)
                h = heuristic(new_node.state, problem) - new_node.cost
                new_node.set_f(h)  # Update f(n) value
                queue.push(new_node, new_node.f)
    return actions


def discrete_space(state, problem):
    """
    This heuristic function calculates the value by the amount of free targets left
    :param problem: the game problem
    :param state: the current board
    :return: the heuristic value of the current state
    """
    h = 0
    for target in problem.targets:
        if state.get_position(target[1], target[0]) == -1:
            h += 1
    return h


def chebyshev_distance(position_a, position_b):
    """
    This heuristic function calculates the value by the shortest distance between two coordinates
    :param position_a: the first coordinate
    :param position_b: the second coordinate
    :return: the the heuristic value of the current state
    """
    x = position_a[0] - position_b[0]  # add 1 because the arrays of a board start from 0
    y = position_a[1] - position_b[1]
    return max(abs(x), abs(y)) + 1


def heuristic_smallest_dist(state, problem):
    """
    This heuristic function calculates the value by the shortest distance between the pieces and the target
    :param state: the current state
    :param problem: the game problem
    :return: the the heuristic value of the current state
    """
    for target in problem.targets:
        if not state.check_tile_legal(0, target[1], target[0]) and state.get_position(target[1], target[0]) == -1:
            # i was reading in the forum that it is allowed to return infinity if the state cannot reach the goal.
            return math.inf
    shared_tile = dict()  # for each tile create a list of all targets that can be accessed
    # find all legal tiles
    legal_tiles = find_legal_tiles(problem, shared_tile, state)
    # find the nearest tile for each corner
    targ_dist = dict()
    dist_list = all_smallest_distances(legal_tiles, problem.targets, shared_tile, state, targ_dist)
    longest_dist = 0
    for dist in dist_list:
        if dist != math.inf:
            if dist > longest_dist:
                longest_dist = dist
    # create a sorted list of tiles
    tile_vals = list(shared_tile.values())
    tile_vals.sort(key=len)
    tile_vals.reverse()
    tile_sort = list(shared_tile.keys())
    for k in range(len(legal_tiles)):
        for i in range(len(tile_vals)):
            if tile_vals[i] == shared_tile[k]:
                tile_sort[i] = legal_tiles[k]
    dist_sum = find_targets_cut(state, targ_dist, tile_sort, tile_vals)
    if longest_dist > dist_sum:
        return longest_dist
    return dist_sum


def corner_heuristics(state, problem):
    """
    This function finds the smallest sum of tiles to cover the corners
    :param state: the current state
    :param problem: the current problem
    :return:
    """
    targets_to_cover = []
    for target in problem.targets:
        if not state.check_tile_legal(0, target[1], target[0]) and state.get_position(target[1], target[0]) == -1:
            # i was reading in the forum that it is allowed to return infinity if the state cannot reach the goal.
            return math.inf
        target_dist = state.get_position(target[1], target[0])
        if target_dist == FREE:
            targets_to_cover.append(target)
    smallest = heapq.nsmallest(len(targets_to_cover), state.piece_list, key=Piece.get_num_tiles)
    biggest = heapq.nsmallest(len(targets_to_cover) // 2, state.piece_list, key=Piece.get_num_tiles)
    sum = 0
    for s in smallest:
        sum += s.get_num_tiles()
    if len(targets_to_cover) == 4:
        if biggest[0].get_num_tiles() < state.board_w and biggest[0].get_num_tiles() < state.board_h and biggest[
            1].get_num_tiles() < state.board_w and biggest[1].get_num_tiles() < state.board_h:
            return sum
        return state.board_w + state.board_h - 1
    if len(targets_to_cover) >= 2:
        if biggest[0].get_num_tiles() < state.board_w and biggest[0].get_num_tiles() < state.board_h:
            return sum
        return smallest[0].get_num_tiles()
    if len(targets_to_cover) == 1:
        return smallest[0].get_num_tiles()
    return 0


def find_targets_cut(state, targ_dist, tile_sort, tile_vals):
    """
    a function that finds the distances according to the target's cut.
    :param state:  the current state
    :param targ_dist: the distance between the target to the tile.
    :param tile_sort: a sorted list of tiles, sorted by the amount of targets that can get min distance from it
    :param tile_vals: a list of list of targets, sorted by the length of the list
    :return: the distance sum of the biggest cut.
    """
    visited_corners = set()
    dist_sum = 0
    for j, corners in enumerate(tile_vals):
        # if there are more than 1 target that are located at the same distance from the current tile,
        # calculate the common distance only once and delete duplicates
        not_visited = []
        for c in corners:
            if c not in visited_corners:
                not_visited.append(c)
        if len(not_visited) > 1:
            closest_corner = 0
            new_corners = []
            s = 1
            for corner in not_visited:
                new_corners.append(corner)
                s += targ_dist[corner] - 1
                if targ_dist[corner] < closest_corner:
                    closest_corner = targ_dist[corner]
                visited_corners.add(corner)
            tile = tile_sort[j]
            if tile in new_corners:
                new_corners.remove(tile)
            dist_sum += check_adj(tile, new_corners, state, s)
        elif len(corners) == 1 and corners[0] not in visited_corners:
            dist_sum += targ_dist[corners[0]]
            visited_corners.add(corners[0])
    return dist_sum


def check_adj_helper(shared_tile, t, targets, state, i=1):
    """
    a helper function for the check_adj function
    :param t: the current tile
    :param targets: the targets we find using this tile
    :param state: the current state of the board
    :param last_dist: the last distance we saw from the last tile
    :param i: and index
    :return: the min distance from the adjacent tiles
    """
    if t in targets:
        targets.remove(t)
    n = all_smallest_distances([t], targets, shared_tile, state, dict())
    return sum(n) - len(targets) + 1 + i


def check_adj(tile, targets, state, last_dist, i=1):
    """
    :param tile: the current tile
    :param targets: the targets we find using this tile
    :param state: the current state of the board
    :param last_dist: the last distance we saw from the last tile
    :param i: and index
    :return: the min distance from the adjacent tiles
    """
    shared_tile = dict()
    shared_tile[0] = []
    if state.check_tile_legal(0, tile[1] - 1, tile[0] - 1):
        t = (tile[1] - 1, tile[0] - 1)
        n = check_adj_helper(shared_tile, t, targets, state, i)
        if n < last_dist:
            return check_adj(t, targets, state, n, i + 1)
    if state.check_tile_legal(0, tile[1], tile[0] - 1):
        t = (tile[1], tile[0] - 1)
        n = check_adj_helper(shared_tile, t, targets, state, i)
        if n < last_dist:
            return check_adj(t, targets, state, n, i + 1)
    if state.check_tile_legal(0, tile[1] - 1, tile[0]):
        t = (tile[1] - 1, tile[0])
        n = check_adj_helper(shared_tile, t, targets, state, i)
        if n < last_dist:
            return check_adj(t, targets, state, n, i + 1)
    if state.check_tile_legal(0, tile[1] + 1, tile[0] + 1):
        t = (tile[1] + 1, tile[0] + 1)
        n = check_adj_helper(shared_tile, t, targets, state, i)
        if n < last_dist:
            return check_adj(t, targets, state, n, i + 1)
    if state.check_tile_legal(0, tile[1], tile[0] + 1):
        t = (tile[1], tile[0] + 1)
        n = check_adj_helper(shared_tile, t, targets, state, i)
        if n < last_dist:
            return check_adj(t, targets, state, n, i + 1)
    if state.check_tile_legal(0, tile[1] + 1, tile[0]):
        t = (tile[1] + 1, tile[0])
        n = check_adj_helper(shared_tile, t, targets, state, i)
        if n < last_dist:
            return check_adj(t, targets, state, n, i + 1)
    if state.check_tile_legal(0, tile[1] + 1, tile[0] - 1):
        t = (tile[1] + 1, tile[0] - 1)
        n = check_adj_helper(shared_tile, t, targets, state, i)
        if n < last_dist:
            return check_adj(t, targets, state, n, i + 1)
    if state.check_tile_legal(0, tile[1] - 1, tile[0] + 1):
        t = (tile[1] - 1, tile[0] + 1)
        n = check_adj_helper(shared_tile, t, targets, state, i)
        if n < last_dist:
            return check_adj(t, targets, state, n, i + 1)
    return last_dist


def all_smallest_distances(legal_tiles, targets, shared_tile, state, targ_dist):
    """
    This function finds all the smallest distances between nearest tiles and targets
    :param legal_tiles: all legal tiles of the state
    :param problem: the current problem
    :param shared_tile: a dictionary to update
    :param state: the current state
    :return: list of all smallest distances
    """
    dist_list = []
    for target in targets:
        if state.get_position(target[1], target[0]) != -1:
            dist_list.append(math.inf)
            continue
        min_dist = False
        best_tiles = []
        for j, tile in enumerate(legal_tiles):
            if not min_dist:  # first time
                min_dist = chebyshev_distance((tile[1], tile[0]), target)
                best_tiles = [j]
            else:
                curr_dist = chebyshev_distance((tile[1], tile[0]), target)  # calculate the shortest distance
                if curr_dist < min_dist:  # update the minimal best distance
                    min_dist = curr_dist
                    best_tiles = [j]
                elif curr_dist == min_dist:  # there are more than 1 target at the same distance from the current tile
                    best_tiles.append(j)
        for t in best_tiles:
            shared_tile[t].append(target)
        dist_list.append(min_dist)
        targ_dist[target] = min_dist
    return dist_list


def find_legal_tiles(problem, shared_tile, state):
    """
    This function find all legal tiles
    :param problem: the current problem
    :param shared_tile: common tiles dictionary to update
    :param state: the current state
    :return: a list of all legal tiles
    """
    index = 0
    legal_tiles = []
    for row in range(problem.board_h):
        for col in range(problem.board_w):
            if state.check_tile_attached(0, col, row) and state.check_tile_legal(0, col, row):
                legal_tiles.append((col, row))
                shared_tile[index] = []
                index += 1
    return legal_tiles


def min_target_tile(state, problem):
    """
    This function finds tile and target with the smallest distance between them
    :param state: the current state
    :param problem: the current problem
    :return: tile, target
    """
    shared_tile = dict()
    legal_tiles = find_legal_tiles(problem, shared_tile, state)
    dist_list = all_smallest_distances(legal_tiles, problem.targets, shared_tile, state, dict())
    min_dist = min(dist_list)  # find the smallest distance
    ind = dist_list.index(min_dist)
    best_target = problem.targets[ind]
    tile_vals = list(shared_tile.values())  # all common tiles
    tile_vals.sort(key=len)
    tile_vals.reverse()
    best_tile = None
    for targets_list in tile_vals:  # find best target to tile
        if best_target in targets_list:
            best_tile = list(shared_tile.keys())[list(shared_tile.values()).index(targets_list)]
            best_tile = legal_tiles[best_tile]
            break
    return best_tile, best_target
