import math
import numpy as np
import abc
import util
from game import Agent, Action

#############
# Constants #
#############
OUR_AGENT = 0
OPPONENT = 1
MATRIX_WEIGHT = 17
TILE_WEIGHT = 10000
SMOOTH_WEIGHT = 83.5
WIN_TILE = 2048


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Counts all the differences between all the adjacent tiles
        :param current_game_state: the given game state
        :param action: the current action
        :return: a number, where higher numbers are better.
        """
        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board

        sum_of_diff = 0
        for i in range(len(board)):
            for j in range(len(board[i]) - 1):
                sum_of_diff += abs(board[i][j] - board[i][j + 1])
        for i in range(len(board) - 1):
            for j in range(len(board[i])):
                sum_of_diff += abs(board[i][j] - board[i + 1][j])
        return -sum_of_diff


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        return self.get_action_helper(game_state, True, self.depth * 2)

    def get_action_helper(self, game_state, is_max, depth):
        """
        A helper recursive function that finds the MiniMax action according to the given sub-tree
        with its new depth at each time at a min or max level
        :param game_state: current state
        :param is_max: True iff its a max tree level
        :param depth: the given depth to search
        :return: the MiniMax action
        """
        # base case
        if depth == 0:
            return self.evaluation_function(game_state)
        # max tree level
        if is_max:
            legal_actions = game_state.get_agent_legal_actions()
            if not legal_actions:
                return self.evaluation_function(game_state)
            # build successors
            successors = []
            for act in legal_actions:
                successors.append(game_state.generate_successor(OUR_AGENT, act))
            if depth != self.depth * 2:
                return max([self.get_action_helper(succ, False, depth - 1) for succ in successors])
            # we are at the given depth, so need to find the best action according to best max value
            biggest_act = None
            biggest = 0
            for i in range(len(legal_actions)):
                eval = self.get_action_helper(successors[i], not is_max, depth - 1)
                if biggest_act is None or eval > biggest:
                    biggest_act = legal_actions[i]
                    biggest = eval
            return biggest_act
        # min tree level
        if not is_max:
            # base case
            if depth == 0:
                return self.evaluation_function(game_state)
            legal_actions = game_state.get_opponent_legal_actions()
            if not legal_actions:
                return self.evaluation_function(game_state)
            # build successors
            successors = []
            for act in legal_actions:
                successors.append(game_state.generate_successor(OPPONENT, act))
            return min([self.get_action_helper(succ, True, depth - 1) for succ in successors])


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        :param game_state: The given game state
        :return: The minimax action using self.depth and self.evaluationFunction

        """
        alpha = -math.inf
        beta = math.inf
        return self.get_action_helper(game_state, True, self.depth * 2, alpha, beta)

    def get_action_helper(self, game_state, is_max, depth, alpha, beta):
        """
        A helper recursive function that finds the MiniMax action according to the given sub-tree
        with its new depth at each time and a min or max level
        :param game_state: current state
        :param is_max: True iff its a max tree level
        :param depth: the given depth to search
        :param alpha: minus infinity
        :param beta: infinity
        :return: the MiniMax action
        """
        if depth == 0:
            return self.evaluation_function(game_state)
        # max tree level
        if is_max:
            max_eval = -math.inf
            legal_actions = game_state.get_agent_legal_actions()
            if not legal_actions:
                return self.evaluation_function(game_state)
            successors = []
            # we are at the max level, so we are the agent
            for act in legal_actions:
                successors.append(game_state.generate_successor(OUR_AGENT, act))
            if depth != self.depth * 2:
                for succ in successors:
                    eval = self.get_action_helper(succ, False, depth - 1, alpha, beta)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                return max_eval
            # we are at the given depth, so need to find the best action according to best max value
            biggest_act = None
            biggest = 0
            for i in range(len(legal_actions)):
                eval = self.get_action_helper(successors[i], False, depth - 1, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if biggest_act is None or eval > biggest:
                    biggest_act = legal_actions[i]
                    biggest = eval
                if beta <= alpha:
                    break
            return biggest_act
        # min tree level
        if not is_max:
            legal_actions = game_state.get_opponent_legal_actions()
            if not legal_actions:
                return self.evaluation_function(game_state)
            successors = []
            # we are at the max level, so the agent is the opponent
            for act in legal_actions:
                successors.append(game_state.generate_successor(OPPONENT, act))
            minEval = math.inf
            for succ in successors:
                eval = self.get_action_helper(succ, True, depth - 1, alpha, beta)
                minEval = min(minEval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return minEval


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """

        return self.get_action_helper(game_state, True, self.depth * 2)

    def get_action_helper(self, game_state, is_max, depth):
        """
        A helper recursive function that finds the MiniMax action according to the given sub-tree
        with its new depth at each time and a min or max level
        :param game_state: current state
        :param is_max: True iff its a max tree level
        :param depth: the given depth to search
        :return: the MiniMax action
        """
        if depth == 0:
            return self.evaluation_function(game_state)
        # max tree level
        if is_max:
            legal_actions = game_state.get_agent_legal_actions()
            if not legal_actions:
                return self.evaluation_function(game_state)
            successors = []
            # we are at the max level, so we are the agent
            for act in legal_actions:
                successors.append(game_state.generate_successor(OUR_AGENT, act))
            if depth != self.depth * 2:
                return max([self.get_action_helper(succ, False, depth - 1) for succ in successors])
            # we are at the given depth, so need to find the best action according to best max value
            biggest_act = None
            biggest = 0
            for i in range(len(legal_actions)):
                eval = self.get_action_helper(successors[i], not is_max, depth - 1)
                if biggest_act is None or eval > biggest:
                    biggest_act = legal_actions[i]
                    biggest = eval
            return biggest_act
        # min tree level
        if not is_max:
            legal_actions = game_state.get_opponent_legal_actions()
            if not legal_actions:
                return self.evaluation_function(game_state)
            successors = []
            # we are at the max level, so the agent is the opponent
            for act in legal_actions:
                successors.append(game_state.generate_successor(OPPONENT, act))
            value = (sum(self.get_action_helper(succ, True, depth - 1) for succ in successors)) / (len(successors))
            return value


def biggest_tile(current_game_state):
    """
    An evaluation function that finds the biggest tile on the current board state
    :param current_game_state: the given state
    :return: the biggest tile on the current board state
    """
    board = current_game_state.board
    biggest = 0
    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col] > biggest:
                biggest = board[row][col]
    return biggest


def matrix_evaluation_function(current_game_state):
    """
    The steepness score of the board, defined by a weight matrix that gives higher scores
    to boards that focus the weight in one corner.
    :param current_game_state: the given game state
    :return: The steepness score
    """
    board = current_game_state.board
    score = 0
    for i in range(len(board)):
        base_pow = i
        for j in range(len(board[0])):
            score += board[i][j] * (base_pow + j)
    return score


def smoothness_evaluation_function(current_game_state):
    """
    :param current_game_state: a given game state
    :return: The smoothness score of the board, defined as the negative of the sum of differences
    between adjacent tiles on a board. The differences are in base 2 to signify the number of tile
    merges needed for the lower tile to reach the higher tile
    """
    board = current_game_state.board
    sum_of_diff = 0
    # rows differences
    for i in range(len(board)):
        row = []
        for j in range(len(board[0])):
            if board[i][j] != 0:
                row.append(board[i][j])
        for coord in range(len(row) - 1):
            sum_of_diff += abs(math.log2(row[coord]) - math.log2(row[coord + 1]))
    # columns differences
    for j in range(len(board[0])):
        col = []
        for i in range(len(board)):
            if board[i][j] != 0:
                col.append(board[i][j])
        for coord in range(len(col) - 1):
            sum_of_diff += abs(math.log2(col[coord]) - math.log2(col[coord + 1]))
    return -sum_of_diff


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    A weighted sum of 2 heuristic functions. Each function is described in its own
    docstring
    """
    if not current_game_state.get_legal_actions(0):
        return -math.inf
    biggestTile = biggest_tile(current_game_state)
    if biggest_tile(current_game_state) >= WIN_TILE:
        h1 = matrix_evaluation_function(current_game_state)
        h2 = SMOOTH_WEIGHT * smoothness_evaluation_function(current_game_state)
        h3 = TILE_WEIGHT * biggestTile
        return h1 + h2 + h3
    h1 = matrix_evaluation_function(current_game_state)
    h2 = SMOOTH_WEIGHT * smoothness_evaluation_function(current_game_state)
    return h1 + h2


# Abbreviation
better = better_evaluation_function
