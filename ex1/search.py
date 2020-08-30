"""
In search.py, you will implement generic search algorithms
"""

import util


class Node:
    """
    This class represents the structure of a node in a search tree
    """

    def __init__(self, state, action=None, parent=None, cost=0):
        self.state = state
        self.action = action
        self.parent = parent
        self.cost = cost
        self.g = cost
        self.f = 0

    def set_g(self, parent_g):
        """
        Sets the g(n) value of a node
        :param parent_g: the parent`s g(n) value
        """
        self.g = self.cost + parent_g

    def set_f(self, h):
        """
        Sets the f(n) value of a node
        :param h: the heuristic value of the node
        """
        self.f = self.g + h

    def __lt__(self, other):
        """
        Ovverride the "less that" object method
        :param other: other node
        :return: True iff the current node has smaller or equal value than the other
        """
        return self.f < other.f


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Return a list of actions that reaches the goal
    """
    actions = []
    state = problem.get_start_state()  # start state
    visited = set()
    stack = util.Stack()  # all the states to be visited
    i = 0
    stack.push((state, None, i))
    found_goal = False
    while not stack.isEmpty() and not found_goal:
        new_state, last_act, j = stack.pop()
        while actions and actions[-1][1] >= j:
            actions.pop()
        i = j
        if new_state not in visited:
            if last_act is not None:
                actions.append((last_act, i))
            visited.add(new_state)
            lst = problem.get_successors(new_state)
            for succ, act, cost in lst:
                if succ not in visited:
                    stack.push((succ, act, i + 1))
                    if problem.is_goal_state(succ):
                        found_goal = True
                        break
    new_state, last_act, j = stack.pop()
    if last_act is not None:
        actions.append((last_act, i))
    new_actions = [act[0] for act in actions]
    return new_actions


def build_actions(goal):
    """
    Build the path from the root to the node
    :param goal: the goal node
    :return: a list of actions to be done from the root to the node
    """
    actions = []
    nodes = []
    curr_node = goal
    while curr_node:
        nodes.append(curr_node)
        curr_node = curr_node.parent
    for i in range(len(nodes) - 2, -1, -1):
        curr = nodes[i]
        actions.append(curr.action)
    return actions


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    actions = []
    visited = set()
    queue = util.Queue()  # Initialize a queue
    root = Node(problem.get_start_state())
    if problem.is_goal_state(root.state):  # Check if root is a goal
        return actions
    queue.push(root)
    while not queue.isEmpty():  # while queue is not empty
        node = queue.pop()
        while node.state in visited and not queue.isEmpty():  # remove the same states from queue
            # with worse priority
            node = queue.pop()
        visited.add(node.state)
        successors = problem.get_successors(node.state)
        for succ, act, cost in successors:  # Build successors
            if succ not in visited:
                new_node = Node(succ, act, node)
                if problem.is_goal_state(new_node.state):  # if find the goal
                    actions = build_actions(new_node)
                    return actions
                queue.push(new_node)
    return actions


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    return a_star_search(problem)


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    queue = util.PriorityQueue()  # Init the queue
    visited = set()
    actions = []
    root = Node(problem.get_start_state())
    h = heuristic(root.state, problem)
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
                h = heuristic(new_node.state, problem)
                new_node.set_g(node.g)  # Update g(n) value
                new_node.set_f(h)  # Update f(n) value
                queue.push(new_node, new_node.f)
    return actions


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
