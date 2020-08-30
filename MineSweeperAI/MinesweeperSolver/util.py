from functools import reduce
import operator
def dfs(graph, start, visited, visited_closed):
    """
    a DFS search algorithm adjusted for our game to search for disjoint sets of tiles on the board.
    :param graph: a Dictionary, keys = the open tile the values are next to, values = the closed tiles next
    to the key.
    :param start: the current open tile we are at
    :param visited: the visited open tiles
    :param visited_closed: the visited closed tiles
    :return: tuple of :the visited open tiles, the visited closed tiles(the disjoint set).
    """
    if visited is None:
        visited = set()
    visited.add(start)
    visited_closed = visited_closed.union(graph[start])

    for k, v in graph.items():
        if k in visited:
            continue
        if list(set(visited_closed) & set(v)):
            visited, visited_closed = dfs(graph, k, visited, visited_closed)
    return visited, visited_closed
def ncr(n, r):
    """
    a function calculating the nCr of num and choice according to the nCr algorithm
    :param n: the total number we choose from
    :param r: the choice num
    :return: the nCr
    """
    r = min(r, n - r)
    number = reduce(operator.mul, range(n, n - r, -1), 1)
    denom = reduce(operator.mul, range(1, r + 1), 1)
    return number // denom  # or / in Python 2
