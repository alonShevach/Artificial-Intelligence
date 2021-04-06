Comments:
----------------------------------Additions in search.py----------------------------------
1. The Node Class
This class represents the structure of a node in a search tree.
Each time we extend the successors of any state, we create new nodes with a same parent-the current state.
In this way we can build the search tree and find the path from the root to a goal quickly.
2. build_actions Function
This function gets the goal that we found and builds the actions we need to make to get
from the root to the goal.

-----------------Blokus Cover Problem and Blokus Corners Problem heuristic-----------------
For this two problems we used two same heuristic function, because this two problems
have the the same purpose: to cover the given targets in the most effective way.
The only difference between them is the targets to cover.
To deal with this difference, we initialize the dataMember in each problem class - targets.
In Cover Problem the targets are the given targets,
in Corners Problem the targets are the corners of the board.
Our heuristic function returns the maximum value of 2 heuristic admissible functions:
1. discrete_space
This function calculates the value by the amount of free targets left.
In the same way, that eight-puzzle heuristic function
(that calculate the number of puzzles that are not on their places),
we can conclude that it is admissible
2. heuristic_smallest_dist
This function calculates the smallest distance between a tile on a board and the target as follows:
First, for each corner we found the nearest tile that we can move from it to the current target.
Than we calculate the smallest distance from this tile to the target.
This calculation we made by using chebyshev_distance function.
This distance is calculated by maximum value between the x coordinates distance and y coordinates.
(In our code we added 1 to coordinate distance, because the start point on the board is (0,0))
This function returns the smallest distance between tile and a target on a board, on which we can move diagonally.
After calculating the sum of all smallest distances, we delete the common distances :
If there are tile "a" and tile "b" that has the same smallest distance to the same target,
but tile "a" has more nearest targets to cover than tile "b", we will prefer tile "a",
and remove all the common distances between tile "a" to its targets (if there are any).

For the Blokus Corners Problem we use one more heuristic function : corner_heuristics
We first check,if there is any piece that can cover two targets on the same edge
If so,we return the sum of tiles of the minimal piece
Otherwise, we return the sum of tiles of the two minimal pieces

-------------------------------------Suboptimal search-------------------------------------
We implemented the function solve as follows:
In each time we try to find the best successor for the current state.
The successor should cover the nearest target or become closer to the nearest target.
By choosing the best successor we also check if this successor does not block other goals.
After trying cover the targets in this way, it is possible that in extreme cases we will not find the solution.
In this case we will use the A* search algorithm with the heuristic function,
that does not consider the path cost, and only calculates the distance.

-------------------------------------MiniContest search-------------------------------------
The implementation is same as the solve method in Suboptimal search


Hope you enjoy our implementation =)
