
Comments:
--------------------Evaluation function for ReflexAgent--------------------
At each row and at each column on the given board, this function
calculate differences between all adjacent tiles.
We want the differences between the adjacent tiles to be smaller,
which will allow us to connect with them by a smaller number of moves.
Therefore we will return minus the sum of differences.
And this way a board with the smallest number of differences will be chosen.

-------------------------Better evaluation function-------------------------
Our better evaluation function consists of a sum of 3 functions: 
smoothness_evaluation_function and matrix_evaluation_function.
1. smoothness_evaluation_function:
Smoothness approximates the amount of merges that can be made from a
given board, by counting the differences of log2 of the adjacent tiles 
in each row and each column.
As before, we return minus the sum of differences.
2. matrix_evaluation_function:
This function approximates the monotonicity of the board,
giving the lower right corner the highest weight and decreasing
gradually in direction of the upper left corner.
3. biggest_tile:
This function finds the biggest tile on the board
If this tile is 2048, it means that this successor is a goal,
and at this case we will chose the goal, by giving a high weight to this function.
Otherwise, we are not using this function, because a board with a biggest tile
doesn't guarantee us a good choice.
