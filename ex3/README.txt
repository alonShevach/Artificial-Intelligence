
Comments:
---------------------------------Question 9---------------------------------
1. dwr1
The initial state was defined this way:
two robots: r,q are at the room 1 and they are free(do not hold the boxes),
two boxes: a,b are at the room 2
The goal state is:
Move the robots from room 1 to room 2 and
move the boxes from room 2 to room 1, such that the robots are free
2. dwr2
The initial state was defined as before
The goal state is:
Robot r holds the box b, and robot q holds the box b
Since two robots can not hold the same box at the same time,
this instance has no solution
-------------------------------Tower of Hanoi-------------------------------
1. Question 13
We first add all the propositions in this way:
Each one of the pegs can be free(there is no disks on it), so we write: p_i
Each one of the disks can be free(there is no disks on it), so we write: d_i
Each one of the disks can be placed on disk that is bigger or on the peg, so we write:
d_ip_i or d_id_j (i<j)
Then we add all the actions:
Each one of the disks can be moved from the peg or from the other disk,
to another peg or to another disk, so we write:
** Example of moving disk_i from disk_j to another disk_k (i<j, i<k):
In this case, our precondition is that disk_i is free and is placed on disk_j
Also disk_k must be free to move disk_i on it
When this action is occurred, we need to add that disk_j is free now,
and disk_i is on disk_k
Since we moved disk_i, we need to delete some of the propositions:
We delete disk_k, because it is not free anymore,
and we delete d_id_j, because disk_i is not on disk_j anymore
Name: d_i_from_d_j_to_d_k
pre: d_i d_k d_id_j
add: d_j d_id_k
delete: d_k d_id_j
2. Question 14
The initial state was defined this way:
Each one of the pegs is free instead of the peg_0
The disk_n is on the peg_0
The disk_0 is free (there is no disk on it) and it placed on disk_1
For each one of the disk_i: i=1,...,(n-1) it is held that disk_i is on disk(i+1),
so we write: d_id_(i+1)
The goal state is:
Each one of the pegs is free instead of the peg_m
The disk_n is on the peg_m
The disk_0 is free (there is no disk on it) and it placed on disk_1
For each one of the disk_i: i=1,...,(n-1) it is held that disk_i is on disk(i+1),
so we write: d_id_(i+1)
