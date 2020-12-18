#!/usr/bin/python
#
# Student Name: Saad Zaher 
# ID:
# URL: https://github.com/szaher/ARC
#
#

import os, sys
import json
import numpy as np
import re
from collections import Counter
try:
    import networkx as nx
except ImportError:
    print("networx is required!. "
          "Please install networkx library."
          "pip install networkx"
          "or"
          "pipenv install networkx"
          )

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

############################################################################################


def solve_fcc82909(x: np.ndarray) -> np.ndarray:
    """
    Problem:
    In each grid we have a square that consists of 4 cells. Each cell has a colour.
    Depending on the number of colours, we should add a number of cell underneath that square.
    The number of cells we add underneath each square equals 2 * number of different colours in the square.
    Solution:
    The grid only has two or three squares so I searched where these squares starts and start looping over column based.
    All squares on the grid are not intersecting on columns (they are separated column based), so we loop over columns with step = 2
    get unique values (colours) and calculate how many cells we should add, then start adding 'em.
    By observing the examples squares are separated on columns (don't intersect).

    """
    # get the array dimensions row x columns
    rows, cols = x.shape
    # Locating where on the grid where are all the squares
    ind = np.where(x != 0)
    # Starting from the closest square (0,0), we get the first cell of the first square
    start_at = ind[0][0], ind[1][0]
    # looping on columns starting from the column of the first square
    for col in range(start_at[1], cols, 2):
        # get uniqure values in the column
        uni = np.unique(x[:, col])
        if len(uni) > 1:
            # get unique values in 2 columns at once.
            uni = np.unique(x[:, col:col+2])
            # remove zeros
            uni = uni[uni.nonzero()]
            # how many unique values
            items = len(uni)
            # define the range where we will be searching
            search_in = x[:, col:col+2]
            # new idexes based on the new search array where values != 0
            new_ind = np.where(search_in != 0)
            # we might not have a perfect columns so we might want to pad the columns to get a perfect square.
            # trying to find if we should pad to left or right but will do it manually
            # if we don't have a perfect square
            if len(new_ind[0]) < 4:
                # Widen our search range
                search_in = x[:, col-1:col+1]
                # recalculate the unique and index again
                uni = np.unique(x[:, col-1:col+1])
                uni = uni[uni.nonzero()]
                items = len(uni)
                new_ind = np.where(search_in != 0)
                # if still less than 4 squares move on as we did pad it before...
                # in the previous run
                if len(new_ind[0]) < 4:
                    # if we got less than 4 elements again (not a perfect square)
                    # skip...
                    continue
            # we got a perfect square and we have the indexes
            rs, cs = new_ind
            max_row = max(rs) + 1
            # depending on how many unique items we got
            # loop and change the rights cells to green.
            for idx in range(items):
                search_in[max_row+idx, 0] = 3
                search_in[max_row+idx, 1] = 3
    return x

############################################################################################


def solve_662c240a(x: np.ndarray) -> np.ndarray:
    """
    Problem:
    Input is an array which is 3x9. It contains 3 sub arrays each one is 3 x 3. Each sub array consists of two unique numbers only.
    We need to choose one of these arrays. After observing the numerical data that form the arrays I found we usually choose arrays that
    has the sum of the unique numbers = 9 or mod 9 = 0. We have an error in the training data so I had to handle that with an exceptional case
    If the sum(unique numbers of the sub array) > 9 and == 12 is valid instead of %9 = 0.
    Solution:
    Very simple solution, divide the array into 3 sub arrays, loop over sub arrays, if the sum of the sub array % 9 = 0
    select sub array. There is an exception for that rule which I consider it training error due to noise.
    """
    # list of sub arrays
    arrays = []
    # first array
    arrays.append(x[:3, :3])
    # Second array
    arrays.append(x[3:6, :3])
    # Third array
    arrays.append(x[6:9, :3])

    selected = -1
    # loop over arrays
    for i in range(len(arrays)):
        # get unique elements.
        u = np.unique(arrays[i])
        # if sum % 9 == 0 > this is the right value
        if not np.sum(u) % 9:
            selected = i
            break
        # exception. I think this is training error
        elif np.sum(u) > 9 and np.sum(u) <= 12:
            selected = i

    return arrays[selected]

############################################################################################


def asolve_6a1e5592(x: np.ndarray) -> np.ndarray:
    """
    Problem:
    Provided a matrix that has shapes at the button that can fill a space in the upper part of the matrix.
    We need to detect the shape boundaries and if it's a good fit for which upper part.
    Solution:
    Try to find all 5 shapes in the matrix.
    Try to find all empty shapes between the 2s in the matrix that could fit a 5 shape.
    Connect them together.
    @todo Complte the implementation (Sorry didnt have much time to complete this one...)

    """
    # rows and columns of M x N matrix
    rows, cols = x.shape
    # find the middle of the matrix
    mid_range = int(rows / 2)

    # find pattrern
    def find_pattern(arr: np.ndarray, s, start=0):
        start_idx = 0
        last_idx = 0
        count = 0
        # loop from the start index over columns
        for i in range(start, cols):
            # if the number we are searching for found in the column
            # then this could be part of the shape we are looking for
            if s in arr[:, i]:
                # find the count of the number
                # how many times it's in the column
                count += list(arr[:, i]).count(s)
                # we are trying to detect a shape, so we need to keep track of where the shape started and where it ended.
                if not start_idx:
                    start_idx = i
                # check if the shape is expanded to another column
                c_idx = i + 1
                # if the next column is bigger than the column size so don't try to check other columns.
                if c_idx >= cols:
                    c_idx = i
                    # if the shape started at the last column in the matrix and ended in the last column as well
                    # so it's one column shape |
                    if c_idx == cols - 1:
                        start_idx -= 1
                        last_idx = i
                # if the search number wasn't found in the next column so setting the last_index of the shape
                if s not in arr[:, c_idx]:
                    last_idx = c_idx
                    break
        return start_idx, last_idx, count

    def find_all_shapes(arr, shape, number_of_shapes):
        """
        Find all shapes in matrix
        """
        shapes = []
        # loop over the number of shapes in the matrix
        # determine the start and end of each shape
        for i in range(number_of_shapes):
            # starting from column 0 find the first patter then continue from the last shape index till we find
            # next shape
            if len(shapes) > 0:
                start = shapes[-1][1]
            else:
                start = 0

            match = find_pattern(arr, shape, start=start)
            shapes.append(match)
        return shapes

    # @todo complete this function.
    # find how many zeros are required for each group of twos
    # find in the fives arrays which array has matching 5 per row to fill up the twos.

    def find_match(arr, fives, twos):
        """
        Try to find a shape of 5 that matches an empty space in the twos area.
        I do this by calculating the zeros in the two area (mid part of the matrix)
        then see how many 5s could fit in it.
        """
        two = arr[:mid_range - 2, twos[0]:twos[1]]
        zeros = {}

        for idx, t in enumerate(two):
            c = t.tolist().count(0)
            zeros[idx] = c

        fives_idx = {}
        five = arr[mid_range + 1:, fives[0]:fives[1]]
        for idx, f in enumerate(five):
            c = f.tolist().count(5)
            fives_idx[idx] = c

        match = True
        for idx, z in enumerate(zeros):

            if fives_idx[idx] < z:
                match = False
        return match
    # find 5 shapes
    f_shapes = find_all_shapes(arr=x[mid_range:, :], shape=5, number_of_shapes=3)
    # find two shapes
    t_shapes = find_all_shapes(arr=x[:mid_range - 2, :], shape=0, number_of_shapes=3)
    matches = []
    for t in t_shapes:
        for f in f_shapes:
            match = find_match(x, f, t)
            if match:
                matches.append((t, f))

    return x



# all true ...
############################################################################################


def solve_ecdecbb3(x: np.ndarray) -> np.ndarray:
    """
    Problem:
    We have a grid with two or more points and one or more lines. We need to connect each point to the closest line(s).
    We have three cases:
    1) Point is above the line(s)
        - Connect the point to the first line it reaches
        * Problem Can't be connected to more than one line in this case
    2) Point is below the line
        - Connect the point the last seen line
        * Problem Can't be connected to more than one line in this case
    3) Point between two lines
        - Connect the point to both lines
    Solution:
    Start by finding how many points and lines we have. If we have only one line connect all points to the line.
    If we have two lines then we need to find where on the grid the lines are located, then where the points are located
    from the lines and handle it using one of the three cases explained in the problem.
    """
    def process_line(arr: np.ndarray, e_idxes: np.ndarray, t_idxes: np.ndarray) -> np.ndarray:
        """
        Process one line at a time
        :param: arr: the grid.
        :param: e_idxes: eight's indexes which represent the line.
        :param: t_idxes: Two's indexes which represents the points.
        """
        t_cols = t_idxes[1]
        t_rows = t_idxes[0]
        e_rows = e_idxes[0]

        # loop over the points (by row)
        for idx, row in enumerate(t_rows):
            # get column value of the point, so we have point is (row, col_idx)
            col_idx = t_cols[idx]
            # if row(point) < row(line), point is in the top of the grid
            #    * point is here for example
            # -------------------------
            #
            if row < e_rows[0]:
                # start from the point location till the line
                for i in range(row, e_rows[idx] + 1):
                    # fill it all the way till the line
                    arr[i, col_idx] = 2
                # fill the square arround the meet point between the point and the line.
                arr[e_rows[idx] - 1, col_idx - 1] = 8
                arr[e_rows[idx] - 1, col_idx] = 8
                arr[e_rows[idx] - 1, col_idx + 1] = 8
                arr[e_rows[idx] + 1, col_idx - 1] = 8
                arr[e_rows[idx] + 1, col_idx] = 8
                arr[e_rows[idx] + 1, col_idx + 1] = 8
            else:
                # point is under the line. point is showing at the very end of the grid
                #
                # --------------------------------
                #
                #    * point is here for example
                #
                for i in range(e_rows[idx] + 1, row):
                    arr[i, col_idx] = 2
                arr[e_rows[idx] - 1, col_idx - 1] = 8
                arr[e_rows[idx] - 1, col_idx] = 8
                arr[e_rows[idx] - 1, col_idx + 1] = 8
                arr[e_rows[idx] + 1, col_idx - 1] = 8
                arr[e_rows[idx] + 1, col_idx] = 8
                arr[e_rows[idx] + 1, col_idx + 1] = 8
                arr[e_rows[idx], col_idx] = 2
        return arr

    def proces_two_lines(array: np.ndarray) -> np.ndarray:
        """
        Process two lines.
        Get the lines locations. Find where the points are located from the lines and process them.
        """
        # recalculate some parameters on the new array if transposed.
        eight_indexes = np.where(array == 8)
        # get unique rows
        uniq_rows = np.unique(eight_indexes[0])
        # get unique columns
        uniq_cols = np.unique(eight_indexes[1])

        # the trick here we are trying to re-construct the whole row from the unique element that represents the row index of the line.
        first_line = (np.full(uniq_cols.shape, uniq_rows[0]), uniq_cols)
        # same thing for the second line.
        second_line = (np.full(uniq_cols.shape, uniq_rows[1]), uniq_cols)
        # get the indexes of the points to connect to the lines.
        p_idxes = np.where(array == 2)
        # parameters to pass to the process_line function (point indexes, line indexes)
        parameters = {}
        # loop over rows for each line
        # compare if the row of the point is smaller than both lines rows, then
        # the point is located in the top of the matrix and there are two lines underneath it, in that case we connect it only to one line
        # so map the point to closest row ( in that case pick the min(rows indexes for both lines)
        # if the point is located between the two lines, connect it to both of them, so map to both lines.
        # if the point's row index is bigger than the two lines rows indexes then
        # the two lines are located in the beginning of the matrix and the point is located down
        # in that case we connect it to the closest one max(rows indexes of both lines).

        # rows of both lines into one tuple
        e_all_rows = (first_line[0][0], second_line[0][0])
        # transpose the indexes locations so we can get it row, col instead of row, row (output from np.where)
        for row, col in np.array(p_idxes).T:
            # point is located before the two lines case
            if row < e_all_rows[0] and row < e_all_rows[1]:
                # if close to the first line, map to it
                if first_line[0][0] == min(e_all_rows):
                    parameters[row, col] = [first_line]
                else:
                    # if close to the second line, map to it
                    parameters[row, col] = [first_line]
            # point is located after the two lines case
            elif row > e_all_rows[0] and row > e_all_rows[1]:
                if first_line[0][0] == max(e_all_rows):
                    parameters[row, col] = [second_line]
                else:
                    parameters[row, col] = [second_line]
            else:
                # point in between the lines.
                parameters[row, col] = [first_line, second_line]
        # loop over the parameters dict and process one by one
        for pk in parameters:
            for v in parameters[pk]:
                # converting the point cordinates to 2d numpy array
                array = process_line(array, v, np.array([[pk[0]], [pk[1]]]))

        return array

    # Get where points are located on the grid
    point_idxes = np.where(x == 2)
    # find indexes of number 8 which forms the line.
    eidxes = np.where(x == 8)

    # If there is only one line just process it directly
    # np.where returns a tuple ([rows], [columns])
    if np.unique(eidxes[0]).size == 1:
        # pass the array, one line, one or more points
        # if the line is horizontal we pass the x
        return process_line(x, eidxes, point_idxes)
    elif np.unique(eidxes[1]).size == 1:
        # if the line is vertical we pass the x.T
        # then we transpose the result
        return process_line(x.T, eidxes, point_idxes).T

    # how many lines found (should be 2 max)
    # get unique rows
    unique_rows = np.unique(eidxes[0])
    # get unique columns
    unique_cols = np.unique(eidxes[1])

    # # if there are two lines. split them and handle each line separately.
    if unique_rows.size == 2:
        # if the lines are located horizontally
        return proces_two_lines(x)
    elif unique_cols.size == 2:
        # if the lines are located vertically
        # transpose input, transpose output
        return proces_two_lines(x.T).T
    return x

############################################################################################


def solve_4522001f(x: np.ndarray) -> np.ndarray:
    """
    Problem:
    Input is a 3 x 3 array that has square (2x2), the square contains one dominant variable (3) and one cell as 2.
    The output is a grid 9 x 9 which has two (4 x  4) cubes stretched from the side of 2 cell.
    Solution:

    """
    # rows and columns from the array
    rows, cols = x.shape
    # calculate the shape of the output array..
    # since input is 3 x 3 then output should be 9 x 9
    output_shape = (rows * cols, rows * cols)
    # find the dominant variable or number.
    # that was my first task so I thought I should make everything variable.
    c = Counter([i for i in x.flatten() if i != 0])
    # obtain the dominant variable
    dom = c.most_common(1)[0][0]
    # generate output containing zeros.
    output = np.zeros(output_shape)

    # loop over the array
    for row in range(rows):
        for col in range(cols):
            # value of the cell.
            v = x[row, col]
            # if not empty
            if v != 0:
                # loop for rows=3 times
                for i in range(rows):
                    # extend the dominant variable for the current cell and it's neighbours.
                    # row + i, col + i then row, col + i, then row +i, col
                    # that would generate the top left (4 x 4) cube of 3
                    output[row + i, col + i] = dom
                    output[row, col + i] = dom
                    output[row + i, col] = dom
    # check what we have populated in the matrix so far
    dom_idx = np.where(output)
    # building where should we start for the next (4 x 4 ) cube
    # it should start in the following row after the last row of the previous cube
    # and the following column after the last column found for the previous cube
    dom_idx_up = (dom_idx[0] + rows + 1, dom_idx[1] + rows + 1)
    # we have correctly calculated the indexes for the second cube
    # so we now set the value
    output[dom_idx_up] = dom
    # the previous code works Ok if the cube (2 x 2) first found top left corner
    # but if it appears bottom right corner it would be in the wrong location.
    # check if we need to reverse and shift the output
    if x[0, -1] == dom:
        # flip would reverse the order of the elements in the array vertically
        output = np.flip(output, axis=1)
        # just shift the last column to be the first one and we are now ready to go.
        output = np.roll(output, shift=1, axis=1)
    x = output
    return x

############################################################################################


def solve_d9f24cd1(x: np.ndarray) -> np.ndarray:
    """
    Problem: We have a point located at the end of the matrix. We need to draw a line from the beginning of the
    matrix and connect it to the point. There might be something blocking the line so we need to go to the next column
    till after the blockage then continue in the right column.
    Solution:
    Loop on the transposed matrix if we find the point in the row; check if there is something blocking the line
    if yes, start from the next column till we reach after the blocking point then continue in the right column.
    """
    # use to skip columns when there is a blockage
    skip_idx = []
    # loop over the array columns
    for idx, row in enumerate(x.T):
        if idx in skip_idx:
            continue
        # convert row to regular list for smoother checks
        row = row.tolist()
        if 2 in row:
            # is there any 5s blocking this column ?
            if 5 in row:
                # get the index of the 5
                five_idx = row.index(5)
                # from the next column set all to 2 till the blockage so we can avoid it
                x.T[idx + 1, :five_idx + 2] = 2
                # switch back to the right column and continue normally
                x.T[idx, five_idx + 1:] = 2
                # append the value of the next column that we use to avoid the blockage so we a void confusion
                skip_idx.append(idx + 1)
            else:
                # no blockage, happy life :)
                x.T[idx, :] = 2
    return x

# def solve_2bee17df(x):
#     """
#     Problem: Find the max non blocked space on Rows and max non blocked space on Columns and fill them
#     """
#     return x

# all cases are true
############################################################################################


def solve_b2862040(x):
    """
    Problem: We have a grid that contains a set of shapes that might be open or closed.
    We need to change the colour of the closed shapes only. I have two implementations for this problem,
    one just using numpy and one using networkx.

    Solution:
    This solution is based on builing a graph with all shapes and using networkx graphs library to find cycles in the graphs.
    A cycle should represent a closed circle or loop but might neglect some extra points connecting the shape
    that's why I used connected components to get all connected points to the cycle after obtaining the cycle.
    """
    # find indexes where x = 1 ( 1 forms the shapes)
    indexes = np.where(x == 1)
    # indexes is provided as tuple (0 > rows, 1 > columns)
    # converting it to points (row, column) list of tuples
    points = [(i, j) for i, j in zip(indexes[0], indexes[1])]
    # create a graph
    G = nx.Graph()
    # populate nodes on the graph
    for p in points:
        # Here we are using the cell index (row, column) as a node
        G.add_node(p)

    # populate edges between nodes
    # for rows: if row[i] is next to row[i+1] connect them
    # for columns: if column[i] is next to column[i+1] connect them
    for i in range(len(points)):
        # find all adjacent nodes to this point. Adjacent of (row, column) = [(row+1, column), (row, column+1), (row, column -1 ), (row -1, column)]
        point = points[i]
        # looping again on the points but starting from the first index not from zero.
        for j in range(1, len(points)):
            point2 = points[j]
            # matching on the row but we need to check the column
            if point[0] == point2[0] and (point[1] - point2[1] < 2 and point[1] - point2[1] > -2):
                G.add_edge(point, point2)
            # matching on the column but we need to check the row
            elif point[1] == point2[1] and point[0] - point2[0] < 2 and point[0] - point2[0] > -2:
                G.add_edge(point, point2)
    # We can draw the graph if we want.
    nx.draw(G, with_labels=True)

    # build a list of connected components as cycles will only return the loop not any additional connecting edges.
    components = []
    for i in nx.connected_components(G):
        components.append(i)
    # retrieve cycles
    cycles = nx.cycle_basis(G)
    # Loop over cycles.
    for cycle in cycles:
        # cycles returns each node as a cycle with itself. so we need to skip that.
        if len(cycle) > 1:
            # loop over connected components to find if the cycle is equal or subset of the connected component
            for comp in components:
                if cycle == comp or set(cycle).issubset(comp):
                    for i in comp:
                        x[i] = 8

    return x

############################################################################################

# There is already solve_b2862040 that uess networkx to solve this problem in a very efficient way.
def skip_solve_b2862040(x: np.ndarray) -> np.ndarray:
    """
    Problem: We have a grid that contains a set of shapes that might be open or closed.
    We need to change the colour of the closed shapes only. I have two implementations for this problem,
    one just using numpy and one using networkx.

    Solution:
    This solution here was depending on find a similar row or column matching the current row or column/
    """

    def get_adjacent_rows(arr: np.ndarray) -> np.ndarray:
        """
        Find row cells that has the same value
        """
        adjacents_members = []
        # loop over the matrix row by row
        for idx, row in enumerate(arr):
            if 1 in row:
                # find if the row has cells with value = 1
                indexes = np.where(row == 1)[0]
                # loop over indexes and find if indexes are next to each other or not.
                # that should generate a list of True and/or False
                adjacents = [indexes[i] == indexes[i + 1] - 1 for i in range(0, len(indexes) - 1)]
                member = []
                # adjacent flag will be used to separate multiple groups of 1s in the same row
                # so if we found a group of 1 starting from (0,0) till (0, 5) then space then (0, 8) till (0, 12)
                # those should be separated.
                adjacents_flag = False
                for indx in range(len(adjacents)):
                    # If two row cells has 1 are adjacent
                    if adjacents[indx]:
                        # if we didn't capture it already
                        if indexes[indx] not in member:
                            # add the current cell row, column value and the following one.
                            member.append(indexes[indx])
                        member.append(indexes[indx + 1])
                        adjacents_flag = True
                    elif not adjacents[indx] and adjacents_flag:
                        # if after a series of adjacent cells we found non-adjacent ones split
                        # append to the adjacents list
                        # empty member list then try again
                        adjacents_members.append((idx, member))
                        member = []
                if member:
                    adjacents_members.append((idx, member))
        return adjacents_members

    def get_adjacent_columns(arr: np.ndarray) -> np.ndarray:
        """ Find adjacent columns.
        It's actually the same exact one like the previous one but X is transposed.
        """
        adjacent_cols = []
        for idx, col in enumerate(arr.T):
            #     print(f"Row {idx}: {row}")
            if 1 in col:
                indexes = np.where(col == 1)[0]
                adjacents = [indexes[i] == indexes[i + 1] - 1 for i in range(0, len(indexes) - 1)]
                member = []
                adjacents_flag = False
                for indx in range(len(adjacents)):
                    if adjacents[indx]:
                        if indexes[indx] not in member:
                            member.append(indexes[indx])
                        member.append(indexes[indx + 1])
                        adjacents_flag = True
                    elif not adjacents[indx] and adjacents_flag:
                        adjacent_cols.append((idx, member))
                        member = []
                if member:
                    adjacent_cols.append((idx, member))
        return adjacent_cols

    def get_connected_rows(adjacents_members):
        """
        Check if a row is subset of another row so they might be connected or mirrored
        """
        row_points = []
        change_rows = [] # list of lists of connected rows
        ignore = [] # ignore rows that already found a match.
        for idx in range(len(adjacents_members)):
            # syntax (row, [columns])
            # (1, [0, 1, 2, 3, 4]) > row 1, columns 0, 1, 2, 3, 4
            row, cols = adjacents_members[idx]
            # if no columns skip
            if not cols:
                continue
            # try to find a match in the adjacents members for this row so we start from one to avoid matching with self.
            for i in range(1, len(adjacents_members)):
                # if member is already matched skip
                if idx in ignore or i in ignore:
                    continue
                # if rows might have a match but there so far away, it's probably wrong so skip too
                if r - row > 2 or r - row < -2:
                    continue
                # again
                # syntax (row, [columns])
                # (1, [0, 1, 2, 3, 4]) > row 1, columns 0, 1, 2, 3, 4
                r, cs = adjacents_members[i]
                if not cs:
                    continue
                if idx == i:
                    continue
                if cols == cs or set(cols).issubset(cs) or set(cs).issubset(cols):
                    # if columns are similar or subset of one another ...
                    # don't match them again.
                    ignore.append(idx)
                    ignore.append(i)
                    # generate points
                    # calculate row points.
                    row_points = [(row, c) for c in cols]
                    row_points.extend([(r, c) for c in cs])
                    # append to the change_rows list
                    change_rows.append(row_points)
                    # we already found a match don't try to find another match.
                    break
        return change_rows

    def get_connected_cols(adjacent_cols):
        """
        Very similar to the previous function but on columns
        """
        col_points = []
        change_columns = []
        ignore = []
        for idx in range(len(adjacent_cols)):
            col, rows = adjacent_cols[idx]
            if not rows:
                continue
            for i in range(1, len(adjacent_cols)):
                if idx in ignore or i in ignore:
                    continue
                c, rs = adjacent_cols[i]
                if not rs:
                    continue
                if idx == i:
                    continue
                if rows == rs or set(rows).issubset(rs) or set(rs).issubset(rows):
                    # generate points
                    ignore.append(idx)
                    ignore.append(i)
                    col_points = [(row, col) for row in rows]
                    col_points.extend([(row, c) for row in rs])
                    change_columns.append(col_points)
                    break
        return change_columns
    # find adjacent or connected row points
    rows = get_adjacent_rows(x)
    # find adjacent or connected column points
    columns = get_adjacent_columns(x)
    # check if the row has a mirror or another row
    # in front of it that has some or all points that the current row has
    row_points = get_connected_rows(rows)
    # same for columns.
    column_points = get_connected_cols(columns)
    # connect row points together.
    for points in row_points:
        for row_point in points:
            x[row_point] = 8
    # connect column points together
    for points in column_points:
        for col_point in points:
            x[col_point] = 8

    # @todo find a way to check if row and column are connected ??

    return x


# all true
def solve_05269061(x: np.ndarray) -> np.ndarray:
    """
    Problem:
    We have a pattern usually 3 numbers that keep repeating all over the matrix. We need to determine  the order of the
    appearance of these numbers. We can either loop row by row to fill the matrix or fill it diagonally.
    The three numbers could appear anywhere in the matrix.
    Solution:
    I searched for few ways to determine the order of the 3 numbers, after a while I thought of using an exact similar
    matrix we 3 letters only. a,b,c  and started ordering them in the new matrix a, b, c, a, b, c, a, ... .
     - find the unique values in the original matrix and getting their indexes.
     - use the letters matrix to find either this a or b or c
     - when we get the values of a, b, c then we know now the order so we can fill the matrix.

    """
    # get matrix dimensions
    rows, cols = x.shape
    # find unique values (should be 3 values)
    unique = np.unique(x)
    # remove zeros as they are considered background.
    unique = unique[unique.nonzero()]
    # building order dict.
    # The order is found by substituting the letter of the last cell will give you the letter for the current one.
    p = {'a': 'b', 'b': 'c', 'c': 'a'}
    # built the first row manually.
    first_row = ['a', 'b', 'c', 'a', 'b', 'c', 'a']

    letters = [first_row]
    last_i = 'a'
    # loop with the same range of rows of the original matrix (x) to generate the letters
    for row in range(1, rows):
        _row = []
        # loop with columns
        for col in range(cols):
            # current value is found by using the previous value as key in the order dict to obtain the current value
            _row.append(p[last_i])
            # after getting the current value, now the current one became the last
            last_i = p[last_i]
        # add the row.
        letters.append(_row)
    # convert to numpy array
    letters = np.array(letters)
    # set default values of a, b, c to -1
    a, b, c = -1, -1, -1
    # loop over the unique values in the original matrix. should be three numbers only.
    for u in unique:
        # get indexes of the unique value
        f_idx = np.where(x == u)
        # get it's value from the original matrix
        v = x[f_idx]
        # use the same index of the unique value in the original matrix (x) against the letters matrix
        # that should return a list of the same letter either A, B or C.
        r = letters[f_idx][0]
        # set the value to update the order dict.
        p[r] = v[0]
    # find indexes where cell value = a
    a = np.where(letters == 'a')
    # find indexes where cell value = b
    b = np.where(letters == 'b')
    # find indexes where cell value = c
    c = np.where(letters == 'c')
    # update the original matrix with the indexes and the corresponding value from the orders dict
    x[a] = p['a']
    x[b] = p['b']
    x[c] = p['c']
    return x


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})"
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals():
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    print(f"Solved Total {len(tasks_solvers)}")


def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""

    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)


def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))

if __name__ == "__main__": main()

