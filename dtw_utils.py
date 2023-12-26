import numpy as np

def dtw(x, y, dist_fun):
    n = len(x)
    m = len(y)

    dist_matrix = np.zeros((n, m))

    # calculate distance matrix
    for i in range(n):
        for j in range(m):
            dist_matrix[i, j] = dist_fun(x[i], y[j])

    # initialize cost matrix and traceback matrix with zeros
    cost_matrix = np.zeros((n, m))
    cost_matrix[0, 0] = dist_matrix[0, 0]
    traceback_matrix = np.zeros((n, m))

    for i in range(1, n):
        cost_matrix[i, 0] = dist_matrix[i, 0] + cost_matrix[i-1, 0]

    for j in range(1, m):
        cost_matrix[0, j] = dist_matrix[0, j] + cost_matrix[0, j-1]

    # finding the optimal cost and path using dp
    for i in range(1, n):
        for j in range(1, m):
            penalty = [
                cost_matrix[i-1, j-1], # match
                cost_matrix[i-1, j],   # insertion
                cost_matrix[i, j-1]    # deletion
            ]
            min_penalty_idx = np.argmin(penalty)
            cost_matrix[i, j] = dist_matrix[i, j] + penalty[min_penalty_idx]
            traceback_matrix[i, j] = min_penalty_idx

    # traceback
    i = n-1
    j = m-1
    path = [(i, j)]

    while (i > 0 or j > 0) and np.abs(i)-1 < n and np.abs(j)-1 < m:
        if traceback_matrix[i, j] == 0: # match
            i -= 1
            j -= 1
        elif traceback_matrix[i, j] == 1: # insertion
            i -= 1
        else: # deletion
            j -= 1
        path.append((i, j))
    
    # return dtw normalized cost and path
    return cost_matrix[n-1, m-1] / (n + m), path[::-1]