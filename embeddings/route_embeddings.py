import numpy as np

N_ROWS, N_COLS = 18, 11

# Takes a list of routes. Each route is represented as a list of strings (holds): 
# routes looks like [['F5', 'C10', 'G13', 'D18'], ...]
# returns a numpy array of size (n, N_ROWS, N_COLS)
def bag_of_holds_1d(routes):
    n = len(routes)
    embedding = np.zeros((n, N_ROWS * N_COLS))
    for i in range(n):
        for hold in routes[i]:
            col = ord(hold[0]) - 65
            row = int(hold[1:]) - 1
            assert col < N_COLS and row < N_ROWS
            embedding[i, col + N_COLS * row] = 1
    return embedding

# Takes a list of routes. Each route is represented as a list of strings (holds): 
# routes looks like [['F5', 'C10', 'G13', 'D18'], ...]
# returns a numpy array of size (n, N_ROWS, N_COLS)
def bag_of_holds_2d(routes):
    n = len(routes)
    embedding = np.zeros((n, N_ROWS, N_COLS))
    for i in range(n):
        for hold in routes[i]:
            col = ord(hold[0]) - 65
            row = int(hold[1:]) - 1
            assert col < N_COLS and row < N_ROWS
            embedding[i, row, col] = 1
    return embedding

# returns a np.array of shape (N_ROWS, N_COLS)
def tf_idf_weights(routes):
    n = len(routes)
    bag_of_holds = bag_of_holds_1d(routes)
    summed = np.sum(bag_of_holds, axis=0)
    return summed / n

def tf_idf_embedding_1d(routes):
    n = len(routes)
    embeddings = np.zeros((n, N_ROWS * N_COLS))
    
    bag_of_holds = bag_of_holds_1d(routes)
    summed = np.sum(bag_of_holds, axis=0)

    for i in range(n):
        embeddings[i] = np.multiply(bag_of_holds[i], summed)
        embeddings[i] = np.where(embeddings[i] != 0, 1 / embeddings[i], embeddings[i])

    return embeddings



def tf_idf_embedding_2d(routes):
    n = len(routes)
    embeddings = np.zeros((n, N_ROWS, N_COLS))
    
    bag_of_holds = bag_of_holds_2d(routes)
    summed = np.sum(bag_of_holds, axis=0)

    for i in range(n):
        embeddings[i] = np.multiply(bag_of_holds[i], summed) 
        embeddings[i] = np.where(embeddings[i] != 0, 1/embeddings[i], embeddings[i])

    return embeddings


# Given a List of Routes: [['A4', 'B4', 'C7', 'D9', 'F12', 'D15', 'F18'], ...]
# Returns a List of indexes (of holds) for each route: [[33, 34, 68, 91, 126, 157, 192], ...]
def strings_to_indexes(routes, n_cols=N_COLS, nrows=N_ROWS):
    row_col = lambda hold: (int(hold[1:]) - 1, ord(hold[0]) - 65)
    hold_index = lambda x: n_cols * x[0] + x[1]
    return [[hold_index(row_col(hold)) for hold in route] for route in routes]


def string_to_index(hold_string, n_cols=N_COLS, nrows=N_ROWS):
    row_col = lambda hold: (int(hold[1:]) - 1, ord(hold[0]) - 65)
    hold_index = lambda x: n_cols * x[0] + x[1]
    return hold_index(row_col(hold_string))


# expects a List of Routes [['A4', 'B4', 'C7', 'D9', 'F12', 'D15', 'F18'], ...]
# and a matrix with embedding vectors for each hold (use get_hold_embedding function from Hold2Vec)
# return a numpy array of size (num_routes, size_hold_embeddings)
def pooled_embedding(routes, hold_embeddings, pool_method='sum', weights=None):
    if pool_method == 'sum':
        pool_function = np.sum
    elif pool_method == 'avg' or pool_method == 'average':
        pool_function = np.average
    elif pool_method == 'max':
        pool_function = np.max
    elif pool_method == 'min':
        pool_function = np.min

    if weights is None:
        weights = np.ones(hold_embeddings.shape[0])
    
    indexes = strings_to_indexes(routes)
    res = np.zeros((len(indexes), hold_embeddings.shape[1]))
    
    for route_i in range(len(indexes)):
        res[route_i,:] = pool_function([hold_embeddings[i] * weights[i] for i in indexes[route_i]], axis=0)
    return res