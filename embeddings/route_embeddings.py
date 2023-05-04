import numpy as np
import config


# Takes a list of routes. Each route is represented as a list of strings (holds): 
# routes looks like [['F5', 'C10', 'G13', 'D18'], ...]
# outputs a numpy array of size (n, N_ROWS * N_COLS) containing 1d bitmap of route.
# output (embedded route) looks like  [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., ...
def bag_of_holds(routes):
    n = len(routes)
    embedding = np.zeros((n, config.N_ROWS * config.N_COLS))
    for i in range(n):
        for hold in routes[i]:
            col = ord(hold[0].upper()) - 65
            row = int(hold[1:]) - 1
            assert col < config.N_COLS and row < config.N_ROWS
            embedding[i, col + config.N_COLS * row] = 1
    return embedding

# Takes a list of routes. Each route is represented as a list of strings (holds): 
# routes looks like [['F5', 'C10', 'G13', 'D18'], ...]
# outputs a numpy array of size (n, N_ROWS * N_COLS) containing tf_idf weighted bitmap
# output (embedded route) looks like  [0., 0., 0., 0.125, 0., 0., 0., 0., 0., 0.25, 0., ...
def tf_idf_embedding(routes):
    n = len(routes)
    embeddings = np.zeros((n, config.N_ROWS * config.N_COLS))
    
    bag_of_holds_embeddings = bag_of_holds(routes)
    summed = np.sum(bag_of_holds_embeddings, axis=0)

    for i in range(n):
        embeddings[i] = np.multiply(bag_of_holds_embeddings[i], summed)
        embeddings[i] = np.where(embeddings[i] != 0, 1 / embeddings[i], embeddings[i])

    return embeddings


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
    
    row_col = lambda hold: (int(hold[1:]) - 1, ord(hold[0]) - 65)
    hold_index = lambda x: config.N_COLS * x[0] + x[1]
    indexes = [[hold_index(row_col(hold)) for hold in route] for route in routes]

    res = np.zeros((len(indexes), hold_embeddings.shape[1]))
    
    for route_i in range(len(indexes)):
        res[route_i,:] = pool_function([hold_embeddings[i] * weights[i] for i in indexes[route_i]], axis=0)
    return res