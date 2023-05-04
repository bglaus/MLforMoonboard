import numpy as np

import config
from embeddings.route_embeddings import bag_of_holds

# returns a np.array of shape (N_ROWS, N_COLS)
def tf_idf_weights(routes):
    n = len(routes)
    bag_of_holds_routes = bag_of_holds(routes)
    summed = np.sum(bag_of_holds_routes, axis=0)
    summed = np.where(summed != 0, 1 / summed, summed)
    return summed / n

# returns a np.array of shape (N_ROWS, N_COLS)
def tf_idf_weights_old(routes):
    n = len(routes)
    bag_of_holds_routes = bag_of_holds(routes)
    summed = np.sum(bag_of_holds_routes, axis=0)
    return summed / n


def reshape_1d_to_2d(embedded_routes):
    'Reshaps a 1d representation of a route (shape: (n, N_ROWS * N_COLS)) into a 2 dimensional one (shape: (n, N_ROWS, N_COLS))'
    n = len(embedded_routes)
    return embedded_routes.reshape(n, config.N_ROWS, config.N_COLS)


def strings_to_indexes(routes):
    '''
    Given a list of routes, returns the indexes of the holds from that routes
    
        Parameters:
            routes (list) : List of Routes, e.g. [['A4', 'B4', 'C7', 'D9', 'F12', 'D15', 'F18'], ...]
            
        Returns: 
            hold_indexes (list) : Lists of Indexes, e.g. [[33, 34, 68, 91, 126, 157, 192], ...]
    '''
    row_col = lambda hold: (int(hold[1:]) - 1, ord(hold[0]) - 65)
    hold_index = lambda x: config.N_COLS * x[0] + x[1]
    return [[hold_index(row_col(hold)) for hold in route] for route in routes]


def string_to_index(hold_string, n_cols=config.N_COLS, nrows=config.N_ROWS):
    '''
    Returns the index of a climbing hold.
    
        Parameters: 
            hold_string (str) : Representation of a climbing hold, e.g. 'B4'

        Returns:
            hold_index (int) : Index of the hold, value between 0 and n_cols * n_rows  
    '''
    row_col = lambda hold: (int(hold[1:]) - 1, ord(hold[0]) - 65)
    hold_index = lambda x: n_cols * x[0] + x[1]
    return hold_index(row_col(hold_string))