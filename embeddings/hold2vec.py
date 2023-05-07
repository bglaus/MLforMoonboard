import numpy as np
from tensorflow import keras


def load_hold_matrix(filename='./data/embeddings/hold2vec_embeddings.npy'):
        with open(filename, 'rb') as f:
            hold_m = np.load(f)
        return hold_m

class hold2vec():
    def __init__(self):
        self.neuralnet = None
        self.n_rows = 18
        self.n_cols = 11
        self.n_holds = 18 * 11
        self.N = 40
        self.window_size = 8
        self.epochs = 400
        self.X_train = []
        self.y_train = []
        self.earlystop_patience = 2
        pass


    def initialize(self, n_rows=18, n_cols=11, N=40, window_size=8, earlystop_patience=2):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_holds = self.n_rows * self.n_cols
        self.N = N
        self.window_size = window_size
        self.earlystop_patience = earlystop_patience
        self.build_model()
        return

    # BUILDING THE MODEL:
    def build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(self.n_holds, input_shape=(self.n_holds,)))
        model.add(keras.layers.Dense(self.N))
        model.add(keras.layers.Dense(self.n_holds, activation='softmax'))
        self.neuralnet = model
        return


    def train(self, epochs=0, batch_size=32):
        self.neuralnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        if epochs > 0:
            self.epochs = epochs
        assert len(self.X_train) == len(self.y_train)
        assert len(self.X_train) > 0

        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=self.earlystop_patience)
        self.neuralnet.fit(
            self.X_train, 
            self.y_train, 
            epochs=self.epochs, 
            batch_size=batch_size,
            callbacks=[callback],
        )
        return

    
    
    def get_hold_matrix(self):
        'returns a matrix of shape ()'
        return self.neuralnet.layers[2].get_weights()[0]

    def get_hold_embedding(self, hold_i):
        return self.neuralnet.layers[2].get_weights()[0][hold_i,:]


    def save_hold_matrix(self, filename='./data/embeddings/hold2vec_embeddings.npy'):
        'Save the matrix of hold embeddings to a npy file (at filename).'
        hold_m = self.get_hold_matrix()
        with open(filename, 'wb') as f:
            np.save(f, hold_m)

    def load_hold_matrix(self, filename='./data/embeddings/hold2vec_embeddings.npy'):
        'Load the hold matrix from a npy file (at filename).'
        with open(filename, 'rb') as f:
            hold_m = np.load(f)
        return hold_m

    # BUILDING THE TRAINING DATA:

    def create_one_hot_hold_vector(self, i):
        'Given the index of a hold, returns a 1-dimensional one-hot vector.'
        one_hot = np.zeros(18*11)
        one_hot[i] = 1
        return one_hot

    def create_context_bitmap(self, i):
        '''
        Given the index of a hold, returns a 1-dimensional bitmap of all holds within reach
        "within reach" is a 2 dimensional concept: it's a circle around the index hold with a radius of "window_size"
        '''
        row = i // self.n_rows
        col = i % self.n_rows
        context_bitmap = np.zeros(self.n_rows * self.n_cols)
        
        for r in range(max(0, row-self.window_size), min(self.n_rows, row+self.window_size)):
            for c in range(max(0, col-self.window_size), min(self.n_cols, col+self.window_size)):

                dist = ((row - r)**2 + (col-c)**2)**.5
                if dist <= self.window_size:
                    context_bitmap[c * self.n_cols + r] = 1

        context_bitmap[i] = 0
        return context_bitmap


    def build_training_data(self, routes, objective='skip-gram'):
        '''
        given a list of routes represented as 1-dimensional "bag of holds", builds a training set of (1-d) one-hot hold-vectors and 
        (1-d) context bitmaps. For any hold, we want to train on predicting the probability of nearby holds being used in the same route
        skip-gram:    given the context, predict which hold was left out
        cbow:         given a hold, predict how likely other holds are to show up together with it
        ''' 
        assert len(routes) > 0
        assert objective == 'skip-gram' or objective == 'cbow'

        for route_i in range(len(routes)):
            route = routes[route_i]
            for hold_i in np.where(route)[0]:   # iterate over indexes where there is a hold (value 1 instead of 0)
                one_hot = self.create_one_hot_hold_vector(hold_i)
                context_bitmap = self.create_context_bitmap(hold_i)  
                context = np.multiply(route, context_bitmap)
                if sum(context) > 0:
                    if objective == 'skip-gram':
                        context = context / sum(context) # for predicting context
                        self.X_train.append(context)
                        self.y_train.append(one_hot)
                    elif objective == 'cbow':
                        self.X_train.append(one_hot)
                        self.y_train.append(context)
                
        self.X_train = np.asarray(self.X_train)
        self.y_train = np.asarray(self.y_train)
        return