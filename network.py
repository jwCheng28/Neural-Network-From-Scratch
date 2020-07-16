import numpy as np

class Network():
    def __init__(self, structure):
        '''
        structure is an array that indicates number of neurons in each layer
            ex. [2, 3, 1] -> 2 input, 3 in hidden, 1 output
        '''
        self.structure = structure
        self.n_layers = len(structure)
        # Initialize Random Weights d = s_(j+1) x (s_j + 1)
        self.theta = [
            np.random.randn(r, c + 1) 
            for r, c in zip(structure[1:], structure[:-1])
            ]

    def sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))

    def forward_(self, a, l):
        return self.sigmoid(np.dot(a, self.theta[l].T))

    def forwardFeed(self, X):
        a = X.copy()
        for i in range(self.n_layers - 1):
            a = np.concatenate([np.ones((a.shape[0], 1)), a], axis=1)
            a = self.forward_(a, i)
        return a

    def costFunction(self, X, y):
        m = y.size
        y = y.copy()
        y = np.eye(10)[y]
        a = self.forwardFeed(X)
        J = (-1 / m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
        return J 

    #def back_(self, l)

#n = Network([2, 3, 1])

#print(n.theta)