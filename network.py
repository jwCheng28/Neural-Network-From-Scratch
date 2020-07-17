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
        self.a = []
        self.delta = []

    def sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))

    def sigmoidGrad(self, z):
        a = self.sigmoid(z)
        return (a * (1 - a))

    def forward_(self, a, l):
        return self.sigmoid(np.dot(a, self.theta[l].T))

    def forwardFeed(self, X):
        self.a = []
        a = X.copy()
        for i in range(self.n_layers - 1):
            a = np.concatenate([np.ones((a.shape[0], 1)), a], axis=1)
            self.a.append(a)
            a = self.forward_(a, i)
        self.a.append(a)
        return a

    def costFunction(self, X, y):
        m = y.size
        y = y.copy()
        y = np.eye(10)[y]
        a = self.forwardFeed(X)
        J = (-1 / m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
        return J 

    def back_(self, l, delta):
        a = self.a[l-1]
        sg = self.sigmoidGrad(np.dot(a, self.theta[l-1].T))
        return np.dot(delta, self.theta[l])[:, 1:] * sg

    def backPropagation(self, y, a_l):
        delta =  a_l - y
        m = y.shape[0]
        self.delta = [delta]
        Delta = []
        thetaGrad = []
        for i in range(self.n_layers - 2, 0, -1):
            delta = self.back_(i, delta)
            self.delta.insert(0, delta)
        for i in range(len(self.delta)):
            Delta.append(np.dot(self.delta[i].T, self.a[i]))
        for i in range(len(self.theta)):
            thetaGrad.append((1 / m) * Delta[i]) 
        return thetaGrad

#n = Network([2, 3, 1])

#print(n.theta)