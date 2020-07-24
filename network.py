import numpy as np
import matplotlib.pyplot as plt
import random

'''
TO DO LIST:
 - Functions:
    - Add Softmax
    - Add ReLU
 - Visualization:
    - Cost Graph
    - Learning Curves (Train vs CV)
    - Predict Probability Graph
'''

'''
Note To Self:
 - Need to re-implement/change the class architecture
    - don't use self.a
    - Change Cost Function & Back_prop implementation for modularity
'''

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

    # Sigmoid Function
    def sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))

    # ReLU Function
    def reLU(self, z):
        return max(0, z)

    # Gradient of Sigmoid Function
    def sigmoidGrad(self, z):
        a = self.sigmoid(z)
        return (a * (1 - a))

    # Pass Linear Function into Sigmoid, and Returns it
    def forward_(self, a, l):
        return self.sigmoid(np.dot(a, self.theta[l].T))

    # One Feed Forward
    def forwardFeed(self, X):
        self.a = []
        a = X.copy()
        for i in range(self.n_layers - 1):
            a = np.concatenate([np.ones((a.shape[0], 1)), a], axis=1)
            self.a.append(a)
            a = self.forward_(a, i)
        self.a.append(a)
        return a

    # Cost/Loss Function using Cross Entropy
    def costFunction(self, X, y, lambda_=0):
        m = y.shape[0]
        a = self.forwardFeed(X)
        reg = (lambda_/m) * np.sum([np.sum(np.square(theta[:,1:])) for theta in self.theta])
        J = (-1 / m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) + reg
        return J 

    # Returns derivative of backing one layer
    def back_(self, l, delta):
        a = self.a[l-1]
        sg = self.sigmoidGrad(np.dot(a, self.theta[l-1].T))
        return np.dot(delta, self.theta[l])[:, 1:] * sg

    # One Back Propagation
    def backPropagation(self, y, a_l, lambda_=0):
        # Initial delta, delta of last layer
        delta =  a_l - y
        m = y.shape[0]
        self.delta = [delta]
        Delta = []
        thetaGrad = []
        # Calculate delta of all layers except first
        for i in range(self.n_layers - 2, 0, -1):
            delta = self.back_(i, delta)
            self.delta.insert(0, delta)
        # Calculate delta.dot(a) (all node val of layer) for all layers
        for i in range(len(self.delta)):
            Delta.append(np.dot(self.delta[i].T, self.a[i]))
        # Calculate of Gradient Cost/Loss Function
        for i in range(len(self.theta)):
            thetaGrad.append((1 / m) * Delta[i]) 
        thetaGrad = np.array(thetaGrad)
        # Add Regularization for Gradient
        for i in range(len(thetaGrad)):
            thetaGrad[i][:, 1:] = thetaGrad[i][:, 1:] + (lambda_/m) * self.theta[i][:, 1:]
        return thetaGrad

    # Performs Gradient Descent
    def gradientDescent(self, X, y, alpha, epoch, lambda_=0, costH=False, accurH=False):
        for i in range(epoch):
            h = self.forwardFeed(X)
            if costH: print("Epoch {} : {} ".format(i, self.costFunction(X, y, lambda_)))
            if accurH: print("Epoch {} : {} ".format(i, self.accuracy(h, y)))
            thetaGrad = self.backPropagation(y, h, lambda_)
            self.theta = self.theta - alpha * thetaGrad
            if not (costH or accurH): print("Epoch {} Completed".format(i))
        h = self.forwardFeed(X)
        print("Results - Cost : {}, Accuracy : {}".format(self.costFunction(X, y, lambda_), self.accuracy(h, y)))        
        return h

    # Performs Stochastic Gradient Descent
    def stochasticGD(self, X, y, alpha, epoch, batch_size, lambda_=0, costH=False, accurH=False):
        for i in range(epoch):
            ind = random.randint(0, len(X)-batch_size)
            X_batches, y_batches = X[ind:ind+batch_size, :], y[ind:ind+batch_size, :]
            h = self.forwardFeed(X_batches)
            if costH: print("Epoch {} : {} ".format(i, self.costFunction(X_batches, y_batches, lambda_)))
            if accurH: print("Epoch {} : {} ".format(i, self.accuracy(h, y_batches)))
            thetaGrad = self.backPropagation(y_batches, h, lambda_)
            self.theta = self.theta - alpha * thetaGrad
            if not (costH or accurH): print("Epoch {} Completed".format(i))
        h = self.forwardFeed(X)
        print("Results - Cost : {}, Accuracy : {}".format(self.costFunction(X, y, lambda_), self.accuracy(h, y)))        
        return h

    # Output test performance
    def accuracy(self, h, y):
        h = np.argmax(h, axis=1)
        y = np.where(y==1)[1]
        return np.mean(h==y)

    # Predict IMG
    def predict(self, X, h):
        n = np.random.randint(0, len(X))
        predict = np.argmax(h, axis=1)[n]
        print("Prediction: " + str(predict))
        img_set = np.asfarray(X)
        img = img_set[n].reshape((28,28))
        plt.imshow(img, cmap="Greys")
        plt.show()