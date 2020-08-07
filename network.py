import numpy as np
import matplotlib.pyplot as plt
import random

'''
TO DO LIST:
 - Functions:
    - Add Softmax
 - Visualization:
    - Predict Probability Graph
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
    def costFunction(self, X, y, lambda_=0, getH=False):
        m = y.shape[0]
        a = self.forwardFeed(X)
        reg = (lambda_/m) * np.sum([np.sum(np.square(theta[:,1:])) for theta in self.theta])
        J = (-1 / m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) + reg
        if getH: return J, a
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

    # Data Split
    def splitData(self, X, y, size):
        X_split = X[size:, :]
        X = X[:size, :]
        
        y_split = y[size:, :]
        y = y[:size, :]

        return X, y, X_split, y_split 

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
    def stochasticGD(self, X, y, alpha, epoch, batch_size, lambda_=0, cv=0, costH=False, accurH=False, both=False, history=False):
        if history: J_hist = []
        if cv: 
            size = int(y.shape[0] * (1 - cv))
            X, y, X_cv, y_cv = self.splitData(X, y, size)
            cJ_hist = []

        for i in range(epoch):

            # Randome Order
            c = list(zip(X, y))
            random.shuffle(c)
            X_rand, y_rand = list(zip(*c))

            # Train in batches
            for j in range(0, len(X), batch_size):
                X_batches = X[j:j+batch_size]
                y_batches = y[j:j+batch_size]
                h = self.forwardFeed(X_batches)            
                thetaGrad = self.backPropagation(y_batches, h, lambda_)
                self.theta = self.theta - alpha * thetaGrad

            # Epoch Info Display Options    
            if both:
                costH = accurH = False
                J, h = self.costFunction(X, y, lambda_, True)
                if cv: 
                    cJ, ch = self.costFunction(X_cv, y_cv, lambda_, True)
                if history: 
                    J_hist.append(J)
                    if cv: cJ_hist.append(cJ)
                
                print(
                    "Epoch {} : Trainig Cost = {}, Training Accuracy = {}".format(
                        i + 1, round(J, 8) , round(self.accuracy(h, y), 8) 
                        )
                    )
                if cv: print(
                    "        CV Cost = {}, CV Accuracy = {}".format(
                        round(cJ, 8), round(self.accuracy(ch, y_cv), 8)
                        )
                    )

            if costH: print("Epoch {} Cost : {} ".format(i + 1, self.costFunction(X, y, lambda_)))
        
            if accurH: print("Epoch {} Training Accuracy : {} ".format(i + 1, self.accuracy(self.forwardFeed(X), y)))
        
            if not (costH or accurH or both): print("Epoch {} Completed".format(i + 1))
        
        if history:
            if cv: return h, J_hist, cJ_hist 
            return h, J_hist
        return h

    # Output test performance
    def accuracy(self, h, y):
        h = np.argmax(h, axis=1)
        y = np.where(y==1)[1]
        return np.mean(h==y)

    # Predict IMG
    def predict(self, X, h, prob=False, save_file=None):
        fig, (ax1, ax2) = plt.subplots(figsize=(6.4, 4),ncols=2)
        n = np.random.randint(0, len(h))
        predict = np.argmax(h, axis=1)[n]
        print("Prediction: " + str(predict))

        img_set = np.asfarray(X)
        img = img_set[n].reshape((28,28))
        ax1.imshow(img, cmap="Greys")
        ax1.set_title("Test Image")
        ax1.axis('off')

        ax2.bar([i for i in range(len(h[n]))],list(h[n]))
        ax2.set_xticks([i for i in range(10)])
        ax2.set_yticks([0.2 * i for i in range(1, 6)])
        asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
        ax2.set_aspect(asp)

        ax2.set_title("Model Confidence for Each Label")
        ax2.set_ylabel("Confidence")
        ax2.set_xlabel("Labels")

        fig.tight_layout()
        if save_file:
            plt.savefig("pics/" + save_file + "_{}.png".format(n))
        plt.show()

    # Cost over Epoch Graph
    def costHistory(self, history, cv_history=None, save_file=None):
        ep = [i for i in range(1, len(history) + 1)]
        plt.plot(ep, history, color="#14d0f0", label="Training Cost")
        if cv_history:
            plt.plot(ep, cv_history, color="#ffb3ba", label="CV Cost")
        plt.xticks(ep)
        plt.xlabel("Epoch")
        plt.ylabel("Training Cost")
        plt.title("Cost over Epoch")
        plt.legend()
        if save_file:
            plt.savefig("pics/" + save_file + ".png")
        plt.show()