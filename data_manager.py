import numpy as np
import matplotlib.pyplot as plt
import pickle

loc = "dataset/"
train, test = loc + "mnist_train.pydb", loc + "mnist_test.pydb"
train_data = pickle.load(open(train, "rb"))
test_data =  pickle.load(open(test, "rb"))

def display(n):
    fac = 0.99 / 255
    train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
    #test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

    train_labels = np.asfarray(train_data[:, :1])
    #test_labels = np.asfarray(test_data[:, :1])
    
    for i in range(n):
        img = train_imgs[i].reshape((28,28))
        plt.imshow(img, cmap="Greys")
        plt.show()
        