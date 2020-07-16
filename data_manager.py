import matplotlib.pyplot as plt
import numpy as np
import pickle
import gzip

def load_data(binary=True):
    loc = "dataset/"
    train, test = loc + "mnist_train.pydb.gz", loc + "mnist_test.pydb.gz"

    train_file = gzip.GzipFile(train, 'rb')
    test_file = gzip.GzipFile(test, 'rb')

    train_data = pickle.load(train_file)
    test_data = pickle.load(test_file)

    train_file.close()
    test_file.close()

    if binary:
        train_data = (train_data > 0).astype(int)
        test_data = (test_data > 0).astype(int)

    return train_data, test_data


def display(n, train_data, test_data=None):
    train_imgs = np.asfarray(train_data[:, 1:])
    #test_imgs = np.asfarray(test_data[:, 1:])

    train_labels = np.asfarray(train_data[:, :1])
    #test_labels = np.asfarray(test_data[:, :1])

    for i in range(n):
        img = train_imgs[i].reshape((28,28))
        plt.imshow(img, cmap="Greys")
        plt.show()
        
# Seperate data into X, y
def seperate(data):
    X = data[:, 1:]
    y = data[:, 0]
    return X, y