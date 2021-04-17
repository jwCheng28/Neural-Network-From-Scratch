import matplotlib.pyplot as plt
import numpy as np
import pickle
import gzip

def load_data(binary=True):
    loc = "dataset/"
    train, test = loc + "mnist_train.pydb.gz", loc + "mnist_test.pydb.gz"

    train_file, test_file = gzip.GzipFile(train, 'rb'), gzip.GzipFile(test, 'rb')
    train_data, test_data = pickle.load(train_file), pickle.load(test_file)
    train_file.close()
    test_file.close()

    # Reformat RGB values into Binary
    if binary:
        train_data = (np.concatenate([
            train_data[:, :1], 
            (train_data[:, 1:] > 0).astype(int)
            ], axis=1)).astype(int)
        
        test_data = (np.concatenate([
            test_data[:, :1], 
            (test_data[:, 1:] > 0).astype(int)
            ], axis=1)).astype(int)

    return train_data, test_data


def display(n, img_data):
    imgs_arr = np.asfarray(img_data[:, 1:])
    for i in range(n):
        img = imgs_arr[i].reshape((28,28))
        plt.imshow(img, cmap="Greys")
        plt.show()
        
# Seperate data into X, y
def seperate(data):
    X = data[:, 1:]
    # One hot encoding
    y = np.eye(10)[data[:, 0]]
    return X, y
