# Neural Network for Reading Handwritten Digits

## Basic Idea
This project aims to implement a neural network class with various different functionalities from scratch.

Afterwards, we'll use this created neural network class to create a neural network and use MNIST dataset of handwritten digit images to train the neural network to read/recognize Handwritten Digit Images.

## Create and Train a Neural Network
For this example we will be creating a neural network for reading handwritten digits.
### Get Started
To get start first we import the necesarry library and the created neural network & data managing classes.
```
import data_manager as dm
import network
import numpy as np
```
### Getting the Data
We can get the necessary data from the data_manager (created to manage and clean mnist data files) python file.
```
# Get train and test data
train, test = dm.load_data()

# Seperate data into training set and labels
X, y = dm.seperate(train)

# One hot encode all the labels
y = np.eye(10)[y]
```
### Create and Train our Neural Network
To create a neural network, we could simply initialize a network from our network class, with a list the amount of neurons/nodes for each layer.
```
# 784 input nodes, 100 nodes in hidden layer, 10 output nodes
ann = network.Network([784, 100, 10])
```
This class contain 2 options to train the neural network, Gradient Descent and Stochastic Gradient Descent. In our case, with a large dataset we'll use Stochastic Gradient Descent.
```
h, history, cv_history = ann.stochasticGD(X, y, 0.3, 16, 32, 0, cv=0.1, both=True, history=True) 
```
For this training we used:
```
learning rate = 0.3, epochs = 16, batch size = 32
regularization lambda = 0, cross validation size = 0.1
both=True for display all train info, history=True to save history cost
```
### Training Results
With the training parameters as stated above the following results are what I achieved in my testing.
```
Epoch 16 : Trainig Cost = 0.29416113, Training Accuracy = 0.9602037
           Cross Validation Cost = 0.35999273, Cross Validation Accuracy = 0.95133333
```
We could test the performance of our trained neural network by using the predict function in the neural network class to see how well the neural network recognize pictures of handwritten digit.
```
ann.predict(X, h)
```
![alt text 1](https://github.com/jwCheng28/Neural-Network-From-Scratch/blob/master/pics/img_confidence_407.png) ![alt text 2](https://github.com/jwCheng28/Neural-Network-From-Scratch/blob/master/pics/img_confidence_52081.png)

We could also plot the Training and Cross Validation Cost history to ensure we are not overfitting.
```
ann.costHistory(history, cv_history)
```
![alt text](https://github.com/jwCheng28/Neural-Network-From-Scratch/blob/master/pics/history.png) 

In our case, both the training and cross validation cost are relatively the same, so it doesn't seem like our trained neural network is overfitting.

So now we're done; we've successfuly created and train a neural network to recognize handwritten digits.

## Data Source
The original handwritten digits and label data are from MNIST. The dataset I used for my training is from Kaggle (https://www.kaggle.com/c/digit-recognizer/data), which has been process into CSV format. The dataset in the dataset/ directory are the final version used for this project, which I've process the CSV data into numpy arrays and compressed into gzip files.
