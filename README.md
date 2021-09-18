# Neural Network from Scratch

## Basic Idea
This project aims to implement a neural network class with various different functionalities from scratch, where the user can create their own artificial neural network of any size.

To test out the neural network class, I've created a demo notebook where we used MNIST dataset of handwritten digit images to train the neural network to read/recognize Handwritten Digit Images.

## Implementation
If you're curious how the Neural Network is created, all the code for it could be found in the ```network.py``` file; or if you want to create a neural network you can read the example below or look at the testing file in ```demo/mnist_digt_recognition_demo.ipynb```.

## Get Started
This project is written in Python 3, so be sure to have python installed first. Then simply run `pip install -r requirements.txt` on the command line to install all the necessary libraries.

## Create and Train a Neural Network using this Class
For this example we will be creating a neural network for reading handwritten digits.
### Importing Libraries
To get start first we import the necesarry library and neural network classes.
```python
import neural_net as nn
import numpy as np
```
### Getting the Data
You can try it out with your own data, or if you want to follow along you can use the python script and dataset in the demo folder.
```python
import load_mnist
# Get train and test data
train, test = load_mnist.load_data()

# Seperate data into training set and labels
X, y = load_mnist.seperate(train)
```
### Create and Train our Neural Network
To create a neural network, we could simply initialize a network from our network class, with a list the amount of neurons/nodes for each layer.
```python
# 784 input nodes, 128 nodes in hidden layer, 10 output nodes
neuralNet = nn.Network([784, 128, 10])
```
This class contain 2 options to train the neural network, Gradient Descent and Stochastic Gradient Descent. In our case, with a large dataset we'll use Stochastic Gradient Descent.
```python
history = neuralNet.stochasticGD(X, y, 0.33, 12, 64, lambda_=0.05, cv=0.1, both=True) 
```
For this training we used:
```
learning rate = 0.33, epochs = 16, batch size = 64
regularization lambda = 0.05, cross validation size = 0.1
both = True to display all training info
```
### See Test Results
We could test the performance of our trained neural network by looking at the accuracy of the neural net.
```python
# Get Predictions
predictions = neuralNet.forwardFeed(X)
accuracy = neuralNet.accuracy(predictions, y)
print("Test Accuracy:", accuracy)
```

If you follow along the demo, I've created functions in the notebook for visualizing the mnist result as shown below.
![alt text 1](https://github.com/jwCheng28/Neural-Network-From-Scratch/blob/master/pics/img_confidence_407.png) ![alt text 2](https://github.com/jwCheng28/Neural-Network-From-Scratch/blob/master/pics/img_confidence_52081.png)

We could also plot the Training and Cross Validation Cost & Accuracy history to ensure we are not overfitting.
```python
neuralNet.costHistory(history)
neuralNet.accurHistory(history)
```
![alt text 1](https://github.com/jwCheng28/Neural-Network-From-Scratch/blob/master/pics/history_new.png) ![alt text 2](https://github.com/jwCheng28/Neural-Network-From-Scratch/blob/master/pics/accur_hist.png)

In our case, both the training and cross validation Cost & Accuracy are relatively the same, so it doesn't seem like our trained neural network is overfitting.

Finally, we should test our results on the entire test data to see how well our Neural Network performs on unseen data.
```python
X_test, y_test = load_mnist.seperate(test)
result = neuralNet.forwardFeed(X_test)
accuracy = neuralNet.accuracy(result, y_test)
```
In my case, the trained Neural Network achieved 95% accuracy which is not too bad for a simple neural network.

So now we're done; we've successfuly created and train a neural network to recognize handwritten digits.

## Data Source
The original handwritten digits and label data are from MNIST. The dataset I used for my training is from Kaggle (https://www.kaggle.com/c/digit-recognizer/data), which has been process into CSV format. The dataset in the dataset/ directory are the final version used for this project, which I've process the CSV data into numpy arrays and compressed into gzip files.
