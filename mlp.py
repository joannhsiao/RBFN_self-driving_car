import numpy as np
from network import Dense
from activation import Tanh, Sigmoid
from Loss import *

def predict(network, data):
    output = data
    for layer in network:
    	output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs=50, learning_rate=0.01):
	for epoch in range(epochs):
		error = 0
		acc = 0
		for x, y in zip(x_train, y_train):
			output = predict(network, x)
			error += loss(y, output)
			grad = loss_prime(y, output)	# backward
			for layer in reversed(network):
				grad = layer.backward(grad, learning_rate)
		error /= len(x_train)
		
		print("Epoch {} | loss: {:g}".format(epoch, error))

def data_process(file):
	with open(file, "r") as f:
		lines = f.readlines()
	data = []
	y = []
	for num, line in enumerate(lines):
		xdata = line.split()
		y.append(float(xdata[-1]))
		for i in range(len(xdata)):
			xdata[i] = float(xdata[i])
		data.append(xdata[0:-1])
		input_dim = len(xdata) - 1
	x_train = np.reshape(data, (len(data), input_dim, 1))
	y_train = np.reshape(y, (len(y), 1, 1))

	# normalization
	max_y = 40
	min_y = -40
	y_train = (y_train - min_y) / (max_y - min_y)
	return x_train, y_train, input_dim

def process(file, epoch, lr):
	x_train, y_train, input_dim = data_process(file)
	network = [
		Dense(input_dim, 64),
		Sigmoid(),
		Dense(64, 1), 
		Sigmoid()
	]
	Train_acc = train(network, mse, mse_prime, x_train, y_train, epoch, lr)
	y_train = np.reshape(y_train, (len(y_train), 1))
	return network, input_dim

"""
if __name__ == "__main__":
	process("train4dAll.txt", 100, 0.01)
"""