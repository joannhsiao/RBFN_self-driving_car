import numpy as np
from rbf import rbf
from Loss import *

def predict(network, data):
    output = data
    output = network.forward(output)
    return output

def train(network, loss, loss_prime, X, Y, epochs=50, learning_rate=0.01):
	for epoch in range(epochs):
		error = 0
		for x, y in zip(X, Y):
			output = predict(network, x)
			#print(output, y)
			error += loss(y, output)
			grad = loss_prime(y, output)
			network.backward(grad, learning_rate)
		error /= len(X)
		print("Epoch  {}, training loss: {:g}".format(epoch, error))

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

	# normalizatin
	max_y = 40
	min_y = -40
	y_train = (y_train - min_y) / (max_y - min_y)
	return x_train, y_train, input_dim

def process(file, epochs, lr):
	x_train, y_train, input_dim = data_process(file)
	network = rbf(5, x_train, input_dim, 1)
	train(network, lms, lms_prime, x_train, y_train, epochs, lr)
	return network, input_dim
