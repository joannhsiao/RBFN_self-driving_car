import numpy as np
from kmeans import kmeans

class rbf():
    def __init__(self, k, input, input_size, output_size):
        self.input = input
        self.output = None
        self.K = k
        # m: (k, in_dim, 1)
        self.m, self.std = kmeans(self.input, self.K)
        self.weights = np.random.randn(output_size, self.std.shape[1])
        self.bias = np.random.randn(output_size, 1)
        self.input = None

    def forward(self, input):
        self.input = input
        # euclidean.T: (k, 1) -> (1, k); same as std: (1, k)
        self.phi = np.exp(-1 * ((np.linalg.norm(self.input - self.m, axis=1).T)**2 / (2 * self.std**2)))
        # w: (1, k), phi: (1, k)
        return np.dot(self.weights, self.phi.T) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.phi)
        """ std """
        gradient = np.dot(output_gradient, self.weights)
        gradient = np.dot(gradient, self.phi.T)
        std_gradient = np.dot(gradient, (np.linalg.norm(self.input-self.m, axis=1).T)**2 / self.std**3)
        """ center """
        # gradient: (y-F) x w x phi = constant: (1, 1)
        m_gradient = np.dot((self.input - self.m), gradient)
        # squeeze (axis=(2,)): (k, in_dim, 1) -> (k, in_dim); std.T: (1, k) -> (k, 1)
        m_gradient = np.squeeze(m_gradient, axis=(2,)) / (self.std**2).T
        # reshape: (k, in_dim) -> (k, in_dim, 1)
        m_gradient = np.reshape(m_gradient, (m_gradient.shape[0], m_gradient.shape[1], 1))

        self.weights += learning_rate * weights_gradient
        self.bias += learning_rate * output_gradient    # phi_0 = 1
        self.std += learning_rate * std_gradient
        self.m += learning_rate * m_gradient
