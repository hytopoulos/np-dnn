import numpy as np

from activation import Activation

class Layer:
    w: np.matrix # weights matrix
    b: np.matrix # bias vector
    z: np.matrix # last forward linear transformation
    a: np.matrix # last forward output post-activation
    activation: Activation # activation on this layer


    def __init__(self, D1: int, D2: int, activation: str, init_range: float):
        '''
            D1: layer input dim
            D2: layer output dim (unit count)
            activation: activation function ["tanh", "sig", "relu"]
            init_range: range of distribution for weight initialization
        '''
        self.activation = Activation(activation)

        self.w = np.random.uniform(-init_range, init_range, (D1, D2))
        self.b = np.random.uniform(-init_range, init_range, (D2,))


    def forward(self, x: np.matrix):
        ''' Linear transform and activation of input

            x: NxD matrix of N input data points in R^D
        '''
        self.z = x @ self.w + self.b
        self.a = self.activation.transform(self.z)
        return self.a


    def transform(self, x: np.matrix):
        z = x @ self.w + self.b
        return self.activation.transform(z)


    def __str__(self):
        return f"[ w={self.w.shape}, b={self.b.shape}, a={self.activation} ]"
