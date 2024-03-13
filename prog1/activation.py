import numpy as np
from math import tanh

class Activation:
    a: object
    d: object
    name: str


    def __init__(self, fn):
        self.name = fn
        match fn:
            case "sig":
                self.a = lambda Z: 1 / (1 + np.exp(-Z))
                self.d = lambda Z: 1 / (1 + np.exp(-Z)) * (1 - 1 / (1 + np.exp(-Z)))
            case "tanh":
                self.a = lambda Z: np.tanh(Z)
                self.d = lambda Z: 1 - np.square(np.tanh(Z))
            case "relu":
                self.a = lambda Z: np.maximum(0, Z)
                self.d = lambda Z: Z >= 0
            case "softmax":
                self.a = lambda Z: np.exp(Z) / np.sum(np.exp(Z))
                self.d = lambda Z: RuntimeError()
            case "ident": # default to identity
                self.a = lambda Z: Z
                self.d = lambda Z: 1


    def transform(self, Z):
        return self.a(Z)


    def derivative(self, Z):
        return self.d(Z)


    def __str__(self):
        return f"{self.name}"

