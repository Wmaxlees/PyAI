import numpy as np


class SigmoidActivationFunction:
    @staticmethod
    def apply(input_vector: np.array) -> np.array:
        return 1.0 / (1.0 + np.exp(input_vector))

    @staticmethod
    def backprop(backprop_vector: np.array) -> np.array:
        bottom_half = (1 - np.exp(backprop_vector))
        return np.exp(backprop_vector) / (bottom_half * bottom_half)
