import numpy as np


class FullyConnectedLayer:
    __output_size = 0
    __input_size = 0

    __W = None
    __b = None

    __previous_input = None

    def __init__(self, input_size: int, output_size: int):
        self.__input_size = 0
        self.__output_size = 0

        self.__W = np.random.rand(output_size, input_size)
        self.__b = np.random.rand(output_size)[:, None]

    def apply(self, input_vector: np.array) -> np.array:
        self.__previous_input = input_vector
        return np.dot(self.__W, input_vector) + self.__b

    def backprop(self, backprop_vector: np.array, step_size: float) -> np.array:
        dW = np.transpose(np.outer(self.__previous_input, backprop_vector))
        dx = np.dot(np.transpose(self.__W), backprop_vector)
        db = backprop_vector

        self.__W = np.add(self.__W, dW*step_size)
        self.__b = np.add(self.__b, db*step_size)

        return dx




