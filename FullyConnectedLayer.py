import numpy as np

class FullyConnectedLayer:
    __output_size = 0
    __input_size = 0

    __W = None
    __b = None

    def __init__(self, input_size: int, output_size: int):
        self.__input_size = 0
        self.__output_size = 0

        self.__W = np.random.rand(output_size, input_size)
        self.__b = np.random.rand(output_size)

    def apply(self, input_vector: np.array) -> np.array:
        return self.__W * input_vector + self.__b

