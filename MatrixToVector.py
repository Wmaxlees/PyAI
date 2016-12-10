import numpy as np


class MatrixToVector:
    __original_dimensions = None

    def apply(self, input_matrix: np.array) -> np.array:
        self.__original_dimensions = input_matrix.shape

        return input_matrix.flatten()

    def undo(self, input_vector: np.array) -> np.array:
        return np.reshape(input_vector, self.__original_dimensions)
