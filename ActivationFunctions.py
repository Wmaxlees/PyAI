import numpy as np


class ActivationFunctions:
    @staticmethod
    def apply_logistic_sigmoid(input_vector: np.array) -> np.array:
        """A function to apply the sigmoid function to an input vector:

        Args:
            input_vector (numpy.array): The input vector to apply the sigmoid
            function to.

        Returns:
            numpy.array: The result of the sigmoid function on the input vector.

        """
        return 1.0 / (1.0 + np.exp(input_vector))

    @staticmethod
    def apply_rectifier_linear_unit(input_vector: np.array) -> np.array:
        """A function to apply the rectifier linear unit to an input vector:

        Args:
            input_vector (numpy.array): The input vector to apply the rectifier
            linear unit to.

        Returns:
            numpy.array: The result of the rectifier linear unit on the input vector.

        """
        zeros = np.zeros(input_vector.size)
        return np.maximum(input_vector, zeros)

    @staticmethod
    def apply_softmax(input_vector: np.array) -> np.array:
        exponentiated = np.exp(input_vector)
        bottom = np.sum(exponentiated)

        return exponentiated/bottom


