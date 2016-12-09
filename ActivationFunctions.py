import numpy


class ActivationFunctions:
    @staticmethod
    def apply_logistic_sigmoid(input_vector: numpy.array) -> numpy.array:
        """A function to apply the sigmoid function to an input vector:

        Args:
            input_vector (numpy.array): The input vector to apply the sigmoid
            function to.

        Returns:
            numpy.array: The result of the sigmoid function on the input vector.

        """
        return 1.0 / (1.0 + numpy.exp(input_vector))

    @staticmethod
    def apply_rectifier_linear_unit(input_vector: numpy.array) -> numpy.array:
        """A function to apply the rectifier linear unit to an input vector:

        Args:
            input_vector (numpy.array): The input vector to apply the rectifier
            linear unit to.

        Returns:
            numpy.array: The result of the rectifier linear unit on the input vector.

        """
        zeros = numpy.zeros(input_vector.size)
        return numpy.maximum(input_vector, zeros)
