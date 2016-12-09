import numpy as np


class LSTM:
    __activation_function = None
    __input_size = 0

    # Input Gate Variables
    __W_xi = None
    __W_hi = None
    __W_ci = None
    __b_i = None

    # Forget Gate Variables
    __W_xf = None
    __W_hf = None
    __W_cf = None
    __b_f = None

    # Output Gate Variables
    __W_xo = None
    __W_ho = None
    __W_co = None
    __b_o = None

    # Cell Variables
    __W_xc = None
    __W_hc = None
    __b_c = None

    # Final Layer
    __W_hy = None
    __b_y = None

    __previous_c = None
    __previous_h = None

    def __init__(self, input_size: int, output_size: int, activation_function: staticmethod):
        self.__activation_function = activation_function
        self.__input_size = input_size

        # Initialize the Input Gate
        self.__W_xi = np.random.rand(input_size, input_size)
        self.__W_hi = np.random.rand(input_size, input_size)
        self.__W_ci = np.random.rand(input_size, input_size)
        self._b_i = np.random.rand(input_size)

        # Initialize the Forget Gate
        self.__W_xf = np.random.rand(input_size, input_size)
        self.__W_hf = np.random.rand(input_size, input_size)
        self.__W_cf = np.random.rand(input_size, input_size)
        self.__b_f = np.random.rand(input_size)

        # Initialize the Output Gate
        self.__W_xo = np.random.rand(input_size, input_size)
        self.__W_ho = np.random.rand(input_size, input_size)
        self.__W_co = np.random.rand(input_size, input_size)
        self.__b_o = np.random.rand(input_size)

        # Cell Variables
        self.__W_xc = np.random.rand(input_size, input_size)
        self.__W_hc = np.random.rand(input_size, input_size)
        self.__b_c = np.random.rand(input_size)

        # Final Layer Variables
        self.__W_hy = np.random.rand(input_size, output_size)
        self._b_y = np.random.rand(output_size)

        self.__previous_c = np.random.rand(input_size)
        self.__previous_h = np.random.rand(input_size)

    def apply(self, input_vector: np.array) -> np.array:
        input_vector = self.__activation_function(input_vector)
        input_gate_result = self.__calculate_input_gate(input_vector)
        forget_gate_result = self.__calculate_forget_gate(input_vector)
        cell_value = self.__calculate_cell_value(input_vector, input_gate_result, forget_gate_result)
        output_gate_result = self.__calculate_output_gate(input_vector, cell_value)

        h = np.multiply(output_gate_result, np.tanh(cell_value))

        self.__previous_c = cell_value
        self.__previous_h = h

        output = np.add(np.multiply(self.__W_hy, h), self.__b_y)

        return output

    def __calculate_input_gate(self, input_vector: np.array) -> np.array:
        first_term = np.multiply(self.__W_xi, input_vector)
        second_term = np.multiply(self.__W_hi, self.__previous_h)
        third_term = np.multiply(self.__W_ci, self.__previous_c)

        summation = np.add(first_term, second_term)
        summation = np.add(summation, third_term)
        summation = np.add(summation, self.__b_i)

        return self.__activation_function(summation)

    def __calculate_forget_gate(self, input_vector: np.array) -> np.array:
        first_term = np.multiply(self.__W_xf, input_vector)
        second_term = np.multiply(self.__W_hf, self.__previous_h)
        third_term = np.multiply(self.__W_cf, self.__previous_c)

        summation = np.add(first_term, second_term)
        summation = np.add(summation, third_term)
        summation = np.add(summation, self.__b_f)

        return self.__activation_function(summation)

    def __calculate_cell_value(
            self, input_vector: np.array, forget_gate_vector: np.array,
            input_gate_vector: np.array) -> np.array:

        first_term = np.multiply(self.__W_xc, input_vector)
        second_term = np.multiply(self.__W_hc, self.__previous_h)

        summation = np.add(first_term, second_term)
        summation = np.add(summation, self.__b_c)

        tanh_result = np.tanh(summation)
        tanh_result = np.multiply(input_gate_vector, tanh_result)

        first_term = np.multiply(forget_gate_vector, self.__previous_c)

        return np.add(first_term, tanh_result)

    def __calculate_output_gate(self, input_vector: np.array, cell_value: np.array) -> np.array:
        first_term = np.multiply(self.__W_xo, input_vector)
        second_term = np.multiply(self.__W_ho, self.__previous_h)
        third_term = np.multiply(self.__W_co, cell_value)

        summation = np.add(first_term, second_term)
        summation = np.add(summation, third_term)
        summation = np.add(summation, self.__b_o)

        return self.__activation_function(summation)