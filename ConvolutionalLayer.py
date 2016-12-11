import numpy as np


class ConvolutionalLayer:
    __number_of_filters = 0
    __filter_size = 0
    __step_size = 0
    __zero_padding = 0
    __input_height = 0
    __input_width = 0
    __depth = 0

    __W = None
    __b = None

    def __init__(self, number_of_filters: int, filter_size: int, step_size: int, zero_padding: int,
                 input_height: int, input_width: int, depth: int=3):
        self.__number_of_filters = number_of_filters
        self.__filter_size = filter_size
        self.__step_size = step_size
        self.__zero_padding = zero_padding
        self.__input_height = input_height
        self.__input_width = input_width
        self.__depth = depth

        self.__W = np.random.rand(number_of_filters, filter_size * filter_size * depth)
        self.__b = np.random.rand(number_of_filters)[:, None]

    def apply(self, input_matrix: np.array) -> np.array:
        columnated = self.__im2col(input_matrix)

        raw_results = np.dot(self.__W, columnated) + self.__b

        return self.__col2im(raw_results)

    def __im2col(self, input_matrix: np.array) -> np.array:
        input_width, input_height, depth = input_matrix.shape

        frames_per_width = int((input_width - self.__filter_size) / self.__step_size) + 1
        frames_per_height = int((input_height - self.__filter_size) / self.__step_size) + 1

        consecutive_numbers = np.arange(self.__filter_size * depth)
        column = consecutive_numbers
        for i in range(self.__filter_size - 1):
            column = np.append(column, consecutive_numbers + input_width * (i + 1) * depth)
        column = column[:, None]

        consecutive_numbers = np.arange(frames_per_width) * depth
        row = consecutive_numbers
        for i in range(frames_per_height - 1):
            row = np.append(row, consecutive_numbers + (i + 1) * depth * input_width)

        selection_array = row + column

        return np.take(input_matrix, selection_array)

    def __col2im(self, input_matrix: np.array) -> np.array:
        result_width = int((self.__input_width - self.__filter_size + (2*self.__zero_padding)) / self.__step_size + 1)
        result_height = int((self.__input_height - self.__filter_size + (2*self.__zero_padding)) / self.__step_size + 1)
        result_depth = self.__number_of_filters

        depth = np.arange(result_depth)
        column = np.arange(result_height)[:, None] * result_depth
        row = np.arange(result_width)[:, None][:, :, None] * result_depth * result_height

        selection_matrix = depth + column + row

        return np.take(input_matrix, selection_matrix)

    def backprop(self, backprop_matrix: np.array) -> np.array:
        columnation = self.__im2col_back(backprop_matrix)
        print(columnation)

    def __im2col_back(self, backprop_matrix: np.array) -> np.array:
        input_width, input_height, depth = backprop_matrix.shape

        single_row = np.arange(input_height * input_width)
        single_column = np.arange(depth)[:, None]

        selection_matrix = single_row + single_column

        return np.take(backprop_matrix, selection_matrix)

