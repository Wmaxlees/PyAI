import numpy as np

class MaxPoolingLayer:
    __filter_size = 0
    __stride_length = 0
    __input_width = 0
    __input_height = 0
    __input_depth = 0

    def __init__(self, filter_size: int, stride_length: int, input_width: int, input_height: int, input_depth: int):
        self.__filter_size = filter_size
        self.__stride_length = stride_length
        self.__input_width = input_width
        self.__input_height = input_height
        self.__input_depth = input_depth

    def apply(self, input_matrix: np.array) -> np.array:
        columnated = self.__im2col(input_matrix)

        selection_matrix = np.argmax(columnated, 1)
        offsetter = np.arange(columnated.shape[0]) * columnated.shape[1]
        selection_matrix = selection_matrix + offsetter

        return self.__col2im(np.take(columnated, selection_matrix))

    def __im2col(self, input_matrix: np.array) -> np.array:
        input_width, input_height, depth = input_matrix.shape
        filters_per_width = int((input_width - self.__filter_size) / self.__stride_length) + 1
        filters_per_height = int((input_height - self.__filter_size) / self.__stride_length) + 1

        consecutive_numbers = np.arange(self.__filter_size)
        filter_mask = consecutive_numbers
        for i in range(self.__filter_size - 1):
            filter_mask = np.append(filter_mask, consecutive_numbers+input_width)

        single_row = np.arange(filters_per_width) * self.__stride_length
        column = single_row
        for i in range(filters_per_height-1):
            column = np.append(column, single_row + ((i+1) * input_width))

        final = column
        for i in range(depth-1):
            final = np.append(final, column + ((i+1) * input_width * input_height))
        final = final[:, None]

        selection_matrix = filter_mask + final

        return np.take(input_matrix, selection_matrix)

    def __col2im(self, input_matrix: np.array) -> np.array:
        result_width = int((self.__input_width - self.__filter_size) / self.__stride_length) + 1
        result_height = int((self.__input_height - self.__filter_size) / self.__stride_length) + 1
        result_depth = self.__input_depth

        single_depth = np.arange(result_depth)
        single_column = np.arange(result_height)[:, None] * result_depth
        single_row = np.arange(result_width)[:, None][:, :, None] * (result_depth * result_height)

        selection_matrix = single_depth + single_column + single_row

        return np.take(input_matrix, selection_matrix)
