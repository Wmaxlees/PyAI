import numpy as np


class ConvolutionalLayer:
    __K = 0
    __F = 0
    __S = 0
    __P = 0
    __input_height = 0
    __input_width = 0
    __depth = 0

    __filters = None

    def __init__(self, k: int, f: int, s: int, p: int,
                 input_height: int, input_width: int, depth: int=3):
        self.__K = k
        self.__F = f
        self.__S = s
        self.__P = p
        self.__input_height = input_height
        self.__input_width = input_width
        self.__depth = depth

        frames_per_width = int((input_width - self.__F + 1) / self.__S)
        frames_per_height = int((input_height - self.__F + 1) / self.__S)
        self.__filters = np.random.rand(frames_per_height * frames_per_width, k)

    def apply(self, input_matrix: np.array) -> np.array:
        columnated = self.__im2col(input_matrix)

        print(columnated.shape)
        print(self.__filters.shape)

        raw_results = np.dot(columnated, self.__filters)

        return raw_results

    def __im2col(self, input_matrix: np.array) -> np.array:
        input_width, input_height, depth = input_matrix.shape

        frames_per_width = int((input_width - self.__F + 1) / self.__S)
        frames_per_height = int((input_height - self.__F + 1) / self.__S)

        consecutive_numbers = np.arange(self.__F * depth)
        column = np.append(consecutive_numbers, consecutive_numbers + input_width * depth)
        column = column[:, None]

        consecutive_numbers = np.arange(frames_per_width) * depth
        row = consecutive_numbers
        for i in range(frames_per_height - 1):
            row = np.append(row, consecutive_numbers + (i + 1) * depth * input_width)

        selection_array = row + column

        return np.take(input_matrix, selection_array)

    def __col2im(self, input_matrix: np.array, original_width: int, original_height: int) -> np.array:
        result_width = (original_width - self.__F + (2*self.__P)) / self.__S + 1
        result_height = (original_height - self.__F + (2*self.__P)) / self.__S + 1
        result_depth = self.__K

        
