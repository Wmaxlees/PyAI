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

        width = ((input_height - f) / s) * ((input_width - f) / s) * 3
        self.__filters = np.random.rand(k, width)

    def apply(self, input_matrix: np.array) -> np.array:
        columnated = self.__im2col(input_matrix)
        print(columnated)

        raw_results = np.multiply(self.__filters, columnated)

        return raw_results

    def __im2col(self, input_matrix: np.array) -> np.array:
        # Parameters
        height, width, depth = input_matrix.shape
        column_extent = width - self.__F + 1
        row_extent = height - self.__F + 1

        # Get Starting block indices
        start_idx = np.arange(self.__F)[:, None] * width + np.arange(self.__F)

        # Get offsetted indices across the height and width of input array
        offset_idx = np.arange(row_extent)[:, None] * width + np.arange(column_extent)

        # Get all actual indices & index into input array for final output
        out = np.take(input_matrix, start_idx.ravel()[:, None] + offset_idx.ravel())

        return out

        # number_of_filters = (self.__input_width/self.__F)/self.__S
        # number_of_filters += (self.__input_height/self.__F)/self.__S
        #
        # number_of_filters /= self.__S
        #
        # size_of_filter = self.__F * self.__F * self.__depth
        #
        # result = np.zeros(size_of_filter, number_of_filters)
        #
        # row = 0
        # column = 0
        #
        # target_row = 0
        # target_column = 0
        #
        # while row+self.__F < self.__input_height and column+self.__F < self.__input_width:
        #     for row_offset in range(row, row + self.__F):
        #         for column_offset in range(column, column + self.__F):
        #             for stack in range(self.__depth):
        #                 result[target_row][target_column] = input_matrix[row_offset][column_offset][stack]
        #                 target_row += 1
        #
        #     target_column += 1
        #
        #     row += self.__S
        #     if row + self.__F >= self.__input_height:
        #         row = 0
        #         column += self.__S
        #
        # return result
