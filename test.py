import numpy as np
import ConvolutionalLayer
import MaxPoolingLayer

# input_array = np.array(
#     [
#         [
#             [1, 2, 3], [4, 5, 6], [7, 8, 9]
#         ],
#         [
#             [10, 11, 12], [13, 14, 15], [16, 17, 18]
#         ],
#         [
#             [19, 20, 21], [22, 23, 24], [25, 26, 27]
#         ]
#     ]
# )

# input_array = np.random.rand(5, 10, 3)
#
# filter_width = 3
# filter_height = 3
# step_size = 1
#
# input_width, input_height, depth = input_array.shape
#
# frames_per_width = int((input_width - filter_width + 1) / step_size)
# frames_per_height = int((input_height - filter_height + 1) / step_size)
#
# number_of_windows = frames_per_width * frames_per_height
#
# consecutive_numbers = np.arange(filter_width * depth)
# row = consecutive_numbers
# for i in range(frames_per_width - 1):
#     row = np.append(row, consecutive_numbers+input_width*(i+1)*depth)
#
# print(row)
#
# consecutive_numbers = np.arange(frames_per_width) * depth
# column = consecutive_numbers
# for i in range(frames_per_height - 1):
#     column = np.append(column, consecutive_numbers+(i+1)*depth*input_width)[:, None]
#
# selection_array = row+column
#
# print(selection_array)

# print(np.take(input_array, selection_array))

myCNN = ConvolutionalLayer.ConvolutionalLayer(10, 8, 4, 0, 1024, 768)
myMaxPoolingLayer = MaxPoolingLayer.MaxPoolingLayer(2, 2, 191, 255, 10)
myCNN2 = ConvolutionalLayer.ConvolutionalLayer(8, 5, 1, 0, 95, 127, 10)
myMaxPoolingLayer2 = MaxPoolingLayer.MaxPoolingLayer(2, 2, 123, 91, 8)

result = myCNN.apply(np.random.rand(1024, 768, 3))
result = myMaxPoolingLayer.apply(result)
result = myCNN2.apply(result)
result = myMaxPoolingLayer2.apply(result)

print(result.shape)



# [01 02 03] [04 05 06] [07 08 09]
# [10 11 12] [13 14 15] [16 17 18]
# [19 20 21] [22 23 24] [25 26 27]
#
# 01 02 03 04 05 06 10 11 12 13 14 15
# 04 05 06 07 08 09 13 14 15 16 17 18
# 10 11 12 13 14 15 19 20 21 22 23 24
# 13 14 15 16 17 18 22 23 24 25 26 27


# [000 001 002] [003 004 005] [006 007 008] [009 010 011] [012 013 014]
# [015 016 017] [018 019 020] [021 022 023] [024 025 026] [027 028 029]
# [030 031 032] [033 034 035] [036 037 038] [039 040 041] [042 043 044]
# [045 046 047] [048 049 050] [051 052 053] [054 055 056] [057 058 059]
# [060 061 062] [063 064 065] [066 067 068] [069 070 071] [072 073 074]
# [075 076 077] [078 079 080] [081 082 083] [084 085 086] [087 088 089]
# [090 091 092] [093 094 095] [096 097 098] [099 100 101] [102 103 104]
# [105 106 107] [108 109 110] [111 112 113] [114 115 116] [117 118 119]
# [120 121 122] [123 124 125] [126 127 128] [129 130 131] [132 133 134]
# [135 136 137] [138 139 140] [141 142 143] [144 145 146] [147 148 149]

# 00 01 02 03 04 05 15 16 17 18 19 20
# 03 04 05 06 06 08 18 19 20 21 22 23
# 06 07 08 09 10 11 21 22 23 24 25 26
# 09 10 11 12 13 14 24 25 26 27 28 29
# 15 16 17 18 19 20 30 31 32 33 34 35

# 120 121 122 123 124 135 136 137 138 139 140

# 129 130 131 132 133 134 144 145 146 147 148
