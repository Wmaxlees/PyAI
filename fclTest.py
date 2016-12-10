import FullyConnectedLayer
import numpy as np
import ActivationFunctions
import LossFunctions

hidden_layer = FullyConnectedLayer.FullyConnectedLayer(10, 100)
output_layer = FullyConnectedLayer.FullyConnectedLayer(100, 5)

for i in range(1000000):
    input_vector = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    label = np.array([0, 1, 0, 0, 0])

    result = hidden_layer.apply(input_vector[:, None])
    result = ActivationFunctions.ActivationFunctions.apply_logistic_sigmoid(result)
    result = output_layer.apply(result)

    loss = LossFunctions.LossFunctions.calculate_cross_entropy_loss(result, np.array([label]))
    if i % 10000 == 0:
        print(loss)
        print(str(label) + ' ?= ' + str(result))

    backprop = np.multiply(loss, label)[:, None]
    result = output_layer.backprop(backprop, 0.001)
    result = hidden_layer.backprop(result, 0.001)
