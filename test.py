import numpy as np
import ConvolutionalLayer
import MaxPoolingLayer
import ActivationFunctions
import FullyConnectedLayer

myCNN = ConvolutionalLayer.ConvolutionalLayer(10, 8, 4, 0, 1024, 768)
myMaxPoolingLayer = MaxPoolingLayer.MaxPoolingLayer(2, 2, 191, 255, 10)
myCNN2 = ConvolutionalLayer.ConvolutionalLayer(8, 5, 1, 0, 95, 127, 10)
myMaxPoolingLayer2 = MaxPoolingLayer.MaxPoolingLayer(2, 2, 123, 91, 8)
myHiddenLayer = FullyConnectedLayer.FullyConnectedLayer(21960, 35000)

result = ActivationFunctions.ActivationFunctions.apply_logistic_sigmoid(np.random.rand(1024, 768, 3))
result = myCNN.apply(result)
result = myMaxPoolingLayer.apply(result)
result = ActivationFunctions.ActivationFunctions.apply_logistic_sigmoid(result)
result = myCNN2.apply(result)
result = myMaxPoolingLayer2.apply(result)
result = ActivationFunctions.ActivationFunctions.apply_logistic_sigmoid(result)
result = result.flatten()
print(result.shape)

# print(ActivationFunctions.ActivationFunctions.apply_softmax(result))

