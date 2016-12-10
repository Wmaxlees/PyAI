import numpy as np
import ConvolutionalLayer
import MaxPoolingLayer
import ActivationFunctions
import FullyConnectedLayer
import LossFunctions
import MatrixToVector

myCNN = ConvolutionalLayer.ConvolutionalLayer(10, 8, 4, 0, 1024, 768)
myMaxPoolingLayer = MaxPoolingLayer.MaxPoolingLayer(2, 2, 191, 255, 10)
myCNN2 = ConvolutionalLayer.ConvolutionalLayer(8, 5, 1, 0, 95, 127, 10)
myMaxPoolingLayer2 = MaxPoolingLayer.MaxPoolingLayer(2, 2, 123, 91, 8)
myCNN3 = ConvolutionalLayer.ConvolutionalLayer(6, 3, 1, 0, 61, 45, 8)
myMaxPoolingLayer3 = MaxPoolingLayer.MaxPoolingLayer(2, 2, 43, 59, 6)
myMatrixToVector = MatrixToVector.MatrixToVector()
myHiddenLayer = FullyConnectedLayer.FullyConnectedLayer(3654, 1000)
myOutputLayer = FullyConnectedLayer.FullyConnectedLayer(1000, 5)

label = np.array([1, 0, 0, 0, 0])

for i in range(10000):
    result = ActivationFunctions.ActivationFunctions.apply_logistic_sigmoid(np.random.rand(1024, 768, 3))
    result = myCNN.apply(result)
    result = myMaxPoolingLayer.apply(result)
    result = ActivationFunctions.ActivationFunctions.apply_logistic_sigmoid(result)
    result = myCNN2.apply(result)
    result = myMaxPoolingLayer2.apply(result)
    result = ActivationFunctions.ActivationFunctions.apply_logistic_sigmoid(result)
    result = myCNN3.apply(result)
    result = myMaxPoolingLayer3.apply(result)
    result = ActivationFunctions.ActivationFunctions.apply_logistic_sigmoid(result)
    result = myMatrixToVector.apply(result)[:, None]
    result = myHiddenLayer.apply(result)
    result = ActivationFunctions.ActivationFunctions.apply_logistic_sigmoid(result)
    result = myOutputLayer.apply(result)
    result = ActivationFunctions.ActivationFunctions.apply_softmax(result)

    loss = LossFunctions.LossFunctions.calculate_cross_entropy_loss(result, np.array([label]))
    if i % 10 == 0:
        print(loss)
        print(str(label) + ' ?= ' + str(result))

    backprop = np.multiply(loss, label)[:, None]

    result = myOutputLayer.backprop(backprop, 0.001)
    result = myHiddenLayer.backprop(result, 0.001)
    result = myMatrixToVector.undo(result)
    result = myMaxPoolingLayer3.backprop(result)




