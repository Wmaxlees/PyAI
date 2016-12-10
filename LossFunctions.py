import numpy as np

class LossFunctions:
    @staticmethod
    def calculate_cross_entropy_loss(predictions: np.array, labels: np.array) -> float:
        first_term = - 1 / predictions.shape[0]

        predictions = np.transpose(predictions)
        predictions = np.log(predictions)
        predictions = np.multiply(predictions, labels)
        predictions = np.sum(predictions)

        return first_term * predictions
