from enum import Enum

class Loss(Enum):
    MSE = 0 # Mean-squared-error : corresponds to optically obtainable loss.
    CCE = 1 # Categorical cross-entropy : standard computational loss for classification problems.