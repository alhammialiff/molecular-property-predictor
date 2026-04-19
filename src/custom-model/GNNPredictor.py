import numpy as np
import deepchem as dc
from PredictionModel import PredictionModel

class GNNPredictor:
    
    def __init__(self, xTrain, yTrain, xTest, yTest):
        
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xTest = xTest
        self.yTest = yTest
        self.yPred = None