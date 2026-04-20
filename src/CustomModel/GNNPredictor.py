import time

import numpy as np

import deepchem as dc
from PredictionModel import PredictionModel

class GNNPredictor:
    
    def __init__(self, smilesTrain, smilesTest, yTest, trainDataset, testDataset):
        
        self.smilesTrain = smilesTrain
        self.trainDataset = trainDataset
        self.smilesTest = smilesTest
        self.testDataset = testDataset
        self.yTest = yTest
        self.yPred = None
        self.model = None
        
    '''
    PROCESS 2: Message Passing (inside AttentiveFP)
    During training, the GNN performs message passing:
        1. Each atom collects feature messages from its bonded neighbors
        2. Aggregates them (sum/mean)
        3. Updates its own feature vector
    This repeats for `num_layers` iterations, expanding the atom's
    "receptive field" — after 3 layers, each atom knows about atoms
    up to 3 bonds away.
    '''    
    def fitModel(self):
        
        self.model = PredictionModel(modelType='GNN')
        
        print("=" * 60 + "\n\n")
        
        print("Begin fitting AttentiveFP model...\n\n")
        
        startTime = time.time()
        self.model.fit(self.trainDataset, nb_epoch=50)
        endTime = time.time()
        
        print(f"AttentiveFP model fitting completed in {endTime - startTime:.2f} seconds.\n\n")
        
    
    '''
    PROCESS 3: Readout & Prediction
    After message passing, a readout function aggregates all atom
    feature vectors into a single molecular-level embedding vector.
    This vector is then passed through a fully connected layer
    to predict the solubility value.
    '''
    def evaluateModelPerformance(self):
        
        # Predict on test set
        self.yPred = self.model.predict(self.testDataset).flatten()
        
        # Define residual (yTest - yPred)
        residuals = self.yTest - self.yPred
        
        # Calculate evaluation metrics (MSE, RMSE, MAE, R2)
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse) 
        mae = np.mean(np.abs(residuals))
        r2 = 1 - (np.sum(residuals ** 2) / np.sum((self.yTest - np.mean(self.yTest)) ** 2))
        
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
        
    def runPipeline(self):
        
        self.fitModel()
        self.evaluateModelPerformance()