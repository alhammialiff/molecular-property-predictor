import time

import numpy as np

import deepchem as dc
from PredictionModel import PredictionModel
import matplotlib.pyplot as plt
import seaborn as sns

class GNNPredictor:
    
    def __init__(self, smilesTrain, smilesTest, smilesValidation, yTest, yValidation, trainDataset, testDataset, validationDataset):
        
        self.smilesTrain = smilesTrain
        self.trainDataset = trainDataset
        
        self.smilesTest = smilesTest
        self.testDataset = testDataset
        self.yTest = yTest
        
        self.smilesValidation = smilesValidation
        self.yValidation = yValidation
        self.validationDataset = validationDataset
        
        self.yPred = None
        self.model = None
        
        self.residuals = None
        
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
        
        # Initialise AttentiveFP model via factory builder
        self.model = PredictionModel(modelType='GNN')
        
        print("=" * 60 + "\n\n")
        
        print("Begin fitting AttentiveFP model...\n\n")
        
        # [Debug] Start time
        startTime = time.time()
        
        # Fit model
        self.model.fit(
            self.trainDataset, 
            nb_epoch=30,
            callbacks=dc.models.ValidationCallback(
                self.validationDataset, # Dataset 1 from DiskDataset
                interval=10, # Validate every 10 epochs
                metrics=[dc.metrics.Metric(dc.metrics.pearson_r2_score)] 
            )    
        )
        
        # [Debug] End time
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
        self.residuals = self.yTest - self.yPred
        
        # Calculate evaluation metrics (MSE, RMSE, MAE, R2)
        mse = np.mean(self.residuals ** 2)
        rmse = np.sqrt(mse) 
        mae = np.mean(np.abs(self.residuals))
        r2 = 1 - (np.sum(self.residuals ** 2) / np.sum((self.yTest - np.mean(self.yTest)) ** 2))
        
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
    
    
    '''
    Plot Residuals and Predictions
    '''
    def plotPerformance(self):
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.yPred, y=self.residuals, alpha=0.5, color='teal')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        
        plt.title('Residual Analysis: Predicted Molecular Solubility vs Errors', fontsize=14)
        plt.xlabel('Predicted Molecular Solubility', fontsize=12)
        plt.ylabel('Residuals / Error (mol/L)', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        
    '''
    Run pipeline
    '''
    def runPipeline(self):
        
        self.fitModel()
        self.evaluateModelPerformance()
        self.plotPerformance()