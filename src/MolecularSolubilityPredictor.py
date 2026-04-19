# Utilities
import glob
import os
import time
from typing import Literal

# Dataset and data manipulation libs
import kagglehub
import numpy as np
import pandas as pd

# Plots
import matplotlib.pyplot as plt
import seaborn as sns

#RDKit 
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# ML lib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split

from PredictionModel import PredictionModel

class MolecularSolubilityPredictor:
    
    def __init__(
        self, 
        xTrain = None, 
        yTrain = None, 
        xTest = None, 
        yTest = None,
        # modelType: Literal['GNN', 'RandomForest', 'XGBoost'] = 'RandomForest'
    ):
        
        # These get populated as the pipeline runs
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xTest = xTest
        self.yTest = yTest
        self.yPred = None
        # self.modelType = modelType
    
    
    '''
    Trains a Random Forest model using the provided dataset and performs hyperparameter tuning using RandomizedSearchCV.
    '''
    def modelTraining(self):
        
        # 1. Initialise the Random Forest Regressor
        predictionModel = PredictionModel(modelType='RandomForest')
        
        # 2. Define the hyperparameters to tune and their respective ranges/values for RandomizedSearchCV
        paramGrid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        
        # 3. Perform RandomizedSearchCV to find the best hyperparameters
        randomSearch = RandomizedSearchCV(
            estimator=predictionModel,
            param_distributions=paramGrid,
            n_iter=50,
            cv=5,
            verbose=2,
            n_jobs=-1,
            random_state=42,
        )
        
        
        print("\n" + "=" * 60 + "\n\n")
        
        startTime = time.time()
        print("Begin fitting with RandomizedSearchCV...\n\n")
        
        randomSearch.fit(self.xTrain, self.yTrain)
        
        endTime = time.time()
        print(f"RandomizedSearchCV fitting completed in {endTime - startTime:.2f} seconds.\n\n")
        print(f"Best parameters found: {randomSearch.best_params_}\n")
        print(f"Best score achieved with Best Parameters: {randomSearch.best_score_}\n\n")
        
        self.yPred = randomSearch.predict(self.xTest)
        
        print("=" * 60 + "\n\n")

        return None
    
    
    def evaluateModelPerformance(self):
        
        # Define residual (yTest - yPred)
        residuals = self.yTest - self.yPred
        
        # Calculate evaluation metrics (MSE, RMSE, MAE, R2)
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        r2 = 1 - (np.sum(residuals ** 2) / np.sum((self.yTest - np.mean(self.yTest)) ** 2))
        
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared (R2): {r2} \n")
        
        print("\n" + "=" * 60 + "\n\n")

        
    def printResult(self, yTest, yPred, compoundIdTest):
        
        return None
    
    
    def runPipeline(self):
        
        self.modelTraining()
        self.evaluateModelPerformance()
        # self.printResult(self.yTest, self.yPred, self.compoundIdTest)
        
    