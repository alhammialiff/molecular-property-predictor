import os
import shutil
import time

import numpy as np

import deepchem as dc
import test
from CustomModels.GNNHyperparamGrid import GNNHyperparamGrid
from PredictionModel import PredictionModel
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import MolToInchiKey, MolFromSmiles

from Utilities.TextColorGenerator import TextColorGenerator
from Utilities.ReportDirectoryGenerator import DirectoryGenerator
from Utilities.SiUnitGenerator import SiUnitGenerator
from colorama import Fore, Style, init

init()

class GNNPredictor:
    
    def __init__(self, smilesTrain, smilesTest, smilesValidation, yTest, yValidation, trainDataset, testDataset, validationDataset, modelName=None, epoch = 30, admetScreeningType=None):
        
        # [Model and Dataset Info]
        self.admetScreeningType = admetScreeningType
        self.model = None # Current model instance being trained
        self.bestModel = None # Best model instance based on validation R2 across all epochs and hyperparameter combinations
        self.modelName = modelName  # Update model name to reflect the current model being used

        # [Directories]
        self.directories = {}

        # [Training Set]
        self.epoch = epoch
        self.globalTrainingId = None
        self.hyperparameterId = None
        self.smilesTrain = smilesTrain
        self.trainDataset = trainDataset
        
        # [Test Set]
        self.smilesTest = smilesTest
        self.testDataset = testDataset
        self.yTest = yTest
        
        # [Validation Set]
        self.smilesValidation = smilesValidation
        self.yValidation = yValidation
        self.validationDataset = validationDataset
        self.bestValidationR2 = 0
        self.validationPatience = 15
        self.bestHyperparameters = {}
        
        # [Predictions]
        self.yPred = None
        
        # [Evaluation Metrics]
        self.trainingDuration = None
        self.residuals = None
        self.mae = None
        self.mse = None
        self.rmse = None
        self.r2 = None

        # [Utilities]
        self.textColor = TextColorGenerator().getColour(self.modelName)
        

        
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
        
        # [Initialise] AttentiveFP model via factory builder
        self.model = PredictionModel(modelType='GNN', modelName=self.modelName, hyperparameters={})
        
        # [Initialise] directory generator to get checkpoint and summary directories based on model name
        self.directories = DirectoryGenerator(self.modelName).getDirectories()
        validationSummariesDir = self.directories['validationSummariesDir']
        checkpointDir = self.directories['bestModelDir']

        # [Initialise] hyperparameter grid specific to the model being trained
        hyperparamGrid = GNNHyperparamGrid(self.modelName).getHyperparamGrid()

        # [Initialize] Patience counter for early stopping during epoch training
        patienceCounter = 0
        patience = self.validationPatience

        # Debug: check for molecule leakage between train and test
        # This is crucial to ensure that our model's performance metrics are valid and 
        # not artificially inflated by having the same molecules in both sets.
        print(Fore.CYAN + "=" * 60 + "\n\n" + Style.RESET_ALL)
        trainInchiKeys = set(MolToInchiKey(MolFromSmiles(s)) for s in self.smilesTrain if MolFromSmiles(s))
        testInchiKeys  = set(MolToInchiKey(MolFromSmiles(s)) for s in self.smilesTest  if MolFromSmiles(s))
        overlap = trainInchiKeys & testInchiKeys
        print(Fore.YELLOW + f"Overlapping molecules between train and test: {len(overlap)}\n\n" + Style.RESET_ALL)
        
        print(Fore.CYAN + "=" * 60 + "\n\n" + Style.RESET_ALL)
        
        print(self.textColor + f"Begin fitting {self.modelName} model...\n\n" + Style.RESET_ALL)
        
        # [Debug] Start time
        startTime = time.time()   
                
        # Define global training ID with local timestamp for uniqueness across runs
        self.globalTrainingId = time.strftime("%Y%m%d-%H%M%S")  # Using timestamp for uniqueness
        
        # [[-Grid Search Loop-]] We iterate through the hyperparameter combinations, training a model for each and evaluating on the validation set.
        for idx, hyperparams in enumerate(hyperparamGrid):

            # Re-initialize model for each hyperparameter combination to ensure a fresh start
            self.model = PredictionModel(modelType='GNN', modelName=self.modelName, hyperparameters=hyperparams)
            
            # Reset Training Sequence Best Valid R2 score everytime we start a new 
            # training sequence with a different hyperparameter combination.
            trainingSequenceBestValidationR2 = 0
            
            # For tracking which training sequence we're on
            self.hyperparameterId = idx + 1 
            
            print(self.textColor + f"[{self.modelName}] Grid Search Iteration {idx + 1}/{len(hyperparamGrid)}: Testing hyperparameters: {hyperparams}\n" + Style.RESET_ALL)
            
            
            # [[-Training Sequence-]] We train for a maximum of 30 epochs, but with early stopping based on validation R2.
            for epoch in range(30):
                
                # [LR Decay] Decay learning rate every 10 epochs to help convergence
                if epoch > 0 and epoch % 10 == 0:
                    currentLR = self.model.learning_rate
                    self.model.learning_rate = currentLR * 0.5
                    print(self.textColor + f"[{self.modelName}] Epoch {epoch}: Learning rate decayed to {self.model.learning_rate:.6f}" + Style.RESET_ALL)
                
                # [Fit model]
                self.model.fit(self.trainDataset,nb_epoch=1)
                
                # [Evaluate] on validation set
                metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
                validScore = self.model.evaluate(self.validationDataset, [metric])
                currentEpochValidationR2 = validScore['pearson_r2_score']
                
                print(self.textColor + f"[{self.modelName}] Validation R2 after epoch {epoch + 1}: {currentEpochValidationR2:.4f}" + Style.RESET_ALL)
                
                # [Early stopping logic]
                # If R2 improves, reset patience counter. If not, increment it.
                if currentEpochValidationR2 > trainingSequenceBestValidationR2:
                    
                    # [Local Update] the best validation R2 for this training sequence 
                    # (i.e this hyperparameter combination)
                    trainingSequenceBestValidationR2 = currentEpochValidationR2
                    
                    # [Global Update] If this is the best validation R2 we've seen across all epochs 
                    # and hyperparameter combinations, update the global best 
                    # and save the model checkpoint.
                    if trainingSequenceBestValidationR2 > self.bestValidationR2:
                        
                        # Update global best validation R2 and corresponding hyperparameters
                        self.bestValidationR2 = trainingSequenceBestValidationR2
                        self.bestHyperparameters = hyperparams

                        # Save the best model instance
                        self.bestModel = self.model  
                        
                        # [Validation Summaries] Write validation performance and hyperparameters to a summary file for 
                        # this training sequence
                        with open(os.path.join(validationSummariesDir, f'validation_summary.txt'), 'a') as f:
                            
                            f.write(f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
                            f.write(f"Global Training ID: {self.globalTrainingId}\n")
                            f.write(f"Hyperparameter Combination ID: {self.hyperparameterId}\n")
                            f.write(f"Grid Search Iteration {idx + 1}/{len(hyperparamGrid)}\n")
                            f.write(f"Hyperparameters: {hyperparams}\n")
                            f.write(f"Best Validation R2: {trainingSequenceBestValidationR2:.4f}\n\n")
                            
                        # [Keeping only one Checkpoint at any time]
                        # Clear old checkpoints — keep only the global best
                        if os.path.exists(checkpointDir):
                            
                            shutil.rmtree(checkpointDir)
                            
                        # [Edge case - first best during run] Ensure checkpoint directory exists
                        os.makedirs(checkpointDir, exist_ok=True)
                        
                        # Save the best model checkpoint
                        self.bestModel.save_checkpoint(model_dir=checkpointDir)

                    # We want patienceCounter to reset at the sequence level 
                    # (not just epoch level) because we want to give each hyperparameter combination a fair chance to train for multiple epochs before we decide to stop it. 
                    # If we reset patienceCounter at the epoch level, we might end up giving some hyperparameter combinations more training time than others, which could bias our results.
                    patienceCounter = 0
                    
                else:
                    
                    patienceCounter += 1
                    
                    
                # If we've grown patient (hitting the patience threshold), 
                # break and conclude training
                if patienceCounter >= patience:
                    
                    print(self.textColor + f"[{self.modelName}] Early stopping triggered after {epoch + 1} epochs with best R2: {self.bestValidationR2:.4f}" + Style.RESET_ALL)
                    
                    # Restore the best model checkpoint before breaking, ensuring we 
                    # evaluate the best version of the model on the test set.
                    self.bestModel.restore(model_dir=checkpointDir)
  
                    break



            
        # [Debug] End time
        endTime = time.time()
        self.trainingDuration = endTime - startTime

        # After all hyperparameter combinations and epochs have been evaluated, ensure we have the 
        # best model loaded for final evaluation on the test set.
        self.bestModel.restore(model_dir=checkpointDir)
        
        print(self.textColor + f"[{self.modelName}] Model fitting completed in {self.trainingDuration:.2f} seconds.\n\n" + Style.RESET_ALL)
        
    
    '''
    PROCESS 3: Readout & Prediction
    After message passing, a readout function aggregates all atom
    feature vectors into a single molecular-level embedding vector.
    This vector is then passed through a fully connected layer
    to predict the solubility value.
    '''
    def evaluateModelPerformance(self):

        # [Test Summaries Dir]
        testSummariesDir = self.directories['testSummariesDir']
        
        # Predict on test set
        self.yPred = self.bestModel.predict(self.testDataset).flatten()
        
        # Define residual (yTest - yPred)
        self.residuals = self.yTest - self.yPred
        
        # Calculate evaluation metrics (MSE, RMSE, MAE, R2)
        self.mse = np.mean(self.residuals ** 2)
        self.rmse = np.sqrt(self.mse) 
        self.mae = np.mean(np.abs(self.residuals))
        self.r2 = 1 - (np.sum(self.residuals ** 2) / np.sum((self.yTest - np.mean(self.yTest)) ** 2))
        
        
        print(self.textColor + f"MSE:  {self.mse:.4f}" + Style.RESET_ALL)
        print(self.textColor + f"RMSE: {self.rmse:.4f}" + Style.RESET_ALL)
        print(self.textColor + f"MAE:  {self.mae:.4f}" + Style.RESET_ALL)
        print(self.textColor + f"R²:   {self.r2:.4f}" + Style.RESET_ALL)
        

        # Export summary, timestamp (local timezone), model hyper parameters, performance into .txt
        with open(os.path.join(testSummariesDir, f'test_summary.txt'), 'a') as f:

            f.write(f"Model Performance Summary - {self.modelName}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n\n")
            f.write(f"Training Duration: {self.trainingDuration:.2f} seconds\n\n")
            f.write(f"Model Hyperparameters:\n")

            for hyperparam, value in self.bestHyperparameters.items():

                f.write(f" - {hyperparam}: {value}\n")

            f.write(f"Best Validation R²: {self.bestValidationR2:.4f}\n\n")
            f.write(f"Evaluation Metrics on Test Set:\n")
            f.write(f"MSE:  {self.mse:.4f}\n")
            f.write(f"RMSE: {self.rmse:.4f}\n")
            f.write(f"MAE:  {self.mae:.4f}\n")
            f.write(f"R²:   {self.r2:.4f}\n")
            f.write(f"=" * 60 + "\n\n")

    
    '''
    Plot Residuals and Predictions
    '''
    def plotPerformance(self):
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.yPred, y=self.residuals, alpha=0.5, color='teal')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        
        plt.title(f'Residual Analysis: {self.modelName} - ({self.admetScreeningType.capitalize()} vs Errors)', fontsize=14)
        plt.xlabel(f'Predicted Molecular {self.admetScreeningType.capitalize()}', fontsize=12)
        plt.ylabel(f'Residuals / Error ({SiUnitGenerator(self.admetScreeningType).generate()})', fontsize=11)
        
        # Annotate with performance metrics
        plt.annotate(
            f'MSE:  {self.mse:.4f}\nRMSE: {self.rmse:.4f}\nMAE:  {self.mae:.4f}\nR²:   {self.r2:.4f}',
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        plt.show()
        
        
    '''
    Run pipeline
    '''
    def runPipeline(self):
        
        self.fitModel()
        self.evaluateModelPerformance()
        self.plotPerformance()