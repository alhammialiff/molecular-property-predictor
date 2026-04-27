import glob
import os
import pandas as pd
import deepchem as dc

# Import kagglehub
import kagglehub
from datasets import load_dataset

'''
This class loads up dataset from all over the place really.
Optimising retrieval of datasets can come later
'''
class DatasetLoader:
    
    def __init__(self, datasetPath = None):
        
        self.dataset = None
        self.datasetPath = datasetPath


    '''
    For datasets that are manually downloaded off the web
    '''  
    def loadLocalDataset(self, datasetFileName):
        
        # Set the dataset path
        self.datasetPath = os.path.join(os.getcwd(), "datasets",datasetFileName)
        
        # 1. Check if the dataset file exists at the specified path
        if not os.path.isfile(self.datasetPath):
            raise FileNotFoundError(f"Dataset file not found at path: {self.datasetPath}")
        
        # 2. Load the dataset
        self.dataset = pd.read_csv(self.datasetPath)
        return self.dataset
    

    '''
    For Kaggle datasets (using the kagglehub library to download and load the dataset from Kaggle's repository)
    '''    
    def loadKaggleDataset(self, kaggleDatasetName):
        
        dirPath = kagglehub.dataset_download(kaggleDatasetName)
        
        # 3. Find the CSV file in the extracted directory (assuming there's only one CSV file)
        csvFiles = glob.glob(os.path.join(dirPath, "*.csv"))
        
        if not csvFiles:
            raise FileNotFoundError(f"No CSV file found in the extracted dataset directory: {dirPath}")
        
        # 4. Load the CSV file into a pandas DataFrame
        self.dataset = pd.read_csv(csvFiles[0])
        
        return self.dataset
    

    '''
    For DeepChem's MoleculeNet built-in datasets (an API call to load the dataset from DeepChem's repository)
    '''    
    def loadLipophilicityDataset(self):
        
        tasks, datasets, transformers = dc.molnet.load_lipo()
        self.dataset = datasets
        
        self.getDatasetInfo()
        
        return datasets
    
    
    '''
    For Hugging Face datasets (an API call to load the dataset from Hugging Face's repository)
    '''
    def loadHuggingFaceDataset(self, datasetType = "default"):

        # For OpenADMET ExpansionRX Challenge Data
        self.dataset = load_dataset("openadmet/openadmet-expansionrx-challenge-data", datasetType)

        # Convert the Hugging Face dataset to pandas DataFrame format for easier handling in the rest of the pipeline
        self.dataset = {
            "train": self.dataset["train"].to_pandas(),
            "test": self.dataset["test"].to_pandas()
        }

        return self.dataset



    def getDatasetInfo(self):
        
        if self.dataset is None:
            raise ValueError("No dataset loaded. Please load a dataset before describing it.")
        
        print("\n" + "=" * 60 + "\n\n")


        
        # For deepchem's Lipo Dataset, it is actually a tuple of DiskDataset objects (train, test, validation)
        if isinstance(self.dataset, tuple) and all(isinstance(ds, dc.data.DiskDataset) for ds in self.dataset):
            
            print(f"Dataset type is DeepChem's DiskDataset (tuple of 3 DiskDatasets actually - Train, Valid, Test Sets):\n\n")
            
            for datasetIndex, diskDataset in enumerate(self.dataset):
                
                print(f"Dataset {datasetIndex} info:\n\n")
                print(f"Number of molecules: {diskDataset.X.shape[0]}")
                print(f"Input shape: {diskDataset.X.shape}")
                print(f"Target shape: {diskDataset.y.shape}")
                print(f"Sample IDs (SMILES): {diskDataset.ids[:5]}")
                print(f"Sample targets: {diskDataset.y[:5]}")
                print("\n" + "=" * 60 + "\n")
                
        elif isinstance(self.dataset, pd.DataFrame):
            
            print(f"Dataset type is pandas DataFrame:\n\n")
        
            print("Dataset info")
            self.dataset.info()
            
            print("Number of rows:", self.dataset.shape[0])
            print("Number of columns:", self.dataset.shape[1], "\n")
            print("Statistical Summary of Numerical Columns")
            print(self.dataset.describe().T)

        # For Hugging Face datasets, it is a dictionary of pandas DataFrames (e.g. with train/test splits)
        elif isinstance(self.dataset, dict) and all(isinstance(df, pd.DataFrame) for df in self.dataset.values()):
            
            print(f"Dataset type is a dictionary of pandas DataFrames (e.g. for Hugging Face datasets with train/test splits):\n\n")
            
            for splitName, df in self.dataset.items():
                print(f"Split: {splitName}")
                print("Dataset info")
                df.info()
                
                print("Number of rows:", df.shape[0])
                print("Number of columns:", df.shape[1], "\n")
                print("Statistical Summary of Numerical Columns")
                print(df.describe().T)
                print("\n" + "=" * 60 + "\n")