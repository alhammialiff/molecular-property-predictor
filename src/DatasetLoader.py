import glob
import os
import pandas as pd

# Import kagglehub
import kagglehub

class DatasetLoader:
    
    def __init__(self, datasetPath = None):
        
        self.dataset = None
        self.datasetPath = datasetPath
        
    def loadLocalDataset(self, datasetFileName):
        
        # Set the dataset path
        self.datasetPath = os.path.join(os.getcwd(), "datasets",datasetFileName)
        
        # 1. Check if the dataset file exists at the specified path
        if not os.path.isfile(self.datasetPath):
            raise FileNotFoundError(f"Dataset file not found at path: {self.datasetPath}")
        
        # 2. Load the dataset
        self.dataset = pd.read_csv(self.datasetPath)
        return self.dataset
    
    def loadKaggleDataset(self, kaggleDatasetName):
        
        dirPath = kagglehub.dataset_download(kaggleDatasetName)
        
        # 3. Find the CSV file in the extracted directory (assuming there's only one CSV file)
        csvFiles = glob.glob(os.path.join(dirPath, "*.csv"))
        
        if not csvFiles:
            raise FileNotFoundError(f"No CSV file found in the extracted dataset directory: {dirPath}")
        
        # 4. Load the CSV file into a pandas DataFrame
        self.dataset = pd.read_csv(csvFiles[0])
        
        return self.dataset
    
    def getDatasetInfo(self):
        
        if self.dataset is None:
            raise ValueError("No dataset loaded. Please load a dataset before describing it.")
        
        print("Dataset info")
        self.dataset.info()
        
        print("\n" + "=" * 60 + "\n")
        print("Number of rows:", self.dataset.shape[0])
        print("Number of columns:", self.dataset.shape[1], "\n")
        print("Statistical Summary of Numerical Columns")
        print(self.dataset.describe().T)