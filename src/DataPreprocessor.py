import pandas as pd
import numpy as np

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Scikit-learn
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    
    def __init__(self, rawData):
        
        # Input - We need the dataset (rawData) to perform the preprocessing steps, so we pass it in the constructor and store it as an instance variable
        self.rawData = rawData
        self.refinedData = None
        
        # Result - Dataset splits
        self.xTrain = None
        self.yTrain = None
        self.xTest = None
        self.yTest = None
        self.compoundIdTrain = None
        self.compoundIdTest = None
    
        
    '''
    Performs structural analysis and investigate missing values in dataset
    '''
    def performStructuralAnalysisAndMissingValues(self, rawData):
        
        # [Guard Clause]
        if rawData is None:
            raise ValueError("No dataset provided. Please provide a dataset to perform structural analysis and investigate missing values.")
        
        # 1. We always want to start with a peek at the data
        print("\n" + "=" * 60 + "\n\n")

        print("Dataset info \n\n")
        self.rawData.info()
        
        print("\n" + "=" * 60 + "\n\n")
        
        # 2. Check for missing values in each column. Output each index (feature) and the sum of its missing values
        print("Analysing missing values in each column\n\n")
        missingValue = rawData.isnull().sum()
        
        if missingValue.sum() > 0:
            
            print(missingValue)
            
            # [Placeholder] This is where we would perform imputation or drop rows/columns with missing values, depending on the context and amount of missing data
            #####
            
        else:
            
            print("There are no missing values in the dataset.")
            
        # Get statistical summary of numerical columns (min, max, mean, std, etc.)
        print("Statistical Summary of Numerical Columns")
        print(self.rawData.describe().T)
        
    
    '''
    We want to check for duplicate entries in the dataset, as duplicates can bias our model and lead to overfitting. If we find duplicates, we will need to decide whether to remove them or keep them based on the context of the data and the amount of duplication.
    '''
    def performDuplicateAnalysis(self, featureName):
        
        # [Test Var]
        featureName = "Compound ID"
        
        # 1. Extract Duplicate Entries - We can use the pandas duplicated() method 
        #   to identify duplicate rows based on a specific feature (column). 
        #   The keep=False argument marks all duplicates as True, so we can extract all duplicate entries.       
        duplicates = self.rawData[self.rawData[featureName].duplicated(keep=False)]

        print("\n" + "=" * 60 + "\n\n")
        print(f"Duplicate Analysis based on feature: '{featureName}'\n\n")

        if(len(duplicates) > 0):
            
            print(f"Found {len(duplicates)} duplicate entries based on the feature '{featureName}'.")
            print(duplicates)
            
            # 1. If multiple Compound IDs have the same SMILES string, we can group them together and take the mean of their measured log(solubility:mol/L) values to create a single entry for each unique compound. This way, we can retain all the information from the duplicates while eliminating redundancy.
            fixedDuplicates = duplicates.groupby(
                ["Compound ID", "SMILES"], as_index=False                   
            )["measured log(solubility:mol/L)"].mean()
            
            # 2. Remove the original duplicate entries from the rawData 
            #   and concatenate the fixed duplicates back to the dataset
            # Note: The tilde inverts the boolean mask, so we keep only the unique entries (those that are not duplicated) in the rawData, and then we concatenate the fixed duplicates back to the dataset to create the refinedData
            self.refinedData = self.rawData[~self.rawData["Compound ID"].duplicated(keep=False)]
            self.refinedData = pd.concat([self.refinedData, fixedDuplicates], ignore_index = True)
            
            # Debug
            print(f"Fixed {len(duplicates)} duplicate rows → merged into {len(fixedDuplicates)} rows")
            print("Cleaned dataset shape:", self.refinedData.shape)

            
        
    '''
    We want to see the distribution of the categorical variables in the dataset, as this can inform us about potential issues such as class imbalance, and also guide us in choosing appropriate encoding techniques for these variables.
    '''
    def investigateCategoricalVariableDistribution(self):
        
        # Select categorical columns
        categoricalColumns = self.refinedData.select_dtypes(include=["object"]).columns
        
        print("\n" + "=" * 60 + "\n\n")
        print("Distribution of Categorical Variables\n\n")
        print("Categorical Columns:")
        print(categoricalColumns.tolist())
        
        # Display the top 5 most frequent values for some key categorical columns (e.g., "Compound ID" and "SMILES") to understand the distribution of these variables. This can help us identify any potential issues such as class imbalance or dominant categories that may need to be addressed during preprocessing.
        keyCategoricalColumns = ["Compound ID", "SMILES"]
        
        for col in keyCategoricalColumns:
            
            print(f"\nTop 5 most frequent values for '{col}':")
            print(self.refinedData[col].value_counts().head(5), "\n")
                    
                    
    '''
    Converts SMILES strings into numerical features that can be used as input for machine learning models. This typically involves using cheminformatics libraries like RDKit to generate molecular descriptors or fingerprints from the SMILES strings, which capture the structural and chemical properties of the molecules. The resulting features can then be used to train predictive models for tasks such as regression or classification based on the molecular properties.
    '''
    def featuriseSmiles(self, smiles):
        
        # Convert SMILES into RDKit molecule objects
        molecule = Chem.MolFromSmiles(smiles)
        
        # Create a Morgan fingerprint generator with:
        # - radius = 3 (captures atom neighborhoods up to 3 bonds away)
        # - fpSize = 2048 (output vector length)
        generator = GetMorganGenerator(radius=3, fpSize=2048)
        
        fingerprint = list(generator.GetFingerprintAsNumPy(molecule))
        
        # [Featurisation] 
        # Append physico-chemical descriptors to the fingerprint vector to
        # capture additional molecular properties that may be relevant for predicting solubility. These descriptors include:
        descriptors = [
            Descriptors.MolWt(molecule), # Molecular weight
            Descriptors.MolLogP(molecule), # Lipophilicity (known to correlate with solubility)
            Descriptors.NumHDonors(molecule), # Hydrogen bond donors
            Descriptors.NumHAcceptors(molecule), # Hydrogen bond acceptors
            Descriptors.TPSA(molecule), # Topological polar surface area
            Descriptors.NumRotatableBonds(molecule) # Molecular flexibility
        ]

        return fingerprint + descriptors

    
    '''
    Performs an 80-20 train-test split on the dataset, ensuring that the distribution of the target variable (measured log(solubility:mol/L)) is maintained in both sets. This is important to ensure that our model can generalize well to unseen data and that the performance metrics we calculate on the test set are representative of how the model will perform in real-world scenarios. The method also returns the compound IDs for both the training and test sets, which can be useful for tracking and analyzing specific compounds during model evaluation.
    '''
    def performTrainTestSplit(self, featurisedData):
        
        # X
        X = self.refinedData["SMILES"].apply(self.featuriseSmiles).tolist()
        
        # Y
        y = self.refinedData["measured log(solubility:mol/L)"]
        
        # Train-test split (80-20)
        self.xTrain, self.xTest, self.yTrain, self.yTest, self.compoundIdTrain, self.compoundIdTest = train_test_split(X, y, self.refinedData["Compound ID"], test_size=0.2, random_state=42)
        
    def getTrainTestSplits(self):
        
        return self.xTrain, self.yTrain, self.xTest, self.yTest, self.compoundIdTrain, self.compoundIdTest    
    
    def run(self):
        
        # 1. Structural Analysis and Missing Values
        self.performStructuralAnalysisAndMissingValues(self.rawData)
        
        # 2. Duplicate Analysis
        self.performDuplicateAnalysis("Compound ID")
        
        # 3. Categorical Variable Distribution
        self.investigateCategoricalVariableDistribution()
        
        # 4. Train-Test Split
        self.performTrainTestSplit(self.refinedData)