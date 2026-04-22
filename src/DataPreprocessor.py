import os
import pandas as pd
import numpy as np
import deepchem as dc
import pathlib

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MolToSmiles, MolFromSmiles
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Scikit-learn
from sklearn.model_selection import train_test_split



class DataPreprocessor:
    
    def __init__(self, rawData, modelType=None, targetProperty='solubility'):
        
        # Input - We need the dataset (rawData) to perform the preprocessing steps, so we pass it in the constructor and store it as an instance variable
        self.rawData = rawData
        self.refinedData = None
        self.targetProperty = targetProperty
        
        # Map property to column name - This allows us to easily switch between different target properties (e.g., solubility, distribution) by simply changing the targetProperty variable, without having to modify the rest of the code that references the target column. It also makes the code more flexible and adaptable to different datasets that 
        # may have different column names for the same property.
        self.targetColumn = {
            'solubility': 'measured log(solubility:mol/L)',
            'distribution': 'exp'   # logP column in Lipophilicity dataset
        }.get(targetProperty)
                
        self.modelType = modelType
        
        # Result - Dataset splits
        self.xTrain = None
        self.yTrain = None
        self.xValidation = None
        self.yValidation = None
        self.xTest = None
        self.yTest = None
        self.compoundIdTrain = None
        self.compoundIdTest = None
        
        # For For AttentiveFP GNN featurisation
        self.smilesTrainAfp = None
        self.smilesValidationAfp = None
        self.smilesTestAfp = None
        
        self.trainDatasetAfp = None
        self.validationDatasetAfp = None
        self.testDatasetAfp = None
        
        # Flags - to deduce what sort of dataset we are dealing with
        self.isDeepChemDataset = isinstance(self.rawData, tuple) and all(isinstance(ds, dc.data.DiskDataset) for ds in self.rawData)
        self.isPandasDataFrame = isinstance(self.rawData, pd.DataFrame)
    
        
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
    Converts SMILES strings into numerical features that can be used as input for 
    machine learning models. This typically involves using cheminformatics libraries 
    like RDKit to generate molecular descriptors or fingerprints from the SMILES 
    strings, which capture the structural and chemical properties of the molecules. 
    The resulting features can then be used to train predictive models for tasks 
    such as regression or classification based on the molecular properties.
    '''
    def featuriseSmilesForML(self, smiles):
        
        
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
    PROCESS 1: Featurisation
    
    This is for non-DeepChem Datasets (e.g., Kaggle Delaney Dataset) where we have a 
    pandas DataFrame and we need to convert the SMILES strings into numerical features that 
    can be used as input for machine learning models.
    
    Unlike Random Forest which uses Morgan fingerprints (flat bit vectors),
    GNNs require the molecule to be represented as a GRAPH:
        - Nodes = atoms (with features like atomic number, charge, etc.)
        - Edges = bonds (with features like bond type, aromaticity, etc.)
    DeepChem's MolGraphConvFeaturizer handles this automatically from SMILES.
    
    (Process 2 & 3 in GNNPredictor.py)
    '''
    def featuriseAndSplitDataFrameSmilesForGNN(self, smiles):
        
        # Filter out invalid/single-atom molecules before splitting
        def isValidSmiles(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                return mol is not None and mol.GetNumAtoms() > 1
            except:
                return False
            
        validMask = self.refinedData["SMILES"].apply(isValidSmiles)
        filteredSmiles = self.refinedData["SMILES"][validMask]
        filteredY = self.refinedData["measured log(solubility:mol/L)"][validMask]
        
        
        # Get SMILES for raw data   
        # Step 1: Split SMILES into train/test
        self.smilesTrainAfp, self.smilesTestAfp, self.yTrain, self.yTest = train_test_split(
            filteredSmiles.tolist(), filteredY.values, test_size=0.2, random_state=42
        )
        # self.smilesTrainAfp, self.smilesTestAfp, self.yTrain, self.yTest = train_test_split(
        #     self.rawData["SMILES"], self.rawData["measured log(solubility:mol/L)"], test_size=0.2, random_state=42
        # )
        
        # Create a MolGraphConvFeaturizer from DeepChem, which can convert the molecule 
        # into a graph representation suitable for GNNs. 
        # The use_edges=True argument indicates that we want to 
        # include edge features (e.g., bond types) in the graph representation, 
        # which can provide additional information for the GNN to learn from.
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        
        # Featurize the SMILES strings for both the training and test sets to create graph representations that can be used as input for a GNN model. This step is crucial for preparing the data in a format that the GNN can process effectively.
        xTrain = featurizer.featurize(self.smilesTrainAfp)
        xTest = featurizer.featurize(self.smilesTestAfp)
        
        # Force dtype=object to preserve GraphData objects
        xTrain = np.array(xTrain, dtype=object)
        xTest = np.array(xTest, dtype=object)
        
        # Wrap into DeepChem Dataset objects
        self.trainDatasetAfp = dc.data.NumpyDataset(X=xTrain, y=self.yTrain)
        self.testDatasetAfp = dc.data.NumpyDataset(X=xTest, y=self.yTest)
        
        print(f"GNN Train size: {len(self.smilesTrainAfp)}")
        print(f"GNN Test size: {len(self.smilesTestAfp)}")
    
        
        # Get SMILES and target values from the DeepChem DiskDataset
        allSmiles = np.concatenate([ds.ids for ds in dataset])  # ✅ concatenates numpy arrays
        allY = np.concatenate([ds.y.flatten() for ds in dataset])
        
        # Step 1: Split SMILES into train/test
        self.smilesTrainAfp, self.smilesTestAfp, self.yTrain, self.yTest = train_test_split(
            allSmiles.tolist(), allY.tolist(), test_size=0.2, random_state=42
        )
        
        # Create a MolGraphConvFeaturizer from DeepChem, which can convert the molecule 
        # into a graph representation suitable for GNNs. 
        # The use_edges=True argument indicates that we want to 
        # include edge features (e.g., bond types) in the graph representation, 
        # which can provide additional information for the GNN to learn from.
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        
        # Featurize the SMILES strings for both the training and test sets to create graph representations that can be used as input for a GNN model. This step is crucial for preparing the data in a format that the GNN can process effectively.
        xTrain = featurizer.featurize(self.smilesTrainAfp)
        xTest = featurizer.featurize(self.smilesTestAfp)
        
        # Force dtype=object to preserve GraphData objects
        xTrain = np.array(xTrain, dtype=object)
        xTest = np.array(xTest, dtype=object)
        
        # Wrap into DeepChem Dataset objects
        self.trainDatasetAfp = dc.data.NumpyDataset(X=xTrain, y=self.yTrain)
        self.testDatasetAfp = dc.data.NumpyDataset(X=xTest, y=self.yTest)
        
        print(f"GNN Train size: {len(self.smilesTrainAfp)}")
        print(f"GNN Test size: {len(self.smilesTestAfp)}")
    
    
    def featuriseAndSplitOnDeepChemDiskDatasets(self):
                
        for datasetIndex, diskDataset in enumerate(self.rawData):
        
            match datasetIndex:
                
                # Need to convert DiskDataset to DEEPCHEM NUMPYDATASET TO dtype=object first 
                case 0:
                    
                    # Extract SMILES and target values from the DiskDataset for training set
                    smilesTrain = diskDataset.ids.tolist()
                    yTrain = diskDataset.y.flatten()
                    
                    # 2. Data Synthesis - Augment existing SMILES strings in the training set to create 
                    # new, synthetic data points. This can help increase the diversity of the 
                    # training data and improve the model's ability to generalize to unseen compounds. 
                    smilesTrain, yTrain = self.augmentSmiles(smilesTrain, yTrain, augmentations=5)
                    
                    # 3. Featurize
                    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
                    xTrain = featurizer.featurize(smilesTrain)
                    # xTrain = np.array([x for x in xTrain if hasattr(x, 'to_dgl_graph')], dtype=object)

                    self.smilesTrainAfp = smilesTrain                    
                    self.yTrain = yTrain
                    
                    self.trainDatasetAfp = dc.data.NumpyDataset(X=xTrain, y=yTrain)

                    # self.trainDatasetAfp = diskDataset
                
                # This is validation set
                case 1:
                    
                    # Validation Set
                    smilesValidation = diskDataset.ids.tolist()
                    yValidation = diskDataset.y.flatten()
                    
                    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
                    xValidation = featurizer.featurize(smilesValidation)
                    self.smilesValidationAfp = smilesValidation                    
                    self.yValidation = yValidation
                    self.validationDatasetAfp = dc.data.NumpyDataset(X=xValidation, y=yValidation)
     
                case 2:
                    
                    # ✅ Re-featurize from SMILES using MolGraphConvFeaturizer
                    smilesTest = diskDataset.ids.tolist()
                    yTest = diskDataset.y.flatten()
                    
                    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
                    xTest = featurizer.featurize(smilesTest)
                    # xTest = np.array([x for x in xTest if hasattr(x, 'to_dgl_graph')], dtype=object)
                    
                    self.smilesTestAfp = smilesTest
                    self.yTest = yTest
                    self.testDatasetAfp = dc.data.NumpyDataset(X=xTest, y=yTest[:len(xTest)])

                case _:
                    
                    raise ValueError(f"Unexpected dataset index: {datasetIndex}. Expected indices are 0 (train), 1 (validation), and 2 (test).")
    
    
    '''
    Performs an 80-20 train-test split on the dataset, ensuring that the distribution of the target variable (measured log(solubility:mol/L)) is maintained in both sets. This is important to ensure that our model can generalize well to unseen data and that the performance metrics we calculate on the test set are representative of how the model will perform in real-world scenarios. The method also returns the compound IDs for both the training and test sets, which can be useful for tracking and analyzing specific compounds during model evaluation.
    '''
    def featuriseAndSplitPandasDataframe(self, featurisedData):
        
        # X
        X = self.refinedData["SMILES"].apply(self.featuriseSmilesForML).tolist()
        
        # Y
        y = self.refinedData["measured log(solubility:mol/L)"]
        
        # Train-test split (80-20)
        self.xTrain, self.xTest, self.yTrain, self.yTest, self.compoundIdTrain, self.compoundIdTest = train_test_split(X, y, self.refinedData["Compound ID"], test_size=0.2, random_state=42)
        
        
    '''
    Returns the train and test splits for traditional ML models (e.g., Random Forest, XGBoost), 
    including the featurised SMILES strings (as numerical features), the corresponding target values 
    (measured log(solubility:mol/L)), and the compound IDs for both the training and test sets. 
    
    This allows us to easily access the preprocessed data for training and 
    evaluating our machine learning models.
    
    Return:
        - xTrain: List of featurised SMILES strings for the training set, which can be used as input for traditional ML models.
        - yTrain: List of target values for the training set, which can be used as labels for traditional ML models.
        - xTest: List of featurised SMILES strings for the test set, which can be used as input for traditional ML models.
        - yTest: List of target values for the test set, which can be used as labels for traditional ML models.
        - compoundIdTrain: List of compound IDs for the training set, which can be used for reference or further analysis.
        - compoundIdTest: List of compound IDs for the test set, which can be used for reference or further analysis.
    '''
    def getTrainTestSplitsForML(self):
        
        return self.xTrain, self.yTrain, self.xTest, self.yTest, self.compoundIdTrain, self.compoundIdTest
    
    
    '''
    Returns the train and test splits specifically for the DeepChem GNN model (e.g., AttentiveFP), 
    including the SMILES strings and the corresponding DeepChem Dataset objects.
    
    Return:
        - smilesTrainAfp: List of SMILES strings for the training set, which can be used for reference or further analysis.
        - smilesTestAfp: List of SMILES strings for the test set, which can be used for reference or further analysis.
        - yTest: List of target values for the test set, which can be used as labels for evaluating the GNN model.
        - trainDatasetAfp: DeepChem Dataset object containing the graph representations and target values for
            the training set, which can be directly used as input for training a GNN model.
        - testDatasetAfp: DeepChem Dataset object containing the graph representations and target values for
            the test set, which can be directly used as input for evaluating a GNN model.
    '''
    def getTrainTestSplitsForGNN(self):
        
    
        return self.smilesTrainAfp, self.smilesTestAfp, self.smilesValidationAfp, self.yTest, self.yValidation, self.trainDatasetAfp, self.testDatasetAfp,  self.validationDatasetAfp
    
    '''
    Performs SMILES augmentation by generating multiple randomized SMILES strings for each input SMILES string.
    This technique can help increase the diversity of the training data and 
    improve the robustness of machine learning models by providing different representations of the same molecule. 
    The method takes a list of SMILES strings and their corresponding target values, and for each SMILES string, 
    it generates a specified number of augmented SMILES strings by randomizing the order of atoms in the molecule. 
    The original SMILES string and its target value are also included in the augmented dataset to ensure that the 
    model learns from both the original and augmented representations.
    
    -- Why did I do this? --
    R2 Score for Lipophilicity Prediction with AttentiveFP constantly gave me 57-60% R2 on the test set, 
    which was much lower than the 0.7+ R2 scores reported in the literature for this task. 
    I suspected that this was due to the limited size of the training data (only [4200,1024] compounds), 
    which may not have been sufficient for the GNN to learn robust representations of the molecules and their properties. 
    By performing SMILES augmentation, I was able to increase the effective size of the training data and 
    provide more diverse examples for the GNN to learn from, which may potentially improved the model's performance 
    on the test set.
    '''
    def augmentSmiles(self, smilesList, yList, augmentations=3):
        
        print("=" * 60 + "\n\n")
        print(f"Data Synthesis\n")
        print(f"Performing SMILES augmentation with {augmentations} augmentations per SMILES string...\n\n")
        
        augmentedSmiles = []
        augmentedY = []
        
        for smiles, y in zip(smilesList, yList):
            
            mol = MolFromSmiles(smiles)
            
            if mol:
                
                # Append the original SMILES and target value to the augmented lists
                augmentedSmiles.append(smiles)
                augmentedY.append(y)
                
                for _ in range(augmentations):
                    
                    randomisedSmiles = MolToSmiles(mol, doRandom=True)
                    
                    # Append the augmented SMILES and the same target value to the augmented lists
                    augmentedSmiles.append(randomisedSmiles)
                    augmentedY.append(y)
        
        print(f"Augmented {len(smilesList)} SMILES to {len(augmentedSmiles)} SMILES with {augmentations} augmentations each.")
        
        # Convert back to numpy array as MolGraphConvFeaturizer expects numpy arrays 
        # as input for Y
        augmentedY = np.array(augmentedY)
        
        # Save augmented dataset so that we do not have to augment it again every run
        self.saveAugmentedDataset()
        
        return augmentedSmiles, augmentedY
    
    
    '''
    Saves the augmented datasets to disk in NumPy format (.npy) for later use. This allows us to easily access the preprocessed 
    data for training and evaluating our GNN model without having to repeat the augmentation and featurisation steps, 
    which can be time-consuming. The method saves the graph representations (X) and the target values (y) for the training, 
    validation, and test sets in separate files within a specified directory.
    '''
    def saveAugmentedDataset(self, savePath='data/augmented'):
    
        os.makedirs(savePath, exist_ok=True)
        
        np.save(os.path.join(savePath, 'xTrain.npy'), self.trainDatasetAfp.X)
        np.save(os.path.join(savePath, 'yTrain.npy'), self.trainDatasetAfp.y)
        np.save(os.path.join(savePath, 'xValidation.npy'), self.validationDatasetAfp.X)
        np.save(os.path.join(savePath, 'yValidation.npy'), self.validationDatasetAfp.y)
        np.save(os.path.join(savePath, 'xTest.npy'), self.testDatasetAfp.X)
        np.save(os.path.join(savePath, 'yTest.npy'), self.testDatasetAfp.y)

        print(f"Augmented datasets saved to {savePath}")

    '''
    Loads the augmented datasets from disk and assigns them to the corresponding instance variables 
    for the training, validation, and test sets. This allows us to easily access the preprocessed data 
    for training and evaluating our GNN model without having to repeat the augmentation and featurisation steps, 
    which can be time-consuming. The method assumes that the augmented datasets are saved in NumPy format (.npy) 
    and that they contain both the graph representations (X) and the target values (y) for each set.
    '''
    def loadAugmentedDataset(self, savePath='data/augmented'):
    
        self.trainDatasetAfp = dc.data.NumpyDataset(
            X=np.load(os.path.join(savePath, 'xTrain.npy'), allow_pickle=True),
            y=np.load(os.path.join(savePath, 'yTrain.npy'), allow_pickle=True)
        )
        self.validationDatasetAfp = dc.data.NumpyDataset(
            X=np.load(os.path.join(savePath, 'xValidation.npy'), allow_pickle=True),
            y=np.load(os.path.join(savePath, 'yValidation.npy'), allow_pickle=True)
        )
        self.testDatasetAfp = dc.data.NumpyDataset(
            X=np.load(os.path.join(savePath, 'xTest.npy'), allow_pickle=True),
            y=np.load(os.path.join(savePath, 'yTest.npy'), allow_pickle=True)
        )
        
        print(f"Augmented datasets loaded from {savePath}")
    
    '''
    Run the pipeline of data preprocessing steps
    '''
    def run(self):
        
        # Some datasets are not Pandas DataFrames (e.g., DeepChem Datasets), 
        # so we need to check the type of rawData before performing 
        # the preprocessing steps that are specific to DataFrames (e.g., structural analysis, duplicate analysis, etc.). If rawData is not a DataFrame, we can skip these steps and proceed directly to the featurisation step, which can be adapted to handle different types of datasets as needed.
        if self.isPandasDataFrame:
        
            # 1. Structural Analysis and Missing Values
            self.performStructuralAnalysisAndMissingValues(self.rawData)
            
            # 2. Duplicate Analysis
            self.performDuplicateAnalysis("Compound ID")
            
            # 3. Categorical Variable Distribution
            self.investigateCategoricalVariableDistribution()
            
            # 4. Featurisation - We will perform different featurisation steps depending on 
            # whether we are using a traditional ML model (e.g., Random Forest) or a GNN. 
            # For traditional ML models, we will convert the SMILES strings into numerical features using molecular descriptors and fingerprints. 
            # For GNNs, we will convert the SMILES strings into graph representations that can be processed by the GNN.
            match self.modelType:
                
                case 'GNN':
                        
                    # 4.1 GNN Featurisation - For converting external datasets that are Pandas Dataframe 
                    # into graph representations suitable for DeepChem's GNNs (i.e AttentiveFP), 
                    # 
                    # We will implement a featurisation method that takes the SMILES strings 
                    # from the refined DataFrame and 
                    # uses DeepChem's MolGraphConvFeaturizer to convert them into 
                    # graph representations.
                    self.featuriseAndSplitDataFrameSmilesForGNN(self.refinedData)
                    
                case 'RandomForest' | 'XGBoost':
            
                    # 4.2 Train-Test Split (for External Datasets that are Pandas DataFrames) - For traditional ML models, we will perform an 80-20 train-test split on the refined dataset, ensuring that the distribution of the target variable (measured log(solubility:mol/L)) is maintained in both sets. This is important to ensure that our model can generalize well to unseen data and that the performance metrics we calculate on the test set are representative of how the model will perform in real-world scenarios. The method also returns the compound IDs for both the training and test sets, which can be useful for tracking and analyzing specific compounds during model evaluation.
                    self.featuriseAndSplitPandasDataframe(self.refinedData)
            
            
        # For DeepChem Datasets, we can skip the structural analysis, duplicate analysis, and categorical variable distribution steps, 
        # as these datasets are typically already preprocessed and formatted for use with 
        # DeepChem's models. 
        # 
        # Instead, we can directly perform the train-test split and featurisation steps that are 
        # specific to DeepChem's GNNs (e.g., AttentiveFP)
        if self.isDeepChemDataset:
            
            filePath = pathlib.Path("data/augmented")
            
            if filePath.exists():
                
                print(f"Augmented Dataset exists in {filePath}, loading dataset...")
                
                self.loadAugmentedDataset()
                
            
            self.featuriseAndSplitOnDeepChemDiskDatasets()
            
        