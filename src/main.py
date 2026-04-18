from DataPreprocessor import DataPreprocessor
from DatasetLoader import DatasetLoader
from MolecularSolubilityPredictor import MolecularSolubilityPredictor


if __name__ == "__main__":

    # Load dataset
    datasetLoader = DatasetLoader()
    dataset = datasetLoader.loadKaggleDataset("yeonseokcho/delaney")
    
    # Load Data Preprocessor and get train and test sets
    preprocessor = DataPreprocessor(dataset)
    preprocessor.run()
    xTrain, yTrain, xTest, yTest, compoundIdTrain, compoundIdTest = preprocessor.getTrainTestSplits()

    # Initialize predictor
    predictor = MolecularSolubilityPredictor(
        xTrain=xTrain,
        yTrain=yTrain,
        xTest=xTest,
        yTest=yTest
    )
    predictor.runPipeline()
    