from DataPreprocessor import DataPreprocessor
from DatasetLoader import DatasetLoader
from MLPredictor import MLPredictor
from CustomModel.GNNPredictor import GNNPredictor


if __name__ == "__main__":

    # Load dataset
    datasetLoader = DatasetLoader()
    dataset = datasetLoader.loadKaggleDataset("yeonseokcho/delaney")
    
    # Load Data Preprocessor and get train and test sets
    preprocessorRandomForest = DataPreprocessor(dataset, "RandomForest")
    preprocessorRandomForest.run()
    xTrain, yTrain, xTest, yTest, compoundIdTrain, compoundIdTest = preprocessorRandomForest.getTrainTestSplits()

    preprocessorGNN = DataPreprocessor(dataset, "GNN")
    preprocessorGNN.run()
    smilesTrainAfp, smilesTestAfp, yTest, trainDatasetAfp, testDatasetAfp = preprocessorGNN.getTrainTestSplitsForGNN()

    #Initialize GNN predictor
    gnnPredictor = GNNPredictor(
        smilesTrain=smilesTrainAfp,
        smilesTest=smilesTestAfp,
        yTest = yTest,
        trainDataset=trainDatasetAfp,
        testDataset=testDatasetAfp
    )
    gnnPredictor.runPipeline()

    # Initialize predictor
    # mlPredictor = MLPredictor(
    #     xTrain=xTrain,
    #     yTrain=yTrain,
    #     xTest=xTest,
    #     yTest=yTest
    # )
    # mlPredictor.runPipeline()
    
    
    