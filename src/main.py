from DataPreprocessor import DataPreprocessor
from DatasetLoader import DatasetLoader
from MLPredictor import MLPredictor
from CustomModels.GNNPredictor import GNNPredictor
import multiprocessing

def runSolubilityPipeline(dataset):
    
    # =============================================================================
    # A. Solubility Screening - Load Data Preprocessor and get train and test sets
    # =============================================================================

    # A.1 For Random Forest
    preprocessorRandomForestSolubility = DataPreprocessor(dataset, "RandomForest", 'solubility')
    preprocessorRandomForestSolubility.run()
    xTrain, yTrain, xTest, yTest, compoundIdTrain, compoundIdTest = preprocessorRandomForestSolubility.getTrainTestSplitsForML()

    # A.1.1 Initialize predictor
    mlPredictor = MLPredictor(
        xTrain=xTrain,
        yTrain=yTrain,
        xTest=xTest,
        yTest=yTest
    )
    mlPredictor.runPipeline()

    # A.2 For GNN
    preprocessorGNNSolubility = DataPreprocessor(dataset, "GNN", "DMPNN", "solubility")
    preprocessorGNNSolubility.run()
    smilesTrainAfp, smilesTestAfp, yTest, trainDatasetAfp, testDatasetAfp = preprocessorGNNSolubility.getTrainTestSplitsForGNN()

    # A.2.1 Initialize GNN predictor
    gnnPredictor = GNNPredictor(
        smilesTrain=smilesTrainAfp,
        smilesTest=smilesTestAfp,
        yTest = yTest,
        trainDataset=trainDatasetAfp,
        testDataset=testDatasetAfp
    )
    gnnPredictor.runPipeline()
    
def runLipophilicityPipelineAttentiveFP(dataset):
    
    preprocessorGNNLipophilicity = DataPreprocessor(dataset, "GNN", "AttentiveFP", 'lipophilicity')
    preprocessorGNNLipophilicity.run()
    
    smilesTrainAfp, smilesTestAfp, smilesValidationAfp, yTest, yValidation, trainDatasetAfp, testDatasetAfp, validationDatasetAfp = preprocessorGNNLipophilicity.getTrainTestSplitsForGNN()

    # A.2.1 Initialize GNN predictor
    gnnPredictor = GNNPredictor(
        smilesTrain=smilesTrainAfp,
        smilesTest=smilesTestAfp,
        smilesValidation=smilesValidationAfp,
        yTest = yTest,
        yValidation = yValidation,
        trainDataset=trainDatasetAfp,
        testDataset=testDatasetAfp,
        validationDataset=validationDatasetAfp,
        modelName='AttentiveFP'
    )
    gnnPredictor.runPipeline()

def runLipophilicityPipelineDMPNN(dataset):
    
    preprocessorGNNLipophilicity = DataPreprocessor(dataset, "GNN", "DMPNN", 'lipophilicity')
    preprocessorGNNLipophilicity.run()
    
    smilesTrainAfp, smilesTestAfp, smilesValidationAfp, yTest, yValidation, trainDatasetAfp, testDatasetAfp, validationDatasetAfp = preprocessorGNNLipophilicity.getTrainTestSplitsForGNN()

    # A.2.1 Initialize GNN predictor
    gnnPredictor = GNNPredictor(
        smilesTrain=smilesTrainAfp,
        smilesTest=smilesTestAfp,
        smilesValidation=smilesValidationAfp,
        yTest = yTest,
        yValidation = yValidation,
        trainDataset=trainDatasetAfp,
        testDataset=testDatasetAfp,
        validationDataset=validationDatasetAfp,
        modelName='DMPNN'
    )
    gnnPredictor.runPipeline()


if __name__ == "__main__":

    # Load dataset
    datasetLoader = DatasetLoader()
    
    # dataset = datasetLoader.loadKaggleDataset("yeonseokcho/delaney")
    dataset = datasetLoader.loadLipophilicityDataset()
    
    # =============================================================================
    # A. Solubility Screening - Load Data Preprocessor and get train and test sets
    # =============================================================================
    # runSolubilityPipeline(dataset)
    
    # =============================================================================
    # B. Lipophilicity Screening - Load Data Preprocessor and get train and test sets
    # =============================================================================
    
    # Instantiate two processes for AttentiveFP and DMPNN to run in parallel
    process1 = multiprocessing.Process(target=runLipophilicityPipelineAttentiveFP, args=(dataset,))
    process2 = multiprocessing.Process(target=runLipophilicityPipelineDMPNN, args=(dataset,))
    
    # Start the processes
    process1.start() 
    process2.start()
    
    # Wait for both processes to finish
    process1.join()
    process2.join()
    



    
    
    