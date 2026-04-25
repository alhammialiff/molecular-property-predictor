import os

class DirectoryGenerator:
    '''
    This class defines the hyperparameter grid for GNN models (AttentiveFP, GCN, DMPNN) that will be used in the hyperparameter tuning process. 
    The hyperparameters include learning rate, number of epochs, batch size, hidden layer sizes, dropout rate, and others specific to the GNN architecture. 
    The grid is defined as a dictionary where each key is a hyperparameter name and the corresponding value is a list of possible values for that hyperparameter.
    '''
    def __init__(self, modelName):

        self.modelName = modelName

        # [Checkpoint and Summary Directories]

        # [Attentive FP]
        self.validationSummariesDirAttentiveFP = os.path.join(os.path.dirname(__file__), '..', 'Reports', 'AttentiveFP','ValidationSummaries')
        self.testSummariesDirAttentiveFP = os.path.join(os.path.dirname(__file__), '..', 'Reports', 'AttentiveFP','TestSummaries')
        self.bestModelDirAttentiveFP = os.path.join(os.path.dirname(__file__), '..', 'ModelCheckpoints', 'AttentiveFP' , 'BestModel')
        
        # [DMPNN]
        self.validationSummariesDirDMPNN = os.path.join(os.path.dirname(__file__), '..', 'Reports', 'DMPNN','ValidationSummaries')
        self.testSummariesDirDMPNN = os.path.join(os.path.dirname(__file__), '..', 'Reports', 'DMPNN','TestSummaries')
        self.bestModelDirDMPNN = os.path.join(os.path.dirname(__file__), '..', 'ModelCheckpoints', 'DMPNN' , 'BestModel')
        
        # [GCN]
        self.validationSummariesDirGCN = os.path.join(os.path.dirname(__file__), '..', 'Reports', 'GCN','ValidationSummaries')
        self.testSummariesDirGCN = os.path.join(os.path.dirname(__file__), '..', 'Reports', 'GCN','TestSummaries')
        self.bestModelDirGCN = os.path.join(os.path.dirname(__file__), '..', 'ModelCheckpoints', 'GCN' , 'BestModel')
        
    
    def getDirectories(self):

        match self.modelName:

            case 'AttentiveFP':

                return {
                    "validationSummariesDir": self.validationSummariesDirAttentiveFP,
                    "bestModelDir": self.bestModelDirAttentiveFP
                }
                
            case 'DMPNN':

                return {
                    "validationSummariesDir": self.validationSummariesDirDMPNN,
                    "bestModelDir": self.bestModelDirDMPNN
                }   

            case 'GCN':

                return {
                    "validationSummariesDir": self.validationSummariesDirGCN,
                    "bestModelDir": self.bestModelDirGCN
                }


        