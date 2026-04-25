

class GNNHyperparamGrid:
    '''
    This class defines the hyperparameter grid for GNN models (AttentiveFP, GCN, DMPNN) that will be used in the hyperparameter tuning process. 
    The hyperparameters include learning rate, number of epochs, batch size, hidden layer sizes, dropout rate, and others specific to the GNN architecture. 
    The grid is defined as a dictionary where each key is a hyperparameter name and the corresponding value is a list of possible values for that hyperparameter.
    '''
    def __init__(self, modelName):

        self.modelName = modelName
        self.hyperparamGrid = {}


    def getHyperparamGrid(self):

        # [Grid Search Parameters] 
        # [TODO] To document grids like a config file in a separate script
        match self.modelName:

            case 'AttentiveFP':

                # For AttentiveFP, we focus on tuning num_layers, num_timesteps, graph_feat_size, dropout and learning_rate for a start
                self.hyperparamGrid = [
                    {'num_layers': 2, 'num_timesteps': 2, 'graph_feat_size': 200, 'dropout': 0.1, 'learning_rate': 0.001 },
                    # {'num_layers': 2, 'num_timesteps': 3, 'graph_feat_size': 200, 'dropout': 0.2, 'learning_rate': 0.001 },
                    # {'num_layers': 3, 'num_timesteps': 2, 'graph_feat_size': 200, 'dropout': 0.2, 'learning_rate': 0.001 },
                    {'num_layers': 3, 'num_timesteps': 3, 'graph_feat_size': 300, 'dropout': 0.3, 'learning_rate': 0.001 },
                    # {'num_layers': 4, 'num_timesteps': 2, 'graph_feat_size': 200, 'dropout': 0.3, 'learning_rate': 0.0005},
                    # {'num_layers': 4, 'num_timesteps': 3, 'graph_feat_size': 300, 'dropout': 0.4, 'learning_rate': 0.0005},
                ]

            case 'DMPNN':

                # For DMPNN, we focus on tuning learning_rate, batch_size and enc_dropout_p for a start
                # We are getting an r2 score of 62.28% with enc_dropout_p of 0.1, learning_rate of 0.001 and batch_size of 64, so we will explore around these values in the grid search.
                self.hyperparamGrid = [
                    {'enc_dropout_p': 0.1, 'learning_rate': 0.001, 'batch_size': 64},
                    # {'enc_dropout_p': 0.2, 'learning_rate': 0.001, 'batch_size': 64},
                    # {'enc_dropout_p': 0.1, 'learning_rate': 0.0005, 'batch_size': 64},
                    {'enc_dropout_p': 0.2, 'learning_rate': 0.0005, 'batch_size': 64},
                ]

            case 'GCN':

                self.hyperparamGrid = [
                    {'graph_conv_layers': [64, 64], 'dropout': 0.1, 'learning_rate': 0.001,  'batch_size': 32, 'residual': True},
                    # {'graph_conv_layers': [128, 128], 'dropout': 0.1, 'learning_rate': 0.001,  'batch_size': 32, 'residual': True},
                    {'graph_conv_layers': [64, 64, 64], 'dropout': 0.2, 'learning_rate': 0.001,  'batch_size': 32, 'residual': True},
                    # {'graph_conv_layers': [128, 64], 'dropout': 0.2, 'learning_rate': 0.0005, 'batch_size': 32, 'residual': True},
                ] 

        return self.hyperparamGrid
    


