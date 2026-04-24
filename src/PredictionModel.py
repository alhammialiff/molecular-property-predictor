from typing import Literal
from numpy import mat
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import deepchem.models as dc_models

'''
A Factory class to create different types of prediction models based on the specified model type.
'''
class PredictionModel:
    
    def __new__(
        cls, 
        modelType: Literal['ML', 'ANN', 'GNN'], 
        modelName: str = None, 
        hyperparameters: dict = {}
    ):
        
        # [TODO] To switch Model Type to ML, ANN, GNN, and then modelName to specify the specific model within that type (e.g. RandomForest, XGBoost for ML; AttentiveFP, DMPNN for GNN)
        match modelType:
            
            case 'ML':

                match modelName:
                    
                    case 'RandomForest':
                        
                        return RandomForestRegressor()
                        
                    case 'XGBoost':
                        
                        return XGBRegressor()
            
            case 'GNN':
                
                # Training 4 
                # MSE: 0.3420
                # RMSE: 0.5848
                # MAE: 0.4304
                # R2: 0.5885
                # Training Duration: 3 hours 10 mins!!!
                #
                # Measures to take: To synthesize more data for the training set
                #
                # return dc_models.AttentiveFPModel(
                #     n_tasks=1,
                #     mode='regression',
                #     num_layers=5,           # ⬆️ deeper network
                #     num_timesteps=3,        # ⬆️ better readout
                #     graph_feat_size=300,    # ⬆️ wider network
                #     dropout=0.1,            # ⬇️ less dropout for larger dataset
                #     batch_size=16,          # ⬇️ smaller batch = better gradients
                #     learning_rate=0.0003,   # fine-tuned LR
                # )
            
                # Training 3
                # MSE: 0.3525
                # RMSE: 0.5937
                # MAE: 0.4502
                # R2: 0.5758
                # Training Duration: 2735.08 secs
                #
                # return dc_models.AttentiveFPModel(
                #     n_tasks=1,
                #     mode='regression',
                #     num_layers=5,           # ⬇️ revert
                #     num_timesteps=2,        # ⬇️ revert
                #     graph_feat_size=200,    # ⬇️ revert
                #     dropout=0.1,
                #     batch_size=32,
                #     learning_rate=0.0005,    # ⬆️ revert
                # )
                
                # Training #2
                # MSE: 0.3565
                # RMSE: 0.5714
                # MAE: 0.4233
                # R2: 0.6071
                # Training Duration: 2023.69 secs
                #
                # return dc_models.AttentiveFPModel(
                #     n_tasks=1,
                #     mode='regression',
                #     num_layers=3,
                #     num_timesteps=2,
                #     graph_feat_size=200,
                #     dropout=0.2,
                #     learning_rate=0.001,
                #     batch_size=32,
                    
                # )

                
                match modelName:

                    case 'AttentiveFP':

                        return dc_models.AttentiveFPModel(
                            n_tasks=1,
                            mode='regression',
                            num_layers=3,
                            num_timesteps=2,
                            graph_feat_size=200,
                            dropout=0.2,
                            learning_rate=0.001,
                            batch_size=32,
                        )
                    
                    case 'DMPNN':

                        # Work on DMPNN model first, modularising of models can come later
                        # Default Model Hyperparameters (if none provided via grid search)
                        if hyperparameters is None or hyperparameters == {}:
                            
                            return dc_models.DMPNNModel(
                                n_tasks=1,
                                batch_size=32,
                                learning_rate=0.001,
                                enc_dropout_p=0.1
                            )
                        
                        # Model Hyperparameters provided via grid search
                        else:

                            return dc_models.DMPNNModel(
                                n_tasks=1,
                                batch_size=hyperparameters['batch_size'],
                                learning_rate=hyperparameters['learning_rate'],
                                enc_dropout_p=hyperparameters['enc_dropout_p']
                            )
                
                
            case _:
                
                raise ValueError(f"Unsupported model type: {modelType}. Supported types are: 'RandomForest', 'XGBoost', 'GNN'.") 