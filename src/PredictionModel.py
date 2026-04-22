from typing import Literal
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import deepchem.models as dc_models

'''
A Factory class to create different types of prediction models based on the specified model type.
'''
class PredictionModel:
    
    def __new__(cls, modelType: Literal['RandomForest', 'XGBoost', 'GNN']):
        
        match modelType:
            
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
                
                return dc_models.MPNNModel(
                    n_tasks=1,
                    n_hidden=3,
                    dropout=0.1,
                    learning_rate=0.001 
                )
                
            case _:
                
                raise ValueError(f"Unsupported model type: {modelType}. Supported types are: 'RandomForest', 'XGBoost', 'GNN'.") 