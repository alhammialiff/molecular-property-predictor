

from typing import Literal
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import deepchem.models as dc_models


class PredictionModel:
    
    def __new__(cls, modelType: Literal['RandomForest', 'XGBoost']):
        
        match modelType:
            
            case 'RandomForest':
                
                return RandomForestRegressor()
                
            
            case 'XGBoost':
                
                return XGBRegressor()
            
            case 'GNN':
                
                return dc_models.AttentiveFPModel(
                    n_tasks=1,
                    mode='regression',
                    num_layers=3,
                    num_timesteps=2,
                    graph_feat_size=200,
                    dropout=0.2,
                    batch_size=32,
                    learning_rate=0.001,
                )
                
            
            case _:
                
                raise ValueError(f"Unsupported model type: {modelType}. Supported types are: 'RandomForest', 'XGBoost', 'LightGBM'.") 