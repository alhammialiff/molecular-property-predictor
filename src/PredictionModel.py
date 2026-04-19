

from typing import Literal
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class PredictionModel:
    
    def __new__(cls, modelType: Literal['RandomForest', 'XGBoost']):
        
        match modelType:
            
            case 'RandomForest':
                
                return RandomForestRegressor()
                
            
            case 'XGBoost':
                
                return XGBRegressor()
                
            
            case _:
                
                raise ValueError(f"Unsupported model type: {modelType}. Supported types are: 'RandomForest', 'XGBoost', 'LightGBM'.") 