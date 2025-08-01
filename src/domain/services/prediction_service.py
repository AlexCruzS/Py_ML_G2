from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class ModelTrainingService(ABC):
    """Servicio abstracto para entrenamiento de modelos."""
    
    @abstractmethod
    def train_model(self, data: pd.DataFrame, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Entrena un modelo con los datos e hiperparámetros proporcionados."""
        pass
    
    @abstractmethod
    def evaluate_model(self, model: Any, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evalúa el rendimiento del modelo."""
        pass

class PredictionService(ABC):
    """Servicio abstracto para predicciones."""
    
    @abstractmethod
    def predict_price(self, property_data: Any) -> float:
        """Realiza una predicción de precio usando el modelo."""
        pass