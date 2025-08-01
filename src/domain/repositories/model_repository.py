from abc import ABC, abstractmethod
from typing import Any, Dict, List
import pandas as pd

class ModelRepository(ABC):
    """Interface para el repositorio de modelos."""
    
    @abstractmethod
    def save_model(self, model: Any, params: Dict[str, Any], metrics: Dict[str, float]) -> str:
        """Guarda un modelo entrenado con sus parámetros y métricas."""
        pass
    
    @abstractmethod
    def load_model(self, model_uri: str) -> Any:
        """Carga un modelo desde su URI."""
        pass
    
    @abstractmethod
    def get_best_model(self, metric_name: str = "rmse") -> Any:
        """Obtiene el mejor modelo basado en una métrica."""
        pass
    
    @abstractmethod
    def set_experiment(self, experiment_name: str) -> None:
        """Configura el experimento para tracking."""
        pass

class DataRepository(ABC):
    """Interface para el repositorio de datos."""
    
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Carga datos desde un archivo."""
        pass
    
    @abstractmethod
    def preprocess_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Preprocesa los datos y retorna features y target."""
        pass