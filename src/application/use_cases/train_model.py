from typing import Dict, Any
from ..dto.property_dto import TrainingResultDTO
from ...domain.repositories.model_repository import DataRepository, ModelRepository
from ...domain.services.prediction_service import ModelTrainingService

class TrainModelUseCase:
    """Caso de uso para entrenar modelos de predicción de precios."""
    
    def __init__(
        self, 
        data_repository: DataRepository,
        model_repository: ModelRepository,
        training_service: ModelTrainingService
    ):
        self.data_repository = data_repository
        self.model_repository = model_repository
        self.training_service = training_service
    
    def execute(
        self, 
        file_path: str, 
        n_estimators: int = 100, 
        max_depth: int = 5,
        experiment_name: str = "Grupo_2_Proyecto_Inmobiliario"
    ) -> TrainingResultDTO:
        """
        Ejecuta el entrenamiento de un modelo.
        
        Args:
            file_path: Ruta al archivo de datos
            n_estimators: Número de estimadores para RandomForest
            max_depth: Profundidad máxima del árbol
            experiment_name: Nombre del experimento en MLflow
            
        Returns:
            TrainingResultDTO con los resultados del entrenamiento
        """
        
        # Configurar experimento
        self.model_repository.set_experiment(experiment_name)
        
        # Entrenar modelo
        training_result = self.training_service.train_model(
            file_path=file_path,
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        
        return TrainingResultDTO(
            model_uri=training_result["model_uri"],
            rmse=training_result["rmse"],
            n_estimators=n_estimators,
            max_depth=max_depth,
            experiment_id=training_result["experiment_id"],
            run_id=training_result["run_id"]
        )