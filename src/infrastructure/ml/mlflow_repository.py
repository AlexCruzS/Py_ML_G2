import mlflow
import mlflow.sklearn
from typing import Any, Dict
from domain.repositories.model_repository import ModelRepository

class MLflowModelRepository(ModelRepository):
    """Implementación del repositorio de modelos usando MLflow."""
    
    def __init__(self):
        """Inicializa el repositorio MLflow."""
        self.client = mlflow.tracking.MlflowClient()
    
    def set_experiment(self, experiment_name: str) -> None:
        """
        Configura el experimento para tracking.
        
        Args:
            experiment_name: Nombre del experimento
        """
        mlflow.set_experiment(experiment_name)
        print(f"Experimento configurado: {experiment_name}")
    
    def save_model(
        self, 
        model: Any, 
        params: Dict[str, Any], 
        metrics: Dict[str, float],
        input_example: Any = None,
        signature: Any = None,
        registered_model_name: str = "Proyec_Inmobiliario_Model"
    ) -> str:
        """
        Guarda un modelo entrenado con sus parámetros y métricas.
        
        Args:
            model: Modelo entrenado
            params: Parámetros del modelo
            metrics: Métricas de evaluación
            input_example: Ejemplo de entrada
            signature: Firma del modelo
            registered_model_name: Nombre en el registry
            
        Returns:
            URI del modelo guardado
        """
        
        with mlflow.start_run() as run:
            # Log parámetros
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Log métricas
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log modelo
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="random_forest_model",
                input_example=input_example,
                signature=signature,
                registered_model_name=registered_model_name
            )
            
            print(f"Modelo guardado con RMSE: {metrics.get('rmse', 'N/A'):.2f}")
            print(f"Run ID: {run.info.run_id}")
            
            return model_info.model_uri
    
    def load_model(self, model_uri: str) -> Any:
        """
        Carga un modelo desde su URI.
        
        Args:
            model_uri: URI del modelo
            
        Returns:
            Modelo cargado
        """
        return mlflow.sklearn.load_model(model_uri)
    
    def get_best_model(self, metric_name: str = "rmse") -> Any:
        """
        Obtiene el mejor modelo basado en una métrica.
        
        Args:
            metric_name: Nombre de la métrica para comparar
            
        Returns:
            Mejor modelo encontrado
        """
        try:
            # Buscar todos los modelos registrados
            registered_models = self.client.search_registered_models()
            
            best_model_uri = None
            best_metric_value = float('inf') if metric_name == "rmse" else float('-inf')
            
            for rm in registered_models:
                if rm.name == "Proyec_Inmobiliario_Model":
                    for version in rm.latest_versions:
                        run = self.client.get_run(version.run_id)
                        metric_value = run.data.metrics.get(metric_name)
                        
                        if metric_value is not None:
                            if (metric_name == "rmse" and metric_value < best_metric_value) or \
                               (metric_name != "rmse" and metric_value > best_metric_value):
                                best_metric_value = metric_value
                                best_model_uri = version.source
            
            if best_model_uri:
                return self.load_model(best_model_uri)
            else:
                raise Exception(f"No se encontró ningún modelo con métrica {metric_name}")
                
        except Exception as e:
            print(f"Error al obtener el mejor modelo: {str(e)}")
            raise