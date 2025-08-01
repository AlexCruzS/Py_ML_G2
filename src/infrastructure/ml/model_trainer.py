import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from typing import Dict, Any
from ...domain.services.prediction_service import ModelTrainingService, PredictionService
from ...domain.repositories.model_repository import DataRepository, ModelRepository
from ...domain.entities.property import Property

class RealEstateModelTrainer(ModelTrainingService):
    """Implementación del servicio de entrenamiento para modelos inmobiliarios."""
    
    def __init__(
        self, 
        data_repository: DataRepository,
        model_repository: ModelRepository
    ):
        self.data_repository = data_repository
        self.model_repository = model_repository
    
    def train_model(self, file_path: str, **hyperparams) -> Dict[str, Any]:
        """
        Entrena un modelo RandomForest con los datos especificados.
        
        Args:
            file_path: Ruta al archivo de datos
            **hyperparams: Hiperparámetros del modelo
            
        Returns:
            Diccionario con información del entrenamiento
        """
        
        # Parámetros por defecto
        n_estimators = hyperparams.get('n_estimators', 100)
        max_depth = hyperparams.get('max_depth', 5)
        random_state = hyperparams.get('random_state', 42)
        test_size = hyperparams.get('test_size', 0.2)
        
        # Cargar y preprocesar datos
        df = self.data_repository.load_data(file_path)
        X, y = self.data_repository.preprocess_data(df)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Crear y entrenar modelo
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        
        print(f"Entrenando modelo con {n_estimators} estimadores y profundidad {max_depth}...")
        model.fit(X_train, y_train)
        
        # Evaluar modelo
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # Preparar parámetros y signature para MLflow
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state
        }
        
        # Crear ejemplo de entrada y signature
        input_example = X_train.iloc[:2] if hasattr(X_train, 'iloc') else X_train[:2]
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        
        # Guardar modelo en MLflow
        model_uri = self.model_repository.save_model(
            model=model,
            params=params,
            metrics=metrics,
            input_example=input_example,
            signature=signature
        )
        
        return {
            'model_uri': model_uri,
            'rmse': metrics['rmse'],
            'experiment_id': mlflow.active_run().info.experiment_id,
            'run_id': mlflow.active_run().info.run_id,
            'model': model
        }
    
    def evaluate_model(self, model, X_test, y_test) -> Dict[str, float]:
        """
        Evalúa un modelo entrenado.
        
        Args:
            model: Modelo entrenado
            X_test: Features de prueba
            y_test: Target de prueba
            
        Returns:
            Diccionario con métricas de evaluación
        """
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return {
            'rmse': rmse
        }

class RealEstatePredictionService(PredictionService):
    """Implementación del servicio de predicción para propiedades inmobiliarias."""
    
    def __init__(self, model_repository: ModelRepository):
        self.model_repository = model_repository
        self._model = None
    
    def _get_model(self):
        """Obtiene el modelo actual (lazy loading)."""
        if self._model is None:
            self._model = self.model_repository.get_best_model()
        return self._model
    
    def predict_price(self, property_data: Property) -> float:
        """
        Predice el precio de una propiedad.
        
        Args:
            property_data: Datos de la propiedad
            
        Returns:
            Precio predicho
        """
        
        model = self._get_model()
        
        # Convertir propiedad a formato de predicción
        feature_dict = property_data.to_dict()
        
        # Crear array de features en el orden correcto
        features = [
            feature_dict['Assessed Value'],
            feature_dict['area_m2'],
            feature_dict['meses_en_venta'],
            feature_dict['nro_habitaciones'],
            feature_dict['nro_pisos'],
            feature_dict['Property Type_Residential'],
            feature_dict['Property Type_Single Family']
        ]
        
        # Predecir
        prediction = model.predict([features])
        return float(prediction[0])
    
    def predict_batch(self, properties: list[Property]) -> list[float]:
        """
        Predice precios para múltiples propiedades.
        
        Args:
            properties: Lista de propiedades
            
        Returns:
            Lista de precios predichos
        """
        
        model = self._get_model()
        
        # Convertir todas las propiedades a features
        features_list = []
        for prop in properties:
            feature_dict = prop.to_dict()
            features = [
                feature_dict['Assessed Value'],
                feature_dict['area_m2'],
                feature_dict['meses_en_venta'],
                feature_dict['nro_habitaciones'],
                feature_dict['nro_pisos'],
                feature_dict['Property Type_Residential'],
                feature_dict['Property Type_Single Family']
            ]
            features_list.append(features)
        
        # Predecir en lote
        predictions = model.predict(features_list)
        return [float(pred) for pred in predictions]
    
    def validate_property(self, property_data: Property) -> bool:
        """
        Valida que los datos de la propiedad sean correctos.
        
        Args:
            property_data: Datos de la propiedad
            
        Returns:
            True si es válida, False en caso contrario
        """
        
        if property_data.assessed_value <= 0:
            return False
        if property_data.area_m2 <= 0:
            return False
        if property_data.meses_en_venta < 0:
            return False
        if property_data.nro_habitaciones < 1:
            return False
        if property_data.nro_pisos < 1:
            return False
        
        return True