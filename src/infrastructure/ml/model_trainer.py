import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 👈 Agregado MAE y R²
from typing import Dict, Any
from domain.services.prediction_service import ModelTrainingService, PredictionService
from domain.repositories.model_repository import DataRepository, ModelRepository
from domain.entities.property import Property

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
        
        # Evaluar modelo (ahora incluye MAE y R²)
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
            'mae': metrics['mae'],                    # 👈 Agregado MAE
            'r2_score': metrics['r2_score'],          # 👈 Agregado R²
            'experiment_id': "experiment_id_placeholder",
            'run_id': "run_id_placeholder", 
            'model': model
        }
    
    def evaluate_model(self, model, X_test, y_test) -> Dict[str, float]:
        """
        Evalúa un modelo entrenado con múltiples métricas.
        
        Args:
            model: Modelo entrenado
            X_test: Features de prueba
            y_test: Target de prueba
            
        Returns:
            Diccionario con métricas de evaluación completas
        """
        
        # Hacer predicciones
        y_pred = model.predict(X_test)
        
        # Calcular todas las métricas
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)              # 👈 Calcular MAE
        r2 = r2_score(y_test, y_pred)                          # 👈 Calcular R²
        
        # Calcular métricas adicionales útiles
        mse = mean_squared_error(y_test, y_pred)
        
        # Estadísticas descriptivas de los errores
        errors = y_test - y_pred
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        print(f"📊 Métricas de Evaluación:")
        print(f"   RMSE: ${rmse:,.2f}")
        print(f"   MAE:  ${mae:,.2f}")
        print(f"   R²:   {r2:.4f}")
        print(f"   MSE:  ${mse:,.2f}")
        print(f"   Error promedio: ${mean_error:,.2f}")
        print(f"   Desv. estándar del error: ${std_error:,.2f}")
        
        return {
            'rmse': rmse,
            'mae': mae,                    # 👈 Incluir MAE
            'r2_score': r2,               # 👈 Incluir R²
            'mse': mse,
            'mean_error': mean_error,
            'std_error': std_error
        }
    
    def evaluate_model_comprehensive(self, model, X_test, y_test) -> Dict[str, Any]:
        """
        Evaluación más comprehensiva del modelo (opcional).
        
        Args:
            model: Modelo entrenado
            X_test: Features de prueba
            y_test: Target de prueba
            
        Returns:
            Diccionario con evaluación detallada
        """
        
        y_pred = model.predict(X_test)
        
        # Métricas básicas
        basic_metrics = self.evaluate_model(model, X_test, y_test)
        
        # Métricas adicionales para inmobiliaria
        errors = y_test - y_pred
        abs_errors = np.abs(errors)
        
        # Porcentaje de predicciones dentro de rangos aceptables
        within_10_percent = np.mean(abs_errors / y_test <= 0.10) * 100
        within_20_percent = np.mean(abs_errors / y_test <= 0.20) * 100
        within_30_percent = np.mean(abs_errors / y_test <= 0.30) * 100
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(abs_errors / y_test) * 100
        
        comprehensive_metrics = {
            **basic_metrics,
            'mape': mape,
            'within_10_percent': within_10_percent,
            'within_20_percent': within_20_percent,
            'within_30_percent': within_30_percent,
            'max_error': np.max(abs_errors),
            'min_error': np.min(abs_errors),
            'median_error': np.median(abs_errors)
        }
        
        print(f"📈 Métricas Adicionales:")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Predicciones dentro del 10%: {within_10_percent:.1f}%")
        print(f"   Predicciones dentro del 20%: {within_20_percent:.1f}%")
        print(f"   Predicciones dentro del 30%: {within_30_percent:.1f}%")
        
        return comprehensive_metrics

class RealEstatePredictionService(PredictionService):
    """Implementación del servicio de predicción para propiedades inmobiliarias."""
    
    def __init__(self, model_repository: ModelRepository):
        self.model_repository = model_repository
        self._model = None
        self._model_metrics = None  # 👈 Cache para métricas del modelo
    
    def _get_model(self):
        """Obtiene el modelo actual (lazy loading)."""
        if self._model is None:
            self._model = self.model_repository.get_best_model()
        return self._model
    
    def get_model_metrics(self) -> Dict[str, float]:
        """
        Obtiene las métricas del modelo actual.
        
        Returns:
            Diccionario con métricas del modelo
        """
        if self._model_metrics is None:
            # Intentar obtener métricas del repositorio
            try:
                self._model_metrics = self.model_repository.get_model_metrics()
            except:
                # Métricas por defecto si no están disponibles
                self._model_metrics = {
                    'rmse': 0.0,
                    'mae': 0.0,
                    'r2_score': 0.0
                }
        
        return self._model_metrics
    
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
    
    def predict_with_confidence(self, property_data: Property) -> Dict[str, Any]:
        """
        Predice el precio con información de confianza.
        
        Args:
            property_data: Datos de la propiedad
            
        Returns:
            Diccionario con predicción y métricas de confianza
        """
        
        prediction = self.predict_price(property_data)
        metrics = self.get_model_metrics()
        
        # Calcular intervalos de confianza basados en MAE y RMSE
        mae = metrics.get('mae', 0)
        rmse = metrics.get('rmse', 0)
        r2 = metrics.get('r2_score', 0)
        
        # Intervalos conservadores
        confidence_interval_mae = {
            'lower': prediction - mae,
            'upper': prediction + mae
        }
        
        confidence_interval_rmse = {
            'lower': prediction - rmse,
            'upper': prediction + rmse
        }
        
        return {
            'predicted_price': prediction,
            'model_metrics': metrics,
            'confidence_interval_mae': confidence_interval_mae,
            'confidence_interval_rmse': confidence_interval_rmse,
            'model_quality': 'Excelente' if r2 >= 0.90 else 
                           'Bueno' if r2 >= 0.80 else 
                           'Aceptable' if r2 >= 0.70 else 'Necesita mejora'
        }
    
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
    
    def validate_property(self, property_data: Property) -> Dict[str, Any]:
        """
        Valida que los datos de la propiedad sean correctos.
        
        Args:
            property_data: Datos de la propiedad
            
        Returns:
            Diccionario con resultado de validación y detalles
        """
        
        errors = []
        warnings = []
        
        # Validaciones estrictas
        if property_data.assessed_value <= 0:
            errors.append("El valor tasado debe ser mayor a 0")
        if property_data.area_m2 <= 0:
            errors.append("El área debe ser mayor a 0 m²")
        if property_data.meses_en_venta < 0:
            errors.append("Los meses en venta no pueden ser negativos")
        if property_data.nro_habitaciones < 1:
            errors.append("Debe tener al menos 1 habitación")
        if property_data.nro_pisos < 1:
            errors.append("Debe tener al menos 1 piso")
        
        # Validaciones de advertencia (valores inusuales pero posibles)
        if property_data.assessed_value > 10000000:  # $10M
            warnings.append("Valor tasado muy alto - verificar precisión")
        if property_data.area_m2 > 1000:
            warnings.append("Área muy grande - verificar unidades")
        if property_data.area_m2 < 30:
            warnings.append("Área muy pequeña para una vivienda")
        if property_data.meses_en_venta > 24:
            warnings.append("Tiempo en mercado muy largo")
        if property_data.nro_habitaciones > 10:
            warnings.append("Número inusual de habitaciones")
        if property_data.nro_pisos > 5:
            warnings.append("Número inusual de pisos para vivienda")
        
        is_valid = len(errors) == 0
        
        return {
            'is_valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'validation_summary': f"{'✅ Válido' if is_valid else '❌ Inválido'} - {len(errors)} errores, {len(warnings)} advertencias"
        }