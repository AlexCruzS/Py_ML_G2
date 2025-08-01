"""
Punto de entrada principal para el proyecto de ML Inmobiliario.
Implementa arquitectura hexagonal con inyección de dependencias.
"""

import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar MLflow
import mlflow
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))

from infrastructure.data.data_loader import CSVDataLoader
from infrastructure.ml.mlflow_repository import MLflowModelRepository
from infrastructure.ml.model_trainer import RealEstateModelTrainer, RealEstatePredictionService
from application.use_cases.train_model import TrainModelUseCase
from application.use_cases.predict_price import PredictPriceUseCase
from application.dto.property_dto import PropertyInputDTO

def setup_dependencies():
    """
    Configura la inyección de dependencias para la aplicación.
    
    Returns:
        Tuple con los casos de uso configurados
    """
    
    # Repositorios (Infrastructure)
    data_repository = CSVDataLoader()
    model_repository = MLflowModelRepository()
    
    # Servicios (Infrastructure)
    training_service = RealEstateModelTrainer(data_repository, model_repository)
    prediction_service = RealEstatePredictionService(model_repository)
    
    # Casos de uso (Application)
    train_use_case = TrainModelUseCase(data_repository, model_repository, training_service)
    predict_use_case = PredictPriceUseCase(prediction_service, model_repository)
    
    return train_use_case, predict_use_case

def train_example():
    """Ejemplo de entrenamiento de modelo."""
    
    print("🚀 Iniciando entrenamiento de modelo...")
    
    train_use_case, _ = setup_dependencies()
    
    # Configurar parámetros de entrenamiento
    data_path = "data/dataset_inmobi.csv"  # Ajustar según tu estructura
    
    try:
        result = train_use_case.execute(
            file_path=data_path,
            n_estimators=100,
            max_depth=5,
            experiment_name="Grupo_2_Proyecto_Inmobiliario"
        )
        
        print(f"✅ Entrenamiento completado!")
        print(f"   RMSE: {result.rmse:.2f}")
        print(f"   Model URI: {result.model_uri}")
        print(f"   Run ID: {result.run_id}")
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {str(e)}")

def predict_example():
    """Ejemplo de predicción de precios."""
    
    print("🔮 Realizando predicción de ejemplo...")
    
    _, predict_use_case = setup_dependencies()
    
    # Crear propiedad de ejemplo
    property_example = PropertyInputDTO(
        assessed_value=300000.0,
        area_m2=150.0,
        meses_en_venta=6,
        nro_habitaciones=3,
        nro_pisos=2,
        property_type="Single Family"
    )
    
    try:
        result = predict_use_case.execute(property_example)
        
        print(f"✅ Predicción completada!")
        print(f"   Precio predicho: ${result.predicted_price:,.2f}")
        print(f"   Versión del modelo: {result.model_version}")
        
    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")

def main():
    """Función principal."""
    
    print("🏠 Sistema de Predicción de Precios Inmobiliarios")
    print("   Arquitectura Hexagonal + MLflow + Streamlit")
    print("=" * 50)
    
    # Verificar configuración
    print(f"📊 MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    
    # Mostrar opciones
    print("\nOpciones disponibles:")
    print("1. Entrenar modelo de ejemplo")
    print("2. Realizar predicción de ejemplo")
    print("3. Iniciar aplicación Streamlit")
    
    choice = input("\nSelecciona una opción (1-3): ").strip()
    
    if choice == "1":
        train_example()
    elif choice == "2":
        predict_example()
    elif choice == "3":
        print("🌐 Iniciando aplicación Streamlit...")
        print("   Ejecuta: streamlit run src/infrastructure/web/streamlit_app.py")
    else:
        print("❌ Opción no válida")

if __name__ == "__main__":
    main()