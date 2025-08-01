import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
import sys
import os

# Agregar el directorio src al path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from application.dto.property_dto import PropertyInputDTO
from application.use_cases.train_model import TrainModelUseCase
from application.use_cases.predict_price import PredictPriceUseCase
from infrastructure.data.data_loader import CSVDataLoader
from infrastructure.ml.mlflow_repository import MLflowModelRepository
from infrastructure.ml.model_trainer import RealEstateModelTrainer, RealEstatePredictionService

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Precios Inmobiliarios",
    page_icon="🏠",
    layout="wide"
)

# Configurar dependencias (Inyección de dependencias manual)
@st.cache_resource
def setup_dependencies():
    """Configura las dependencias de la aplicación."""
    data_repository = CSVDataLoader()
    model_repository = MLflowModelRepository()
    training_service = RealEstateModelTrainer(data_repository, model_repository)
    prediction_service = RealEstatePredictionService(model_repository)
    
    train_use_case = TrainModelUseCase(data_repository, model_repository, training_service)
    predict_use_case = PredictPriceUseCase(prediction_service, model_repository)
    
    return train_use_case, predict_use_case, data_repository

def main():
    """Función principal de la aplicación Streamlit."""
    
    st.title("🏠 Predictor de Precios Inmobiliarios")
    st.markdown("### Sistema de ML con Arquitectura Hexagonal")
    
    # Configurar dependencias
    train_use_case, predict_use_case, data_repository = setup_dependencies()
    
    # Sidebar para navegación
    st.sidebar.title("Navegación")
    page = st.sidebar.radio(
        "Selecciona una página:",
        ["🔮 Predicción", "🎯 Entrenamiento", "📊 Análisis de Datos"]
    )
    
    if page == "🔮 Predicción":
        prediction_page(predict_use_case)
    elif page == "🎯 Entrenamiento":
        training_page(train_use_case)
    elif page == "📊 Análisis de Datos":
        data_analysis_page(data_repository)

def prediction_page(predict_use_case):
    """Página de predicción de precios."""
    
    st.header("Predicción de Precios")
    st.markdown("Ingresa los datos de la propiedad para obtener una predicción de precio.")
    
    # Formulario de entrada
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            assessed_value = st.number_input(
                "Valor Tasado ($)",
                min_value=1000,
                max_value=10000000,
                value=300000,
                step=1000
            )
            
            area_m2 = st.number_input(
                "Área (m²)",
                min_value=10.0,
                max_value=2000.0,
                value=150.0,
                step=5.0
            )
            
            meses_en_venta = st.number_input(
                "Meses en Venta",
                min_value=0,
                max_value=60,
                value=6,
                step=1
            )
        
        with col2:
            nro_habitaciones = st.number_input(
                "Número de Habitaciones",
                min_value=1,
                max_value=20,
                value=3,
                step=1
            )
            
            nro_pisos = st.number_input(
                "Número de Pisos",
                min_value=1,
                max_value=10,
                value=2,
                step=1
            )
            
            property_type = st.selectbox(
                "Tipo de Propiedad",
                ["Single Family", "Residential", "Condo", "Two Family", "Three Family", "Four Family"]
            )
        
        submit_button = st.form_submit_button("🔮 Predecir Precio")
        
        if submit_button:
            try:
                # Crear DTO
                property_input = PropertyInputDTO(
                    assessed_value=assessed_value,
                    area_m2=area_m2,
                    meses_en_venta=meses_en_venta,
                    nro_habitaciones=nro_habitaciones,
                    nro_pisos=nro_pisos,
                    property_type=property_type
                )
                
                # Realizar predicción
                result = predict_use_case.execute(property_input)
                
                # Mostrar resultado
                st.success("¡Predicción realizada exitosamente!")
                
                col1, col2, col3 = st.columns(3)
                with col2:
                    st.metric(
                        label="Precio Predicho",
                        value=f"${result.predicted_price:,.2f}",
                        delta=f"Modelo: {result.model_version}"
                    )
                
                # Mostrar detalles adicionales
                with st.expander("Detalles de la Predicción"):
                    st.json({
                        "Precio Predicho": f"${result.predicted_price:,.2f}",
                        "Valor Tasado": f"${assessed_value:,.2f}",
                        "Diferencia": f"${result.predicted_price - assessed_value:,.2f}",
                        "% Diferencia": f"{((result.predicted_price - assessed_value) / assessed_value) * 100:.1f}%"
                    })
                
            except Exception as e:
                st.error(f"Error en la predicción: {str(e)}")

def training_page(train_use_case):
    """Página de entrenamiento de modelos."""
    
    st.header("Entrenamiento de Modelos")
    st.markdown("Entrena nuevos modelos con diferentes hiperparámetros.")
    
    # Formulario de entrenamiento
    with st.form("training_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Sube tu archivo CSV",
                type=['csv'],
                help="Archivo con datos de propiedades inmobiliarias"
            )
            
            n_estimators = st.slider(
                "Número de Estimadores",
                min_value=50,
                max_value=500,
                value=100,
                step=50
            )
        
        with col2:
            max_depth = st.slider(
                "Profundidad Máxima",
                min_value=3,
                max_value=20,
                value=5,
                step=1
            )
            
            experiment_name = st.text_input(
                "Nombre del Experimento",
                value="Grupo_2_Proyecto_Inmobiliario"
            )
        
        train_button = st.form_submit_button("🎯 Entrenar Modelo")
        
        if train_button:
            if uploaded_file is not None:
                try:
                    # Guardar archivo temporalmente
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Mostrar progreso
                    with st.spinner("Entrenando modelo..."):
                        result = train_use_case.execute(
                            file_path=tmp_file_path,
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            experiment_name=experiment_name
                        )
                    
                    # Mostrar resultados
                    st.success("¡Modelo entrenado exitosamente!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RMSE", f"{result.rmse:,.2f}")
                    with col2:
                        st.metric("Estimadores", result.n_estimators)
                    with col3:
                        st.metric("Profundidad", result.max_depth)
                    
                    # Mostrar información adicional
                    with st.expander("Información del Entrenamiento"):
                        st.code(f"""
Experiment ID: {result.experiment_id}
Run ID: {result.run_id}
Model URI: {result.model_uri}
                        """)
                    
                    # Limpiar archivo temporal
                    os.unlink(tmp_file_path)
                    
                except Exception as e:
                    st.error(f"Error en el entrenamiento: {str(e)}")
            else:
                st.warning("Por favor, sube un archivo CSV para entrenar el modelo.")

def data_analysis_page(data_repository):
    """Página de análisis de datos."""
    
    st.header("Análisis de Datos")
    st.markdown("Explora y analiza los datos de propiedades inmobiliarias.")
    
    uploaded_file = st.file_uploader(
        "Sube tu archivo CSV para análisis",
        type=['csv'],
        key="analysis_file"
    )
    
    if uploaded_file is not None:
        try:
            # Cargar datos
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            df = data_repository.load_data(tmp_file_path)
            
            # Mostrar información básica
            st.subheader("Información General")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Registros", len(df))
            with col2:
                st.metric("Columnas", len(df.columns))
            with col3:
                st.metric("Valores Nulos", df.isnull().sum().sum())
            with col4:
                st.metric("Memoria (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f}")
            
            # Mostrar primeras filas
            st.subheader("Vista Previa de Datos")
            st.dataframe(df.head())
            
            # Estadísticas descriptivas
            st.subheader("Estadísticas Descriptivas")
            st.dataframe(df.describe())
            
            # Gráficos
            if 'Sale Amount' in df.columns:
                st.subheader("Visualizaciones")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histograma de precios
                    fig_hist = px.histogram(
                        df, 
                        x='Sale Amount', 
                        title="Distribución de Precios de Venta",
                        nbins=50
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Scatter plot área vs precio
                    if 'area_m2' in df.columns:
                        # Limpiar datos de área
                        df_clean = df.copy()
                        if df_clean['area_m2'].dtype == 'object':
                            df_clean['area_m2'] = df_clean['area_m2'].str.replace('m2', '').astype(float)
                        
                        fig_scatter = px.scatter(
                            df_clean, 
                            x='area_m2', 
                            y='Sale Amount',
                            title="Área vs Precio de Venta"
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Limpiar archivo temporal
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"Error al analizar los datos: {str(e)}")

if __name__ == "__main__":
    main()