import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
import sys
import os

# Configurar el path para imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, src_dir)

# Imports del proyecto
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
    layout="centered"  # Cambiado de "wide" a "centered" para mejor diseño
)

def load_custom_css():
    """Carga los estilos CSS personalizados desde app1.py"""
    st.markdown("""
        <style>
            /* Importar fuentes */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
            /* Resetear estilos de Streamlit */
            .stApp {
                min-height: 100vh;
                font-family: 'Inter', sans-serif;
            }
            
            /* Ocultar elementos de Streamlit */
            .stDeployButton { display: none; }
            #MainMenu { visibility: hidden; }
            header { visibility: hidden; }
            footer { visibility: hidden; }
            .stToolbar { display: none; }

            /* Contenedor principal */
            .main-container {
                background: #ffffff;
                border-radius: 24px;
                padding: 32px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
                max-width: 480px;
                margin: 40px auto;
                border: 3px solid #6366f1;
                position: relative;
            }

            /* Header exacto de la imagen */
            .header-container {
                display: flex;
                align-items: center;
                margin-bottom: 8px;
            }

            .house-icon {
                width: 48px;
                height: 48px;
                background: #1f2937;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 16px;
                color: white;
                font-size: 24px;
            }

            .main-title {
                text-align: center;
                font-size: 24px;
                font-weight: 700;
                color: #1f2937;
                margin: 0;
                line-height: 1.2;
            }

            .subtitle {
                color: white;
                font-size: 20px;
                text-align: center;
                margin-bottom: 32px;
                font-weight: 400;
            }

            /* Estilos para labels exactos de la imagen */
            .custom-label {
                font-size: 14px;
                font-weight: 600;
                color: #374151;
                margin-bottom: 8px;
                display: block;
            }

            /* Estilos para selectbox de Streamlit */
            .stSelectbox > div > div {
                background: #f9fafb !important;
                border: 1px solid #d1d5db !important;
                border-radius: 12px !important;
                min-height: 48px !important;
            }

            .stSelectbox > div > div > div {
                padding: 12px 16px !important;
                font-size: 14px !important;
                color: #374151 !important;
            }

            /* Estilos para number_input de Streamlit */
            .stNumberInput > div > div {
                background: #f9fafb !important;
                border-radius: 12px !important;
            }

            .stNumberInput > div > div > input {
                background: #f9fafb !important;
                border: 1px solid #d1d5db !important;
                border-radius: 12px !important;
                padding: 12px 16px !important;
                font-size: 14px !important;
                color: #374151 !important;
                min-height: 48px !important;
            }

            .stTextInput > div > div > input {
                background: #f9fafb !important;
                border: 1px solid #d1d5db !important;
                border-radius: 12px !important;
                padding: 12px 16px !important;
                font-size: 14px !important;
                color: #374151 !important;
                min-height: 48px !important;
            }

            .stNumberInput > div > div > input:focus,
            .stTextInput > div > div > input:focus {
                border-color: #6366f1 !important;
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
                outline: none !important;
            }

            .stSelectbox > div > div:focus-within {
                border-color: #6366f1 !important;
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
            }

            /* Ocultar labels por defecto de Streamlit */
            .stSelectbox > label,
            .stNumberInput > label,
            .stTextInput > label {
                display: none !important;
            }

            /* Botón exacto de la imagen */
            .stButton > button {
                width: 100% !important;
                background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
                color: white !important;
                border: none !important;
                padding: 16px 24px !important;
                border-radius: 12px !important;
                font-size: 16px !important;
                font-weight: 600 !important;
                margin-top: 24px !important;
                min-height: 52px !important;
                transition: all 0.2s ease !important;
            }

            .stButton > button:hover {
                background: linear-gradient(135deg, #5b21b6, #7c3aed) !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
            }

            .stButton > button:focus {
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3) !important;
            }

            /* Resultado */
            .result-container {
                margin-top: 24px;
                padding: 20px;
                background: linear-gradient(135deg, #10b981, #34d399);
                border-radius: 12px;
                text-align: center;
                color: white;
                font-size: 18px;
                font-weight: 600;
                animation: slideIn 0.3s ease;
            }

            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            /* Estilos para mensajes de error */
            .stAlert {
                margin-top: 16px;
            }

            /* Navegación del sidebar con estilo */
            .css-1d391kg {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
            }

            /* Responsive */
            @media (max-width: 640px) {
                .main-container {
                    margin: 20px;
                    padding: 24px;
                    max-width: calc(100% - 40px);
                }

                .main-title {
                    font-size: 20px;
                }
            }
        </style>
    """, unsafe_allow_html=True)

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

def show_header():
    """Muestra el header principal con el diseño de app1.py"""
    st.markdown("""
        <div class="header-container">
            <div class="house-icon">🏠</div>
            <h1 class="main-title">Predictor de Precios Inmobiliarios</h1>
        </div>
        <p class="subtitle">Obtén una estimación precisa del valor de mercado de tu propiedad</p>
    """, unsafe_allow_html=True)

def client_info_page():
    """Página 1: Información del cliente (del diseño app1.py)"""
    show_header()
    
    st.markdown('<label class="custom-label">Cliente:</label>', unsafe_allow_html=True)
    nombre = st.text_input(   
        "", 
        value='',
        placeholder="Apellidos y nombres",
        key="nombre",
        label_visibility="collapsed"
    )

    st.markdown('<label class="custom-label">Dirección:</label>', unsafe_allow_html=True)
    direccion = st.text_input(   
        "", 
        value='',
        placeholder="Urb, dpto, piso",
        key="direccion",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<label class="custom-label">Documento de identidad:</label>', unsafe_allow_html=True)
        documento = st.text_input(   
            "", 
            value='',
            placeholder="9999999",
            key="documento",
            label_visibility="collapsed"
        )

    with col2:
        st.markdown('<label class="custom-label">Teléfono/móvil:</label>', unsafe_allow_html=True)
        telefono = st.text_input(   
            "", 
            value='',
            placeholder="955555555",
            key="telefono",
            label_visibility="collapsed"
        )

    st.markdown('<label class="custom-label">Correo electrónico:</label>', unsafe_allow_html=True)
    correo = st.text_input(   
        "", 
        value='',
        placeholder="user@example.com",
        key="correo",
        label_visibility="collapsed"
    )
    
    if st.button("Continuar"):
        st.session_state.currentPage = 2
        st.rerun()

def property_info_page(predict_use_case):
    """Página 2: Información de la propiedad con integración hexagonal"""
    show_header()
    
    # Primera fila: Ciudad y Tipo de Propiedad
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<label class="custom-label">Ciudad:</label>', unsafe_allow_html=True)
        ciudad = st.selectbox(
            "", 
            [
                "Selecciona una ciudad",
                "Windham", "Montville", "Lisbon", "Wilton", "Westport", 
                "Enfield", "Waterford", "New Canaan", "Bozrah", "Portland",
                "Rocky Hill", "Southington", "Bridgeport", "East Lyme", 
                "Naugatuck", "Essex", "Greenwich"
            ],
            key="ciudad",
            label_visibility="collapsed"
        )

    with col2:
        st.markdown('<label class="custom-label">Tipo de Propiedad:</label>', unsafe_allow_html=True)
        tipo_propiedad = st.selectbox(
            "",
            ["Selecciona tipo", "Single Family", "Residential", "Condo", "Two Family", "Three Family", "Four Family"],
            key="tipo_prop",
            label_visibility="collapsed"
        )

    # Segunda fila: Tipo de Residencia y Área
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<label class="custom-label">Tipo de Residencia:</label>', unsafe_allow_html=True)
        tipo_residencial = st.selectbox(
            "",
            ["Selecciona subtipo", "Single Family", "Condo", "Townhouse"],
            key="tipo_res",
            label_visibility="collapsed"
        )

    with col4:
        st.markdown('<label class="custom-label">Área (m²):</label>', unsafe_allow_html=True)
        area_m2 = st.number_input(
            "",
            min_value=0.0,
            value=0.0,
            placeholder="Ejemplo: 240 m²",
            key="area",
            label_visibility="collapsed"
        )

    # Tercera fila: Habitaciones y Valor fiscal
    col5, col6 = st.columns(2)

    with col5:
        st.markdown('<label class="custom-label">Cantidad de habitaciones:</label>', unsafe_allow_html=True)
        habitaciones = st.number_input(
            "",
            min_value=0,
            value=0,
            placeholder="Ejemplo: 4",
            key="habitaciones",
            label_visibility="collapsed"
        )

    with col6:
        st.markdown('<label class="custom-label">Valor fiscal o de autovalúo ($):</label>', unsafe_allow_html=True)
        valor_catastral = st.number_input(
            "",
            min_value=0,
            value=0,
            placeholder="Ejemplo: $ 170,000",
            key="valor",
            label_visibility="collapsed"
        )

    # Campos adicionales
    col7, col8 = st.columns(2)
    
    with col7:
        st.markdown('<label class="custom-label">Meses en venta:</label>', unsafe_allow_html=True)
        meses_en_venta = st.number_input(
            "",
            min_value=0,
            value=6,
            key="meses_venta",
            label_visibility="collapsed"
        )
    
    with col8:
        st.markdown('<label class="custom-label">Número de pisos:</label>', unsafe_allow_html=True)
        pisos = st.number_input(
            "",
            min_value=1,
            value=2,
            key="pisos",
            label_visibility="collapsed"
        )

    # Botón de predicción
    if st.button("Predecir Precio de Mercado"):
        # Validaciones
        campos_invalidos = []

        if ciudad == "Selecciona una ciudad":
            campos_invalidos.append("ciudad")
        if tipo_propiedad == "Selecciona tipo":
            campos_invalidos.append("tipo de propiedad")
        if tipo_residencial == "Selecciona subtipo":
            campos_invalidos.append("tipo de residencia")
        if area_m2 <= 0:
            campos_invalidos.append("área en m²")
        if habitaciones <= 0:
            campos_invalidos.append("número de habitaciones")
        if valor_catastral <= 0:
            campos_invalidos.append("valor fiscal")

        if campos_invalidos:
            lista = ", ".join(campos_invalidos)
            st.error(f"❌ Por favor completa correctamente los siguientes campos: {lista}")
        else:
            try:
                with st.spinner('🤖 Analizando propiedades similares...'):
                    # Usar la arquitectura hexagonal correcta
                    property_input = PropertyInputDTO(
                        assessed_value=float(valor_catastral),
                        area_m2=float(area_m2),
                        meses_en_venta=int(meses_en_venta),
                        nro_habitaciones=int(habitaciones),
                        nro_pisos=int(pisos),
                        property_type=tipo_propiedad
                    )
                    
                    # Realizar predicción usando el caso de uso
                    result = predict_use_case.execute(property_input)
                    
                    # Guardar resultado en session_state
                    st.session_state.prediccion = result.predicted_price
                    st.session_state.model_version = result.model_version
                    st.session_state.currentPage = 3
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error en la predicción: {str(e)}")
                
                # Fallback con estimación aproximada
                st.warning("⚠️ Usando estimación aproximada...")
                precio_base = area_m2 * 2500
                factor_habitaciones = habitaciones * 25000
                factor_ubicacion = 50000 if ciudad == "Portland" else 30000
                factor_valor_catastral = valor_catastral * 1.2
                
                resultado_simulado = int((precio_base + factor_habitaciones + 
                                    factor_ubicacion + factor_valor_catastral) / 2)
                
                st.session_state.prediccion = resultado_simulado
                st.session_state.model_version = "Estimación Aproximada"
                st.session_state.currentPage = 3
                st.rerun()

def result_page():
    """Página 3: Resultado de la predicción"""
    show_header()
    
    if "prediccion" in st.session_state:
        st.markdown(f"""
            <div class="result-container">
                💰 Monto estimado de venta: ${st.session_state.prediccion:,.0f}
            </div>
        """, unsafe_allow_html=True)
        
        # Información adicional
        with st.expander("ℹ️ Detalles de la predicción"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Modelo utilizado:** {st.session_state.get('model_version', 'N/A')}")
                st.write(f"**Cliente:** {st.session_state.get('nombre', 'N/A')}")
            with col2:
                st.write(f"**Valor tasado:** ${st.session_state.get('valor', 0):,.0f}")
                st.write(f"**Área:** {st.session_state.get('area', 0):.1f} m²")
        
        # Información sobre el modelo
        with st.expander("ℹ️ Información sobre el modelo"):
            st.write("""
            **Características del modelo:**
            - Utiliza algoritmos de machine learning entrenados con datos inmobiliarios reales
            - Considera múltiples factores: ubicación, área, habitaciones, valor catastral
            - La predicción se basa en propiedades similares en el mercado
            - Integrado con MLflow para tracking y versionado de modelos
            
            **Nota:** Esta es una estimación basada en datos históricos. 
            El precio real puede variar según condiciones actuales del mercado.
            """)
        
        # Botones de navegación
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Nueva Predicción"):
                # Limpiar datos pero mantener info del cliente
                for key in ['prediccion', 'model_version']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.currentPage = 2
                st.rerun()
        
        with col2:
            if st.button("👤 Nuevo Cliente"):
                # Limpiar todos los datos
                for key in list(st.session_state.keys()):
                    if key != 'currentPage':
                        del st.session_state[key]
                st.session_state.currentPage = 1
                st.rerun()

def admin_page(train_use_case, data_repository):
    """Página de administración (entrenamiento y análisis)"""
    st.title("🔧 Panel de Administración")
    
    tab1, tab2 = st.tabs(["🎯 Entrenamiento", "📊 Análisis de Datos"])
    
    with tab1:
        st.header("Entrenamiento de Modelos")
        training_page(train_use_case)
    
    with tab2:
        st.header("Análisis de Datos")
        data_analysis_page(data_repository)

def training_page(train_use_case):
    """Página de entrenamiento (funcionalidad original)"""
    with st.form("training_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Sube tu archivo CSV",
                type=['csv']
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
        
        if train_button and uploaded_file is not None:
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                with st.spinner("Entrenando modelo..."):
                    result = train_use_case.execute(
                        file_path=tmp_file_path,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        experiment_name=experiment_name
                    )
                
                st.success("¡Modelo entrenado exitosamente!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"{result.rmse:,.2f}")
                with col2:
                    st.metric("Estimadores", result.n_estimators)
                with col3:
                    st.metric("Profundidad", result.max_depth)
                
                os.unlink(tmp_file_path)
                
            except Exception as e:
                st.error(f"Error en el entrenamiento: {str(e)}")

def data_analysis_page(data_repository):
    """Página de análisis de datos (funcionalidad original)"""
    uploaded_file = st.file_uploader(
        "Sube tu archivo CSV para análisis",
        type=['csv'],
        key="analysis_file"
    )
    
    if uploaded_file is not None:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            df = data_repository.load_data(tmp_file_path)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Registros", len(df))
            with col2:
                st.metric("Columnas", len(df.columns))
            with col3:
                st.metric("Valores Nulos", df.isnull().sum().sum())
            with col4:
                st.metric("Memoria (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f}")
            
            st.subheader("Vista Previa de Datos")
            st.dataframe(df.head())
            
            st.subheader("Estadísticas Descriptivas")
            st.dataframe(df.describe())
            
            if 'Sale Amount' in df.columns:
                st.subheader("Visualizaciones")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hist = px.histogram(
                        df, 
                        x='Sale Amount', 
                        title="Distribución de Precios de Venta",
                        nbins=50
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    if 'area_m2' in df.columns:
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
            
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"Error al analizar los datos: {str(e)}")

def main():
    """Función principal mejorada con el diseño de app1.py"""
    
    # Cargar estilos CSS personalizados
    load_custom_css()
    
    # Configurar dependencias
    train_use_case, predict_use_case, data_repository = setup_dependencies()
    
    # Inicializar state de página
    if "currentPage" not in st.session_state:
        st.session_state.currentPage = 1
    
    # Sidebar para navegación adicional
    with st.sidebar:
        st.title("🏠 Navegación")
        
        mode = st.radio(
            "Modo de aplicación:",
            ["👤 Cliente", "🔧 Administrador"]
        )
        
        if mode == "🔧 Administrador":
            st.session_state.currentPage = "admin"
        elif st.session_state.currentPage == "admin":
            st.session_state.currentPage = 1
        
        # Mostrar progreso para modo cliente
        if mode == "👤 Cliente" and st.session_state.currentPage != "admin":
            st.markdown("### Progreso")
            progress_steps = ["Datos del Cliente", "Datos de Propiedad", "Resultado"]
            current_step = min(st.session_state.currentPage, 3)
            
            for i, step in enumerate(progress_steps, 1):
                if i == current_step:
                    st.markdown(f"**{i}. {step}** ✅")
                elif i < current_step:
                    st.markdown(f"{i}. {step} ✅")
                else:
                    st.markdown(f"{i}. {step}")
    
    # Renderizar página según el estado
    if st.session_state.currentPage == "admin":
        admin_page(train_use_case, data_repository)
    elif st.session_state.currentPage == 1:
        client_info_page()
    elif st.session_state.currentPage == 2:
        property_info_page(predict_use_case)
    elif st.session_state.currentPage == 3:
        result_page()

if __name__ == "__main__":
    main()