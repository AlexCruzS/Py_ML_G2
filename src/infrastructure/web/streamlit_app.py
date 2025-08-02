import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
import sys
import os
from io import BytesIO
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.units import mm


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
            .info-card {
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                background-color: #f9fafb;
                padding: 24px;
                margin-bottom: 24px;
            }

            .info-title {
                font-weight: 700;
                font-size: 20px;
                margin-bottom: 12px;
                color: #111827;
                border-bottom: 1px solid #e5e7eb;
                padding-bottom: 8px;
            }

            .info-row {
                display: flex;
                flex-wrap: wrap;
                gap: 32px;
                margin-top: 12px;
            }

            .info-col {
                flex: 1;
                min-width: 220px;
            }

            .info-label {
                font-weight: 600;
                text-transform: uppercase;
                font-size: 12px;
                color: #6b7280;
            }

            .info-value {
                font-size: 15px;
                color: #111827;
                margin-top: 4px;
            }
        </style>
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
                font-size: 16px;
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

            /* Aplicar mismo estilo al botón de descarga */
            .classButton {
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

            .classButton:hover {
                background: linear-gradient(135deg, #5b21b6, #7c3aed) !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
            }

            .classButton:focus {
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
    
    nombre = st.markdown('<label class="custom-label">Cliente:</label>', unsafe_allow_html=True)
    st.text_input(   
        "",
        value='',
        placeholder="Apellidos y nombres",
        key="nombre",
        label_visibility="collapsed"
    )

    direccion = st.markdown('<label class="custom-label">Dirección:</label>', unsafe_allow_html=True)
    st.text_input(   
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
        if documento and not documento.isdigit():
            st.warning("El documento debe contener solo números.")

    with col2:
        st.markdown('<label class="custom-label">Teléfono/móvil:</label>', unsafe_allow_html=True)
        telefono = st.text_input(   
            "", 
            value='',
            placeholder="955555555",
            key="telefono",
            label_visibility="collapsed"
        )
        if telefono and not telefono.isdigit():
            st.warning("El teléfono debe contener solo números.")

    st.markdown('<label class="custom-label">Correo electrónico:</label>', unsafe_allow_html=True)
    correo = st.text_input(   
        "", 
        value='',
        placeholder="user@example.com",
        key="correo",
        label_visibility="collapsed"
    )
    
    if st.button("Continuar"):
        campos_invalidos = []

        if not st.session_state["nombre"]:
            campos_invalidos.append("nombre")
        if not st.session_state["direccion"]:
            campos_invalidos.append("direccion")
        if not st.session_state["documento"]:
            campos_invalidos.append("documento")
        if not st.session_state["telefono"]:
            campos_invalidos.append("telefono")
        if not st.session_state["correo"]:
            campos_invalidos.append("correo")

        if campos_invalidos:
            lista = ", ".join(campos_invalidos)
            st.error(f"❌ Por favor completa correctamente los siguientes campos: {lista}")
        else:
            st.session_state.nombre_guardado = st.session_state.get("nombre", "")
            st.session_state.documento_guardado = st.session_state.get("documento", "")
            st.session_state.direccion_guardado = st.session_state.get("direccion", "")
            st.session_state.telefono_guardado = st.session_state.get("telefono", "")
            st.session_state.correo_guardado = st.session_state.get("correo", "")
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
            key="tipo_propiedad",
            label_visibility="collapsed"
        )

    # Segunda fila: Tipo de Residencia y Área
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<label class="custom-label">Tipo de Residencia:</label>', unsafe_allow_html=True)
        tipo_residencia = st.selectbox(
            "",
            ["Selecciona subtipo", "Single Family", "Condo", "Townhouse"],
            key="tipo_residencia",
            label_visibility="collapsed"
        )

    with col4:
        st.markdown('<label class="custom-label">Área (m²):</label>', unsafe_allow_html=True)
        area_m2 = st.number_input(
            "",
            min_value=0.0,
            value=0.0,
            placeholder="Ejemplo: 240 m²",
            key="area_m2",
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
        st.markdown('<label class="custom-label">Valor fiscal o de autovalúo (€):</label>', unsafe_allow_html=True)
        valor_catastral = st.number_input(
            "",
            min_value=0,
            value=0,
            placeholder="Ejemplo: $ 170,000",
            key="valor_catastral",
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
        if tipo_residencia == "Selecciona subtipo":
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
                st.session_state.ciudad_guardado = st.session_state.get("ciudad", "—")
                st.session_state.tipo_propiedad_guardado = st.session_state.get("tipo_propiedad", "—")
                st.session_state.tipo_residencia_guardado = st.session_state.get("tipo_residencia", "—")
                st.session_state.area_m2_guardado = st.session_state.get("area_m2", "—")
                st.session_state.valor_catastral_guardado = st.session_state.get("valor_catastral", "—")
                st.session_state.currentPage = 3
                st.rerun()


def result_page():
    """Página 3: Resultado de la predicción"""
    show_header()

    if "prediccion" in st.session_state:

        nombre = st.session_state.get("nombre_guardado", "—")
        documento = st.session_state.get("documento_guardado", "—")
        direccion = st.session_state.get("direccion_guardado", "—")
        telefono = st.session_state.get("telefono_guardado", "—")
        correo = st.session_state.get("correo_guardado", "—")

        ciudad = st.session_state.get("ciudad_guardado", "—")
        tipo_propiedad = st.session_state.get("tipo_propiedad_guardado", "—")
        tipo_residencia = st.session_state.get("tipo_residencia_guardado", "—")

        area_m2 = st.session_state.get("area_m2_guardado", "—")
        valor_catastral = st.session_state.get("valor_catastral_guardado", "—")
        fecha_analisis = st.session_state.get("fecha_analisis") or datetime.now().strftime("%d/%m/%Y")

        st.markdown(f"""
            <div class="info-card">
                <div class="info-title">Información del Cliente</div>
                <div class="info-row">
                    <div class="info-col">
                        <div class="info-label">Nombre completo</div>
                        <div class="info-value">{nombre}</div>
                    </div>
                    <div class="info-col">
                        <div class="info-label">Documento de identidad</div>
                        <div class="info-value">{documento}</div>
                    </div>
                    <div class="info-col">
                        <div class="info-label">Dirección del cliente</div>
                        <div class="info-value">{direccion}</div>
                    </div>
                    <div class="info-col">
                        <div class="info-label">Teléfono/móvil</div>
                        <div class="info-value">{telefono}</div>
                    </div>
                    <div class="info-col">
                        <div class="info-label">Correo electrónico</div>
                        <div class="info-value">{correo}</div>
                    </div>
                </div>
            </div>

            <div class="info-card">
                <div class="info-title">Información de la Propiedad</div>
                <div class="info-row">
                    <div class="info-col">
                        <div class="info-label">Ubicación (ciudad)</div>
                        <div class="info-value">{ciudad}</div>
                    </div>
                    <div class="info-col">
                        <div class="info-label">Tipo de propiedad</div>
                        <div class="info-value">{tipo_propiedad}</div>
                    </div>
                    <div class="info-col">
                        <div class="info-label">Tipo de residencia</div>
                        <div class="info-value">{tipo_residencia}</div>
                    </div>
                </div>
            </div>

            <div class="info-card">
                <div class="info-title">Características de la Propiedad</div>
                <div class="info-row">
                    <div class="info-col">
                        <div class="info-label">Superficie total</div>
                        <div class="info-value"> {area_m2} m²</div>
                    </div>
                    <div class="info-col">
                        <div class="info-label">Valor fiscal</div>
                        <div class="info-value">€ {valor_catastral}</div>
                    </div>
                    <div class="info-col">
                        <div class="info-label">Fecha de análisis</div>
                        <div class="info-value">{fecha_analisis}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        valor_estimado = st.session_state.prediccion
        valor_fiscal = st.session_state.valor_catastral_guardado
        area_m2 = st.session_state.area_m2_guardado

        prima = valor_estimado - valor_fiscal
        ratio = valor_estimado / valor_fiscal if valor_fiscal else 0
        precio_m2 = valor_estimado / area_m2 if area_m2 else 0

        st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #7e5bef, #5f46e5);
                padding: 32px;
                border-radius: 16px;
                color: white;
                text-align: center;
                margin-top: 20px;
                box-shadow: 0 10px 20px rgba(0,0,0,0.15);
            ">
                <div style="font-size: 18px; font-weight: 500; margin-bottom: 10px;">
                    Valor Estimado de Mercado
                </div>
                <div style="font-size: 48px; font-weight: 700; margin-bottom: 30px;">
                    €{valor_estimado:,.0f}
                </div>
                <div style="display: flex; justify-content: center; gap: 24px;">
                    <div style="
                        background-color: rgba(255, 255, 255, 0.15);
                        padding: 16px 20px;
                        border-radius: 12px;
                        flex: 1;
                        max-width: 200px;
                    ">
                        <div style="font-size: 20px; font-weight: 600;">
                            €{prima:,.0f}
                        </div>
                        <div style="font-size: 14px;">Prima sobre Valor Fiscal</div>
                    </div>
                    <div style="
                        background-color: rgba(255, 255, 255, 0.15);
                        padding: 16px 20px;
                        border-radius: 12px;
                        flex: 1;
                        max-width: 200px;
                    ">
                        <div style="font-size: 20px; font-weight: 600;">
                            {ratio:.2f}
                        </div>
                        <div style="font-size: 14px;">Ratio Mercado/Fiscal</div>
                    </div>
                    <div style="
                        background-color: rgba(255, 255, 255, 0.15);
                        padding: 16px 20px;
                        border-radius: 12px;
                        flex: 1;
                        max-width: 200px;
                    ">
                        <div style="font-size: 20px; font-weight: 600;">
                            €{precio_m2:,.0f}
                        </div>
                        <div style="font-size: 14px;">Precio por m²</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Botones de navegación
        col1, col2, col3 = st.columns([4,4,0.8])
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

        with col3:
            st.markdown('<div style="margin-top: 32px;">', unsafe_allow_html=True)
            st.download_button(
                label="📄",
                help="Exportar predicción a PDF",
                data=generar_pdf(),
                file_name="prediccion.pdf",
                mime="application/pdf",
            )
            st.markdown('</div>', unsafe_allow_html=True)


def generar_pdf():
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER

    # Márgenes y fuentes
    margin_x = 40
    current_y = height - 40
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(colors.HexColor("#5f46e5"))
    c.drawString(margin_x, current_y, "🧾 Informe de Predicción Inmobiliaria")

    # Cliente
    current_y -= 30
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_x, current_y, "👤 Información del Cliente")
    c.setFont("Helvetica", 10)
    current_y -= 18
    c.drawString(margin_x, current_y, f"Nombre: {st.session_state.get('nombre_guardado', '')}")
    current_y -= 14
    c.drawString(margin_x, current_y, f"Documento: {st.session_state.get('documento_guardado', '')}")
    current_y -= 14
    c.drawString(margin_x, current_y, f"Dirección: {st.session_state.get('direccion_guardado', '')}")
    current_y -= 14
    c.drawString(margin_x, current_y, f"Teléfono/Móvil: {st.session_state.get('telefono_guardado', '')}")
    current_y -= 14
    c.drawString(margin_x, current_y, f"Correo: {st.session_state.get('correo_guardado', '')}")

    # Propiedad
    current_y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_x, current_y, "🏠 Información de la Propiedad")
    c.setFont("Helvetica", 10)
    current_y -= 18
    c.drawString(margin_x, current_y, f"Ciudad: {st.session_state.get('ciudad', '')}")
    current_y -= 14
    c.drawString(margin_x, current_y, f"Tipo de Propiedad: {st.session_state.get('tipo_propiedad', '')}")
    current_y -= 14
    c.drawString(margin_x, current_y, f"Tipo de Residencia: {st.session_state.get('tipo_residencia', '')}")

    # Características
    area_m2 = st.session_state.get("area_m2_guardado", 0)
    valor_fiscal = st.session_state.get("valor_catastral_guardado", 0)

    try:
        area_m2 = float(area_m2)
    except (TypeError, ValueError):
        area_m2 = 0

    try:
        valor_fiscal = float(valor_fiscal)
    except (TypeError, ValueError):
        valor_fiscal = 0

    fecha_analisis = st.session_state.get("fecha_analisis", datetime.now().strftime('%d/%m/%Y'))

    current_y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_x, current_y, "📐 Características de la Propiedad")
    c.setFont("Helvetica", 10)
    current_y -= 18
    c.drawString(margin_x, current_y, f"Área (m²): {area_m2}")
    current_y -= 14
    c.drawString(margin_x, current_y, f"Valor Fiscal: €{valor_fiscal:,.0f}")
    current_y -= 14
    c.drawString(margin_x, current_y, f"Fecha de Análisis: {fecha_analisis}")

    # Resultados
    valor_estimado = st.session_state.get("prediccion", 0)
    try:
        valor_estimado = float(valor_estimado)
    except (TypeError, ValueError):
        valor_estimado = 0

    prima = valor_estimado - valor_fiscal
    ratio = valor_estimado / valor_fiscal if valor_fiscal else 0
    precio_m2 = valor_estimado / area_m2 if area_m2 else 0

    current_y -= 40
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(colors.HexColor("#5f46e5"))
    c.drawString(margin_x, current_y, "📊 Valoración del Mercado")

    # Caja principal con valor estimado
    current_y -= 40
    box_width = 180
    box_height = 70
    box_x = margin_x
    c.setFillColor(colors.HexColor("#7e5bef"))
    c.roundRect(box_x, current_y, box_width, box_height, 10, fill=True, stroke=False)

    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.white)
    c.drawCentredString(box_x + box_width/2, current_y + 50, "Valor Estimado")
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(box_x + box_width/2, current_y + 28, f"€{valor_estimado:,.0f}")

    # Tarjetas secundarias (prima, ratio, precio/m2)
    small_w = 140
    small_h = 45
    spacing = 20
    small_y = current_y - 10 - small_h
    small_start_x = margin_x

    card_color = colors.HexColor("#9374eb")
    text_color = colors.white

    def draw_small_box(x, label, value):
        c.setFillColor(card_color)
        c.roundRect(x, small_y, small_w, small_h, 8, fill=True, stroke=False)
        c.setFillColor(text_color)
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(x + small_w/2, small_y + 28, value)
        c.setFont("Helvetica", 9)
        c.drawCentredString(x + small_w/2, small_y + 12, label)

    draw_small_box(small_start_x, "Prima sobre Valor Fiscal", f"€{prima:,.0f}")
    draw_small_box(small_start_x + small_w + spacing, "Ratio Mercado/Fiscal", f"{ratio:.2f}")
    draw_small_box(small_start_x + 2*(small_w + spacing), "Precio por m²", f"€{precio_m2:,.0f}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

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
                
                # Métricas de Performance del Modelo
                st.subheader("📊 Métricas de Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    rmse_value = f"${result.rmse:,.0f}" if hasattr(result, 'rmse') and result.rmse else "N/A"
                    st.metric("🎯 RMSE", rmse_value, help="Root Mean Square Error - Menor es mejor")
                with col2:
                    # Calcular MAE si no existe en el resultado
                    if hasattr(result, 'mae') and result.mae is not None:
                        mae_value = f"${result.mae:,.0f}"
                    else:
                        # Estimar MAE como ~0.7 * RMSE (aproximación típica)
                        mae_estimated = result.rmse * 0.7 if hasattr(result, 'rmse') else 0
                        mae_value = f"${mae_estimated:,.0f}*"
                    st.metric("📏 MAE", mae_value, help="Mean Absolute Error - Menor es mejor (*Estimado)")
                with col3:
                    # Calcular R² si no existe en el resultado
                    if hasattr(result, 'r2_score') and result.r2_score is not None:
                        r2_value = f"{result.r2_score:.4f}"
                    else:
                        # Estimar R² basado en RMSE (aproximación para inmobiliaria)
                        if hasattr(result, 'rmse') and result.rmse:
                            if result.rmse < 80000:
                                r2_estimated = 0.85
                            elif result.rmse < 120000:
                                r2_estimated = 0.75
                            elif result.rmse < 180000:
                                r2_estimated = 0.65
                            else:
                                r2_estimated = 0.55
                            r2_value = f"{r2_estimated:.3f}*"
                        else:
                            r2_value = "N/A"
                    st.metric("📈 R² Score", r2_value, help="Coeficiente de Determinación - Más cercano a 1 es mejor (*Estimado)")
                
                # Parámetros del Modelo
                st.subheader("⚙️ Configuración del Modelo")
                col4, col5, col6 = st.columns(3)
                with col4:
                    st.metric("🌲 Estimadores", result.n_estimators)
                with col5:
                    st.metric("📏 Profundidad", result.max_depth)
                with col6:
                    # Calcular precisión basada en R² disponible o estimado
                    if hasattr(result, 'r2_score') and result.r2_score is not None:
                        accuracy_pct = f"{result.r2_score*100:.1f}%"
                    else:
                        # Usar R² estimado del cálculo anterior
                        if hasattr(result, 'rmse') and result.rmse:
                            if result.rmse < 80000:
                                estimated_r2 = 0.85
                            elif result.rmse < 120000:
                                estimated_r2 = 0.75
                            elif result.rmse < 180000:
                                estimated_r2 = 0.65
                            else:
                                estimated_r2 = 0.55
                            accuracy_pct = f"{estimated_r2*100:.1f}%*"
                        else:
                            accuracy_pct = "N/A"
                    st.metric("✅ Precisión", accuracy_pct, help="Basado en R² Score (*Estimado si no disponible)")
                
                                
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
    
    for key in ['nombre', 'documento', 'direccion', 'ciudad']:
        if key not in st.session_state:
            st.session_state[key] = ""

    if "valor_catastral" not in st.session_state:
        st.session_state["valor_catastral"] = 0

    if "area_m2" not in st.session_state:
        st.session_state["area_m2"] = 0

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