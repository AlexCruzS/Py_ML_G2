# 🏠 Py_ML_G2 - Predictor de Precios Inmobiliarios

Sistema de Machine Learning para predicción de precios inmobiliarios desarrollado con **arquitectura hexagonal**, **MLflow** para tracking y **Streamlit** para la interfaz web.

## 🏗️ Arquitectura

El proyecto implementa **Arquitectura Hexagonal** (Ports & Adapters) para separar la lógica de negocio de los detalles técnicos:

```
src/
├── domain/                 # Lógica de negocio pura
│   ├── entities/          # Entidades de dominio
│   ├── repositories/      # Interfaces de repositorios
│   └── services/          # Servicios de dominio
├── application/           # Casos de uso
│   ├── dto/              # Data Transfer Objects
│   └── use_cases/        # Casos de uso específicos
└── infrastructure/       # Implementaciones técnicas
    ├── data/             # Carga y procesamiento de datos
    ├── ml/               # MLflow y entrenamiento
    └── web/              # Interfaz Streamlit
```

## 🚀 Características

- **Predicción de precios** basada en características de propiedades
- **Entrenamiento de modelos** RandomForest con MLflow tracking
- **Interfaz web interactiva** con Streamlit
- **Análisis exploratorio** de datos integrado
- **Gestión de experimentos** con MLflow
- **Arquitectura limpia** y mantenible

## 📋 Requisitos

- Python 3.8+
- MLflow Server (para tracking)
- Streamlit (para interfaz web)

## 🛠️ Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/Py_ML_G2.git
cd Py_ML_G2
```

### 2. Crear entorno virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
```bash
# Crear archivo .env (ya existe plantilla)
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=Grupo_2_Proyecto_Inmobiliario
DATA_PATH=data/
MODELS_PATH=models/
```

## 🎯 Uso del Sistema

### Opción 1: Interfaz Web (Recomendado)

1. **Iniciar MLflow Server** (Terminal 1):
```bash
mlflow server --host 127.0.0.1 --port 5000
```

2. **Iniciar aplicación Streamlit** (Terminal 2):
```bash
streamlit run src/infrastructure/web/streamlit_app.py
```

3. **Acceder a la aplicación**:
   - Streamlit: http://localhost:8501
   - MLflow UI: http://localhost:5000

### Opción 2: Dockerización
- El proyecto ha sido dockerizado para lograr caracteristicas de portabilidad y podamos llevarlo a los ambientes de pruebas y producción en forma rapida y segura.

## 📊 Funcionalidades de la Interfaz Web

### 🔮 Predicción
- Formulario interactivo para ingresar datos de propiedades
- Predicción en tiempo real de precios
- Comparación con valor tasado
- Visualización de resultados

### 🎯 Entrenamiento
- Carga de archivos CSV personalizados
- Configuración de hiperparámetros
- Seguimiento automático con MLflow
- Métricas de evaluación en tiempo real

### 📊 Análisis de Datos
- Exploración automática de datasets
- Estadísticas descriptivas
- Visualizaciones interactivas
- Detección de datos faltantes

## 🤖 Modelo de Machine Learning

### Características de Entrada
- **Valor Tasado**: Valor de tasación oficial
- **Área (m²)**: Superficie de la propiedad
- **Meses en Venta**: Tiempo en el mercado
- **Habitaciones**: Número de habitaciones
- **Pisos**: Número de plantas
- **Tipo de Propiedad**: Categoría (Single Family, Residential, etc.)

### Algoritmo
- **RandomForestRegressor** de scikit-learn
- Hiperparámetros configurables
- Validación train/test 80/20
- Métrica principal: RMSE

## 📈 MLflow Tracking

El sistema registra automáticamente:
- **Parámetros**: n_estimators, max_depth, etc.
- **Métricas**: RMSE, accuracy
- **Modelos**: Versioning automático
- **Artifacts**: Modelos entrenados

Accede a MLflow UI en: http://localhost:5000

## 📁 Estructura de Datos

### Formato CSV Esperado
```csv
Serial Number,List Year,Date Recorded,Town,Address,Assessed Value,Property Type,Residential Type,area_m2,meses_en_venta,nro_habitaciones,nro_pisos,Sale Amount
21,2021,10/05/2021,Portland,323 JONES HOLLOW RD,279000,Residential,Single Family,155m2,9,5,2,279000
```

### Campos Requeridos
- `Assessed Value`: Valor tasado (numérico)
- `area_m2`: Área en metros cuadrados (formato: "XXXm2")
- `meses_en_venta`: Meses en venta (entero)
- `nro_habitaciones`: Número de habitaciones (entero)
- `nro_pisos`: Número de pisos (entero)
- `Property Type`: Tipo de propiedad (texto)
- `Sale Amount`: Precio de venta (numérico) - Target

## 🔧 Comandos Útiles

### Desarrollo
```bash
# Ejecutar tests (si implementas)
pytest tests/

# Verificar calidad de código
pylint src/

# Formatear código
black src/
```

### MLflow
```bash
# Iniciar servidor MLflow
mlflow server --host 127.0.0.1 --port 5000

# Ver experimentos
mlflow experiments list

# Servir modelo (ejemplo)
mlflow models serve -m "models:/Proyec_Inmobiliario_Model/1" -p 8080
```

### Git
```bash
# Ver estado
git status

# Crear nueva rama para feature
git checkout -b feature/nueva-funcionalidad

# Commit cambios
git add .
git commit -m "Descripción del cambio"

# Push a repositorio
git push origin main
```

## 🐛 Solución de Problemas

### Error: "ModuleNotFoundError"
```bash
# Verificar que estés en el directorio correcto
cd Py_ML_G2

# Activar entorno virtual
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Reinstalar dependencias
pip install -r requirements.txt
```

### Error: "MLflow tracking server not available"
```bash
# Iniciar servidor MLflow
mlflow server --host 127.0.0.1 --port 5000

# Verificar que esté corriendo
curl http://localhost:5000
```

### Error al cargar datos
- Verificar formato del CSV (separador ";")
- Asegurar que todas las columnas requeridas estén presentes
- Revisar encoding del archivo (UTF-8 recomendado)

### 🚀 Docker
docker rmi py_ml_g2_app
docker build -t py_ml_g2_app .
docker run --name proy_ml_gp2 -p 8501:8501 py_ml_g2_app

## 🤝 Contribución

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👥 Autores

- **Grupo 2** - Proyecto de Machine Learning
- **Alex Cruz Salirrosas** - Científico de Datos 
- **Tirso Villanueva Ortiz** - Arquitecto de Software
- **Zaida Doria Delgado** - Especialista en Producto / UX
- **Wilson Guevara Garay** - Desarrollador Python frontend
- **Ronald Canchanya Valenzuela** - Ingeniero/a MLOps / Backend

## 🙏 Agradecimientos

- Docente curso de Machine Learning
- Comunidad de MLflow
- Documentación de Streamlit
- Principios de Arquitectura Hexagonal
