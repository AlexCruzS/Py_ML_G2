"""
Script de configuración para el proyecto Py_ML_G2.
Automatiza la configuración inicial del entorno.
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Ejecuta un comando y maneja errores."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e.stderr}")
        return False

def check_python_version():
    """Verifica la versión de Python."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ es requerido")
        return False
    print(f"✅ Python {version.major}.{version.minor} detectado")
    return True

def create_directories():
    """Crea directorios necesarios."""
    directories = [
        "data",
        "models", 
        "mlruns",
        "logs"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Directorio creado: {directory}")
        else:
            print(f"ℹ️  Directorio ya existe: {directory}")

def setup_git():
    """Configura Git si no está inicializado."""
    if not os.path.exists(".git"):
        run_command("git init", "Inicializando Git")
        run_command("git add .", "Agregando archivos a Git")
        run_command('git commit -m "Initial commit: Proyecto ML Inmobiliario con arquitectura hexagonal"', "Primer commit")
    else:
        print("ℹ️  Git ya está inicializado")

def create_sample_data():
    """Crea un archivo de datos de ejemplo."""
    sample_data = """Serial Number;List Year;Date Recorded;Town;Address;Assessed Value;Property Type;Residential Type;area_m2;meses_en_venta;nro_habitaciones;nro_pisos;Sale Amount
21;2021;10/05/2021;Portland;323 JONES HOLLOW RD;279000;Residential;Single Family;155m2;9;5;2;279000
82;2008;10/01/2008;Windham;17 SUNRISE HILL;443200;Single Family;Single Family;277m2;1;7;3;443200
83;2008;10/01/2008;Windham;33 HIDDEN PINES CIRCLE;454400;Single Family;Single Family;284m2;4;7;3;454400
84;2008;10/01/2008;Windham;24 PILGRIM LANE;262800;Single Family;Single Family;146m2;7;4;2;262800
85;2008;10/01/2008;Windham;46 ORCHID RD;341700;Single Family;Single Family;201m2;8;6;3;341700"""
    
    sample_file = "data/sample_dataset.csv"
    if not os.path.exists(sample_file):
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_data)
        print("✅ Archivo de datos de ejemplo creado: data/sample_dataset.csv")
    else:
        print("ℹ️  Archivo de datos de ejemplo ya existe")

def main():
    """Función principal de configuración."""
    print("🏠 Configuración del Proyecto Py_ML_G2")
    print("=" * 50)
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Crear directorios
    create_directories()
    
    # Configurar Git
    setup_git()
    
    # Crear datos de ejemplo
    create_sample_data()
    
    print("\n🎉 Configuración completada!")
    print("\nPróximos pasos:")
    print("1. Crear entorno virtual: python -m venv venv")
    print("2. Activar entorno: venv\\Scripts\\activate (Windows) o source venv/bin/activate (Linux/Mac)")
    print("3. Instalar dependencias: pip install -r requirements.txt")
    print("4. Iniciar MLflow: mlflow server --host 127.0.0.1 --port 5000")
    print("5. Ejecutar aplicación: streamlit run src/infrastructure/web/streamlit_app.py")

if __name__ == "__main__":
    main()