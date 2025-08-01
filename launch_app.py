"""
Script de lanzamiento para la aplicación Streamlit.
Configura correctamente el PYTHONPATH antes de iniciar Streamlit.
"""

import os
import sys
import subprocess

def main():
    """Lanza la aplicación Streamlit con el PYTHONPATH correcto."""
    
    # Obtener directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    
    # Agregar src al PYTHONPATH
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    if pythonpath:
        env['PYTHONPATH'] = f"{src_dir}{os.pathsep}{pythonpath}"
    else:
        env['PYTHONPATH'] = src_dir
    
    # Ruta al archivo streamlit
    streamlit_file = os.path.join(src_dir, 'infrastructure', 'web', 'streamlit_app.py')
    
    print("🚀 Iniciando aplicación Streamlit...")
    print(f"📁 PYTHONPATH: {env['PYTHONPATH']}")
    print(f"📄 Archivo: {streamlit_file}")
    print("🌐 La aplicación se abrirá en: http://localhost:8501")
    print("=" * 60)
    
    # Ejecutar Streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', streamlit_file
        ], env=env, check=True)
    except KeyboardInterrupt:
        print("\n👋 Aplicación cerrada por el usuario")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar Streamlit: {e}")

if __name__ == "__main__":
    main()