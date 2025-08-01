"""
Script para corregir automáticamente las importaciones relativas a absolutas.
"""

import os
import re

def fix_imports_in_file(file_path):
    """Corrige las importaciones en un archivo específico."""
    
    if not os.path.exists(file_path):
        print(f"❌ Archivo no encontrado: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Patrones de reemplazo para importaciones relativas
        patterns = [
            # Dos puntos hacia arriba (..module)
            (r'from \.\.dto\.', 'from application.dto.'),
            (r'from \.\.use_cases\.', 'from application.use_cases.'),
            
            # Tres puntos hacia arriba (...module)
            (r'from \.\.\.domain\.entities\.', 'from domain.entities.'),
            (r'from \.\.\.domain\.repositories\.', 'from domain.repositories.'),
            (r'from \.\.\.domain\.services\.', 'from domain.services.'),
            (r'from \.\.\.infrastructure\.', 'from infrastructure.'),
            (r'from \.\.\.application\.', 'from application.'),
        ]
        
        # Aplicar reemplazos
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # Verificar si hubo cambios
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Corregido: {file_path}")
            return True
        else:
            print(f"ℹ️  Sin cambios: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error procesando {file_path}: {e}")
        return False

def main():
    """Función principal para corregir todos los archivos."""
    
    print("🔧 Corrigiendo importaciones relativas a absolutas...")
    print("=" * 60)
    
    # Lista de archivos a corregir
    files_to_fix = [
        "src/application/use_cases/train_model.py",
        "src/application/use_cases/predict_price.py",
        "src/infrastructure/data/data_loader.py",
        "src/infrastructure/ml/mlflow_repository.py",
        "src/infrastructure/ml/model_trainer.py",
    ]
    
    fixed_count = 0
    
    for file_path in files_to_fix:
        if fix_imports_in_file(file_path):
            fixed_count += 1
    
    print("=" * 60)
    print(f"📊 Resumen: {fixed_count} archivos corregidos de {len(files_to_fix)}")
    
    if fixed_count > 0:
        print("✅ Importaciones corregidas exitosamente!")
        print("🚀 Ahora puedes ejecutar: python launch_app.py")
    else:
        print("ℹ️  No se encontraron importaciones relativas para corregir")

if __name__ == "__main__":
    main()