import pandas as pd
import numpy as np
from typing import Tuple
from ...domain.repositories.model_repository import DataRepository

class CSVDataLoader(DataRepository):
    """Implementación del repositorio de datos para archivos CSV."""
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga datos desde un archivo CSV.
        
        Args:
            file_path: Ruta al archivo CSV
            
        Returns:
            DataFrame con los datos cargados
        """
        try:
            df = pd.read_csv(file_path, sep=';')
            print(f"Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
            return df
        except Exception as e:
            raise Exception(f"Error al cargar datos desde {file_path}: {str(e)}")
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocesa los datos siguiendo la lógica del notebook original.
        
        Args:
            df: DataFrame con datos raw
            
        Returns:
            Tuple con (X, y) - features y target
        """
        
        # Eliminar filas con NaNs
        df_clean = df.dropna().copy()
        print(f"Después de eliminar NaNs: {df_clean.shape[0]} filas")
        
        # Limpiar columna area_m2 (remover 'm2' y convertir a float)
        df_clean['area_m2'] = df_clean['area_m2'].str.replace('m2', '').astype(float)
        
        # Crear variables dummy para Property Type
        df_encoded = pd.get_dummies(df_clean, columns=['Property Type'], dtype='float')
        
        # Crear variables dummy para Residential Type
        df_encoded = pd.get_dummies(df_encoded, columns=['Residential Type'], dtype='float')
        
        # Seleccionar features específicas (basado en el notebook)
        feature_columns = [
            'Assessed Value', 'area_m2', 'meses_en_venta', 
            'nro_habitaciones', 'nro_pisos', 
            'Property Type_Residential', 'Property Type_Single Family'
        ]
        
        # Verificar que las columnas existan
        missing_cols = [col for col in feature_columns if col not in df_encoded.columns]
        if missing_cols:
            # Si faltan columnas dummy, las creamos con 0
            for col in missing_cols:
                df_encoded[col] = 0.0
        
        # Extraer features y target
        X = df_encoded[feature_columns].copy()
        y = df_encoded['Sale Amount'].copy()
        
        # Convertir a float32 para compatibilidad con MLflow
        X = X.astype(np.float32)
        
        print(f"Features procesadas: {X.shape}")
        print(f"Columnas de features: {list(X.columns)}")
        print(f"Target shape: {y.shape}")
        
        return X, y