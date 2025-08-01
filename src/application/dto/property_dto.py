from dataclasses import dataclass
from typing import Optional

@dataclass
class PropertyInputDTO:
    """DTO para entrada de datos de una propiedad."""
    
    assessed_value: float
    area_m2: float
    meses_en_venta: int
    nro_habitaciones: int
    nro_pisos: int
    property_type: str  # "Residential", "Single Family", etc.
    
    def validate(self) -> bool:
        """Valida que los datos sean correctos."""
        if self.assessed_value <= 0:
            return False
        if self.area_m2 <= 0:
            return False
        if self.meses_en_venta < 0:
            return False
        if self.nro_habitaciones < 1:
            return False
        if self.nro_pisos < 1:
            return False
        if self.property_type not in ["Residential", "Single Family", "Condo", "Two Family", "Three Family", "Four Family"]:
            return False
        return True

@dataclass
class PropertyPredictionDTO:
    """DTO para resultado de predicción."""
    
    predicted_price: float
    confidence_interval: Optional[tuple] = None
    model_version: Optional[str] = None
    
@dataclass
class TrainingResultDTO:
    """DTO para resultado de entrenamiento."""
    
    model_uri: str
    rmse: float
    n_estimators: int
    max_depth: int
    experiment_id: str
    run_id: str