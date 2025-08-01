from dataclasses import dataclass
from typing import Optional

@dataclass
class Property:
    """Entidad que representa una propiedad inmobiliaria."""
    
    assessed_value: float
    area_m2: float
    meses_en_venta: int
    nro_habitaciones: int
    nro_pisos: int
    property_type_residential: float
    property_type_single_family: float
    sale_amount: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convierte la propiedad a diccionario para predicciones."""
        return {
            'Assessed Value': self.assessed_value,
            'area_m2': self.area_m2,
            'meses_en_venta': self.meses_en_venta,
            'nro_habitaciones': self.nro_habitaciones,
            'nro_pisos': self.nro_pisos,
            'Property Type_Residential': self.property_type_residential,
            'Property Type_Single Family': self.property_type_single_family
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Property':
        """Crea una propiedad desde un diccionario."""
        return cls(
            assessed_value=data.get('Assessed Value', 0.0),
            area_m2=data.get('area_m2', 0.0),
            meses_en_venta=data.get('meses_en_venta', 0),
            nro_habitaciones=data.get('nro_habitaciones', 0),
            nro_pisos=data.get('nro_pisos', 0),
            property_type_residential=data.get('Property Type_Residential', 0.0),
            property_type_single_family=data.get('Property Type_Single Family', 0.0),
            sale_amount=data.get('Sale Amount')
        )