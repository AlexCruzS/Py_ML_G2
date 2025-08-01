from application.dto.property_dto import PropertyInputDTO, PropertyPredictionDTO
from domain.entities.property import Property
from domain.services.prediction_service import PredictionService
from domain.repositories.model_repository import ModelRepository

class PredictPriceUseCase:
    """Caso de uso para predecir precios de propiedades."""
    
    def __init__(
        self, 
        prediction_service: PredictionService,
        model_repository: ModelRepository
    ):
        self.prediction_service = prediction_service
        self.model_repository = model_repository
    
    def execute(self, property_input: PropertyInputDTO) -> PropertyPredictionDTO:
        """
        Ejecuta la predicción de precio para una propiedad.
        
        Args:
            property_input: Datos de entrada de la propiedad
            
        Returns:
            PropertyPredictionDTO con la predicción
        """
        
        # Validar entrada
        if not property_input.validate():
            raise ValueError("Datos de propiedad inválidos")
        
        # Convertir DTO a entidad de dominio
        property_entity = self._convert_dto_to_entity(property_input)
        
        # Validar entidad
        if not self.prediction_service.validate_property(property_entity):
            raise ValueError("Propiedad no válida para predicción")
        
        # Realizar predicción
        predicted_price = self.prediction_service.predict_price(property_entity)
        
        return PropertyPredictionDTO(
            predicted_price=float(predicted_price),
            model_version="latest"
        )
    
    def _convert_dto_to_entity(self, dto: PropertyInputDTO) -> Property:
        """Convierte DTO a entidad de dominio."""
        
        # Mapear tipo de propiedad a variables dummy
        property_type_residential = 1.0 if dto.property_type == "Residential" else 0.0
        property_type_single_family = 1.0 if dto.property_type == "Single Family" else 0.0
        
        return Property(
            assessed_value=dto.assessed_value,
            area_m2=dto.area_m2,
            meses_en_venta=dto.meses_en_venta,
            nro_habitaciones=dto.nro_habitaciones,
            nro_pisos=dto.nro_pisos,
            property_type_residential=property_type_residential,
            property_type_single_family=property_type_single_family
        )