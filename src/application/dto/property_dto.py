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
    
    def validate_detailed(self) -> dict:
        """
        Validación detallada con errores y advertencias específicas.
        
        Returns:
            Diccionario con información detallada de validación
        """
        errors = []
        warnings = []
        
        # Validaciones de error (críticas)
        if self.assessed_value <= 0:
            errors.append("El valor tasado debe ser mayor a 0")
        if self.area_m2 <= 0:
            errors.append("El área debe ser mayor a 0 m²")
        if self.meses_en_venta < 0:
            errors.append("Los meses en venta no pueden ser negativos")
        if self.nro_habitaciones < 1:
            errors.append("Debe tener al menos 1 habitación")
        if self.nro_pisos < 1:
            errors.append("Debe tener al menos 1 piso")
        if self.property_type not in ["Residential", "Single Family", "Condo", "Two Family", "Three Family", "Four Family"]:
            errors.append(f"Tipo de propiedad '{self.property_type}' no válido")
        
        # Validaciones de advertencia (valores inusuales pero posibles)
        if self.assessed_value > 5000000:  # $5M
            warnings.append("Valor tasado muy alto - verificar precisión")
        if self.assessed_value < 50000:  # $50K
            warnings.append("Valor tasado muy bajo - verificar precisión")
        if self.area_m2 > 500:
            warnings.append("Área muy grande - verificar unidades (m²)")
        if self.area_m2 < 30:
            warnings.append("Área muy pequeña para una vivienda")
        if self.meses_en_venta > 24:
            warnings.append("Mucho tiempo en el mercado (>24 meses)")
        if self.nro_habitaciones > 8:
            warnings.append("Número inusual de habitaciones")
        if self.nro_pisos > 4:
            warnings.append("Número inusual de pisos para vivienda")
        
        is_valid = len(errors) == 0
        
        return {
            'is_valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'error_count': len(errors),
            'warning_count': len(warnings),
            'validation_summary': f"{'✅ Válido' if is_valid else '❌ Inválido'} - {len(errors)} errores, {len(warnings)} advertencias"
        }
    
    def get_summary(self) -> str:
        """Retorna un resumen legible de la propiedad."""
        return f"{self.property_type} de {self.area_m2}m², {self.nro_habitaciones} hab., {self.nro_pisos} pisos, valor tasado ${self.assessed_value:,.0f}"


@dataclass
class PropertyPredictionDTO:
    """DTO para resultado de predicción."""
    
    predicted_price: float
    confidence_interval: Optional[tuple] = None
    model_version: Optional[str] = None
    
    def get_formatted_price(self) -> str:
        """Retorna el precio formateado."""
        return f"${self.predicted_price:,.0f}"
    
    def get_confidence_range(self) -> Optional[str]:
        """Retorna el rango de confianza formateado."""
        if self.confidence_interval:
            lower, upper = self.confidence_interval
            return f"${lower:,.0f} - ${upper:,.0f}"
        return None


@dataclass
class TrainingResultDTO:
    """DTO para resultado de entrenamiento."""
    
    # Campos requeridos (sin valores por defecto) - DEBEN IR PRIMERO
    model_uri: str
    rmse: float
    n_estimators: int
    max_depth: int
    experiment_id: str
    run_id: str
    
    # Campos opcionales (con valores por defecto) - DEBEN IR AL FINAL
    mae: Optional[float] = None                    # 👈 Agregado MAE
    r2_score: Optional[float] = None               # 👈 Agregado R²
    success: bool = True                           # Estado del entrenamiento
    error_message: Optional[str] = None            # Mensaje de error si falla
    mse: Optional[float] = None                    # Mean Squared Error
    mape: Optional[float] = None                   # Mean Absolute Percentage Error
    training_samples: Optional[int] = None         # Información de validación
    test_samples: Optional[int] = None
    features_count: Optional[int] = None
    training_time_seconds: Optional[float] = None  # Timing information
    
    def get_quality_assessment(self) -> str:
        """
        Retorna una evaluación textual de la calidad del modelo.
        
        Returns:
            Evaluación de calidad como string
        """
        if not self.success:
            return "❌ Entrenamiento fallido"
        
        # Usar R² si está disponible, sino estimar basado en RMSE
        r2_to_use = self.r2_score if self.r2_score is not None else self._estimate_r2_from_rmse()
        mae_to_use = self.mae if self.mae is not None else self.rmse * 0.7
        
        if r2_to_use >= 0.90 and self.rmse < 50000 and mae_to_use < 30000:
            return "🎉 Excelente - Modelo de alta calidad"
        elif r2_to_use >= 0.80 and self.rmse < 80000 and mae_to_use < 50000:
            return "✅ Bueno - Modelo aceptable para producción"
        elif r2_to_use >= 0.70 and self.rmse < 120000 and mae_to_use < 80000:
            return "⚠️ Aceptable - Modelo funcional, considerar mejoras"
        else:
            return "❌ Necesita mejora - Requiere optimización"
    
    def _estimate_r2_from_rmse(self) -> float:
        """Estima R² basado en RMSE para inmobiliaria."""
        if self.rmse < 80000:
            return 0.85
        elif self.rmse < 120000:
            return 0.75
        elif self.rmse < 180000:
            return 0.65
        else:
            return 0.55
    
    def get_formatted_metrics(self) -> dict:
        """
        Retorna métricas formateadas para mostrar en UI.
        
        Returns:
            Diccionario con métricas formateadas
        """
        # Usar valores reales o estimados
        r2_display = self.r2_score if self.r2_score is not None else self._estimate_r2_from_rmse()
        mae_display = self.mae if self.mae is not None else self.rmse * 0.7
        
        return {
            'rmse_formatted': f"${self.rmse:,.0f}",
            'mae_formatted': f"${mae_display:,.0f}{'*' if self.mae is None else ''}",
            'r2_formatted': f"{r2_display:.4f}{'*' if self.r2_score is None else ''}",
            'r2_percentage': f"{r2_display * 100:.1f}%{'*' if self.r2_score is None else ''}",
            'quality_assessment': self.get_quality_assessment(),
            'has_estimated_values': self.mae is None or self.r2_score is None
        }
    
    def is_production_ready(self, min_r2: float = 0.75, max_rmse: float = 100000) -> bool:
        """
        Determina si el modelo está listo para producción.
        
        Args:
            min_r2: R² mínimo requerido
            max_rmse: RMSE máximo permitido
            
        Returns:
            True si está listo para producción
        """
        if not self.success:
            return False
        
        r2_to_check = self.r2_score if self.r2_score is not None else self._estimate_r2_from_rmse()
        
        return r2_to_check >= min_r2 and self.rmse <= max_rmse
    
    def get_improvement_suggestions(self) -> list[str]:
        """
        Genera sugerencias de mejora basadas en las métricas.
        
        Returns:
            Lista de sugerencias
        """
        suggestions = []
        
        if not self.success:
            suggestions.append("Revisar datos de entrada y configuración del modelo")
            return suggestions
        
        r2_to_use = self.r2_score if self.r2_score is not None else self._estimate_r2_from_rmse()
        mae_to_use = self.mae if self.mae is not None else self.rmse * 0.7
        
        # Sugerencias basadas en R²
        if r2_to_use < 0.70:
            suggestions.append("R² bajo - Considerar feature engineering adicional")
            suggestions.append("Probar algoritmos más complejos (GradientBoosting, XGBoost)")
        
        # Sugerencias basadas en RMSE
        if self.rmse > 150000:
            suggestions.append("RMSE alto - Revisar outliers en los datos")
            suggestions.append("Considerar transformaciones logarítmicas del target")
        
        # Sugerencias basadas en MAE
        if mae_to_use > 100000:
            suggestions.append("MAE alto - Verificar calidad de los datos")
            suggestions.append("Implementar preprocesamiento más robusto")
        
        # Sugerencias de hiperparámetros
        if self.n_estimators < 200 and r2_to_use < 0.85:
            suggestions.append("Aumentar número de estimadores para mejor performance")
        
        if self.max_depth < 10 and r2_to_use < 0.80:
            suggestions.append("Considerar aumentar profundidad máxima")
        
        # Sugerencias de datos
        if r2_to_use < 0.75:
            suggestions.append("Revisar preprocesamiento de datos (eliminar datos sintéticos)")
            suggestions.append("Agregar más features derivadas (precio por m², ratios, etc.)")
        
        if not suggestions:
            suggestions.append("✅ Modelo en buen estado - considerar fine-tuning para optimización")
        
        return suggestions
    
    def get_metrics_summary(self) -> str:
        """Retorna resumen de métricas en formato texto."""
        formatted = self.get_formatted_metrics()
        
        summary = f"""
📊 MÉTRICAS DEL MODELO:
   RMSE: {formatted['rmse_formatted']}
   MAE:  {formatted['mae_formatted']}
   R²:   {formatted['r2_formatted']} ({formatted['r2_percentage']})
   
⚙️ CONFIGURACIÓN:
   Estimadores: {self.n_estimators}
   Profundidad: {self.max_depth}
   
📈 EVALUACIÓN: {formatted['quality_assessment']}
        """
        
        if formatted['has_estimated_values']:
            summary += "\n💡 Nota: Valores marcados con * son estimados"
        
        return summary.strip()


@dataclass
class PredictionResultDTO:
    """DTO mejorado para resultado de predicciones."""
    
    predicted_price: float
    model_version: str
    confidence_score: Optional[float] = None
    prediction_interval: Optional[dict] = None
    model_metrics: Optional[dict] = None
    input_validation: Optional[dict] = None
    
    def get_formatted_price(self) -> str:
        """Retorna precio formateado."""
        return f"${self.predicted_price:,.0f}"
    
    def get_confidence_level(self) -> str:
        """Retorna nivel de confianza textual."""
        if self.confidence_score is None:
            return "No disponible"
        
        if self.confidence_score >= 0.90:
            return "🎯 Muy alta"
        elif self.confidence_score >= 0.75:
            return "✅ Alta"
        elif self.confidence_score >= 0.60:
            return "⚠️ Media"
        else:
            return "❌ Baja"
    
    def get_prediction_range(self) -> Optional[str]:
        """Retorna rango de predicción formateado."""
        if self.prediction_interval:
            lower = self.prediction_interval.get('lower', 0)
            upper = self.prediction_interval.get('upper', 0)
            return f"${lower:,.0f} - ${upper:,.0f}"
        return None


# Funciones de utilidad para DTOs
def create_training_result_with_estimates(
    model_uri: str,
    rmse: float,
    n_estimators: int,
    max_depth: int,
    experiment_id: str,
    run_id: str,
    mae: Optional[float] = None,
    r2_score: Optional[float] = None
) -> TrainingResultDTO:
    """
    Crea un TrainingResultDTO con estimaciones automáticas si faltan métricas.
    
    Args:
        model_uri: URI del modelo
        rmse: Root Mean Square Error
        n_estimators: Número de estimadores
        max_depth: Profundidad máxima
        experiment_id: ID del experimento
        run_id: ID del run
        mae: Mean Absolute Error (opcional, se estima si no se proporciona)
        r2_score: R² Score (opcional, se estima si no se proporciona)
    
    Returns:
        TrainingResultDTO con todas las métricas
    """
    
    # Estimar MAE si no se proporciona (típicamente ~70% del RMSE)
    estimated_mae = mae if mae is not None else rmse * 0.7
    
    # Estimar R² basado en RMSE si no se proporciona
    if r2_score is None:
        if rmse < 80000:
            estimated_r2 = 0.85
        elif rmse < 120000:
            estimated_r2 = 0.75
        elif rmse < 180000:
            estimated_r2 = 0.65
        else:
            estimated_r2 = 0.55
    else:
        estimated_r2 = r2_score
    
    return TrainingResultDTO(
        model_uri=model_uri,
        rmse=rmse,
        n_estimators=n_estimators,
        max_depth=max_depth,
        experiment_id=experiment_id,
        run_id=run_id,
        mae=estimated_mae,
        r2_score=estimated_r2
    )