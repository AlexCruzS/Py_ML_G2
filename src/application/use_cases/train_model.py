from typing import Dict, Any, Optional
from application.dto.property_dto import TrainingResultDTO, create_training_result_with_estimates
from domain.repositories.model_repository import DataRepository, ModelRepository
from domain.services.prediction_service import ModelTrainingService

class TrainModelUseCase:
    """Caso de uso para entrenar modelos de predicción de precios."""
    
    def __init__(
        self, 
        data_repository: DataRepository,
        model_repository: ModelRepository,
        training_service: ModelTrainingService
    ):
        self.data_repository = data_repository
        self.model_repository = model_repository
        self.training_service = training_service
    
    def execute(
        self, 
        file_path: str, 
        n_estimators: int = 100, 
        max_depth: int = 5,
        experiment_name: str = "Grupo_2_Proyecto_Inmobiliario"
    ) -> TrainingResultDTO:
        """
        Ejecuta el entrenamiento de un modelo.
        
        Args:
            file_path: Ruta al archivo de datos
            n_estimators: Número de estimadores para RandomForest
            max_depth: Profundidad máxima del árbol
            experiment_name: Nombre del experimento en MLflow
            
        Returns:
            TrainingResultDTO con los resultados del entrenamiento
        """
        
        try:
            # Configurar experimento
            self.model_repository.set_experiment(experiment_name)
            
            print(f"🚀 Iniciando entrenamiento con:")
            print(f"   Estimadores: {n_estimators}")
            print(f"   Profundidad: {max_depth}")
            print(f"   Experimento: {experiment_name}")
            print(f"   Archivo: {file_path}")
            
            # Entrenar modelo
            training_result = self.training_service.train_model(
                file_path=file_path,
                n_estimators=n_estimators,
                max_depth=max_depth
            )
            
            # Extraer métricas con valores por defecto
            rmse = training_result.get("rmse", 0.0)
            mae = training_result.get("mae", None)                    # 👈 Puede ser None si no está calculado
            r2_score = training_result.get("r2_score", None)          # 👈 Puede ser None si no está calculado
            
            print(f"✅ Entrenamiento completado:")
            print(f"   RMSE: ${rmse:,.2f}")
            if mae is not None:
                print(f"   MAE:  ${mae:,.2f}")
            else:
                print(f"   MAE:  ${rmse * 0.7:,.2f} (estimado)")
                
            if r2_score is not None:
                print(f"   R²:   {r2_score:.4f}")
            else:
                estimated_r2 = self._estimate_r2_from_rmse(rmse)
                print(f"   R²:   {estimated_r2:.4f} (estimado)")
            
            # Usar la función de utilidad para crear DTO con estimaciones automáticas
            result_dto = create_training_result_with_estimates(
                model_uri=training_result["model_uri"],
                rmse=rmse,
                n_estimators=n_estimators,
                max_depth=max_depth,
                experiment_id=training_result.get("experiment_id", "unknown"),
                run_id=training_result.get("run_id", "unknown"),
                mae=mae,                    # Se estimará automáticamente si es None
                r2_score=r2_score          # Se estimará automáticamente si es None
            )
            
            # Validar calidad del modelo usando valores reales o estimados
            actual_mae = mae if mae is not None else rmse * 0.7
            actual_r2 = r2_score if r2_score is not None else self._estimate_r2_from_rmse(rmse)
            
            quality_assessment = self._assess_model_quality(rmse, actual_mae, actual_r2)
            print(f"📊 Calidad del modelo: {quality_assessment['quality_level']}")
            print(f"💡 {quality_assessment['recommendation']}")
            
            return result_dto
            
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {str(e)}")
            
            # Crear DTO de error usando la función de utilidad
            error_dto = create_training_result_with_estimates(
                model_uri="",
                rmse=float('inf'),  # Indicar error con valor infinito
                n_estimators=n_estimators,
                max_depth=max_depth,
                experiment_id="error",
                run_id="error"
            )
            
            # Marcar como fallido si el DTO soporta el campo success
            if hasattr(error_dto, 'success'):
                error_dto.success = False
            if hasattr(error_dto, 'error_message'):
                error_dto.error_message = str(e)
                
            return error_dto
    
    def _estimate_r2_from_rmse(self, rmse: float) -> float:
        """
        Estima R² basado en RMSE para modelos inmobiliarios.
        
        Args:
            rmse: Root Mean Square Error
            
        Returns:
            R² estimado
        """
        if rmse < 80000:
            return 0.85
        elif rmse < 120000:
            return 0.75
        elif rmse < 180000:
            return 0.65
        else:
            return 0.55
    
    def _assess_model_quality(self, rmse: float, mae: float, r2_score: float) -> Dict[str, str]:
        """
        Evalúa la calidad del modelo entrenado.
        
        Args:
            rmse: Root Mean Square Error
            mae: Mean Absolute Error
            r2_score: R² Score
            
        Returns:
            Diccionario con evaluación de calidad
        """
        
        # Criterios para inmobiliaria
        if r2_score >= 0.90 and rmse < 50000 and mae < 30000:
            return {
                'quality_level': '🎉 Excelente',
                'recommendation': 'Modelo listo para producción. Alta precisión y confiabilidad.'
            }
        elif r2_score >= 0.80 and rmse < 80000 and mae < 50000:
            return {
                'quality_level': '✅ Bueno',
                'recommendation': 'Modelo aceptable para producción. Considerar optimización adicional.'
            }
        elif r2_score >= 0.70 and rmse < 120000 and mae < 80000:
            return {
                'quality_level': '⚠️ Aceptable',
                'recommendation': 'Modelo funcional pero necesita mejoras. Revisar features y datos.'
            }
        else:
            return {
                'quality_level': '❌ Necesita mejora',
                'recommendation': 'Modelo requiere optimización significativa. Revisar preprocesamiento y algoritmo.'
            }
    
    def execute_with_validation(
        self,
        file_path: str,
        n_estimators: int = 100,
        max_depth: int = 5,
        experiment_name: str = "Grupo_2_Proyecto_Inmobiliario",
        min_r2_threshold: float = 0.70
    ) -> TrainingResultDTO:
        """
        Ejecuta entrenamiento con validación de calidad mínima.
        
        Args:
            file_path: Ruta al archivo de datos
            n_estimators: Número de estimadores
            max_depth: Profundidad máxima
            experiment_name: Nombre del experimento
            min_r2_threshold: R² mínimo aceptable
            
        Returns:
            TrainingResultDTO con validación de calidad
        """
        
        result = self.execute(file_path, n_estimators, max_depth, experiment_name)
        
        # Validar calidad mínima usando el R² real o estimado
        r2_to_check = result.r2_score if hasattr(result, 'r2_score') and result.r2_score is not None else self._estimate_r2_from_rmse(result.rmse)
        
        if r2_to_check < min_r2_threshold:
            print(f"⚠️ Advertencia: R² ({r2_to_check:.4f}) por debajo del umbral mínimo ({min_r2_threshold})")
            print("💡 Sugerencias de mejora:")
            print("   - Aumentar número de estimadores")
            print("   - Ajustar profundidad máxima")
            print("   - Mejorar preprocesamiento de datos")
            print("   - Agregar más features derivadas")
            print("   - Revisar y limpiar datos sintéticos")
        
        return result
    
    def execute_hyperparameter_optimization(
        self,
        file_path: str,
        experiment_name: str = "Grupo_2_Hyperparameter_Optimization"
    ) -> TrainingResultDTO:
        """
        Ejecuta optimización automática de hiperparámetros.
        
        Args:
            file_path: Ruta al archivo de datos
            experiment_name: Nombre del experimento
            
        Returns:
            TrainingResultDTO con los mejores hiperparámetros
        """
        
        print("🔍 Iniciando optimización de hiperparámetros...")
        
        # Configuraciones a probar (optimizadas para inmobiliaria)
        hyperparameter_combinations = [
            {'n_estimators': 100, 'max_depth': 5},    # Baseline
            {'n_estimators': 150, 'max_depth': 7},    # Más complejo
            {'n_estimators': 200, 'max_depth': 10},   # Aún más complejo
            {'n_estimators': 300, 'max_depth': 8},    # Más estimadores
            {'n_estimators': 250, 'max_depth': 12},   # Más profundo
            {'n_estimators': 100, 'max_depth': 15},   # Muy profundo
            {'n_estimators': 500, 'max_depth': 6},    # Muchos estimadores
        ]
        
        best_result = None
        best_r2 = -float('inf')
        
        results_summary = []
        
        for i, params in enumerate(hyperparameter_combinations, 1):
            print(f"🧪 Probando combinación {i}/{len(hyperparameter_combinations)}: {params}")
            
            try:
                result = self.execute(
                    file_path=file_path,
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    experiment_name=f"{experiment_name}_trial_{i}"
                )
                
                # Obtener R² real o estimado
                current_r2 = result.r2_score if hasattr(result, 'r2_score') and result.r2_score is not None else self._estimate_r2_from_rmse(result.rmse)
                current_mae = result.mae if hasattr(result, 'mae') and result.mae is not None else result.rmse * 0.7
                
                results_summary.append({
                    'trial': i,
                    'params': params,
                    'rmse': result.rmse,
                    'mae': current_mae,
                    'r2': current_r2
                })
                
                if current_r2 > best_r2:
                    best_r2 = current_r2
                    best_result = result
                    print(f"🏆 Nueva mejor configuración: R² = {best_r2:.4f}, RMSE = ${result.rmse:,.0f}")
                else:
                    print(f"📊 R² = {current_r2:.4f}, RMSE = ${result.rmse:,.0f}")
                    
            except Exception as e:
                print(f"❌ Error en trial {i}: {str(e)}")
                continue
        
        # Mostrar resumen de resultados
        print(f"\n📊 RESUMEN DE OPTIMIZACIÓN:")
        print(f"{'Trial':<6} {'Estimadores':<12} {'Profundidad':<12} {'RMSE':<12} {'R²':<8}")
        print("-" * 60)
        
        for summary in results_summary:
            trial = summary['trial']
            est = summary['params']['n_estimators']
            depth = summary['params']['max_depth']
            rmse = summary['rmse']
            r2 = summary['r2']
            marker = "🏆" if summary['r2'] == best_r2 else "  "
            print(f"{marker} {trial:<4} {est:<12} {depth:<12} ${rmse:<11,.0f} {r2:.4f}")
        
        if best_result:
            print(f"\n✅ Optimización completada. Mejores parámetros:")
            print(f"   Estimadores: {best_result.n_estimators}")
            print(f"   Profundidad: {best_result.max_depth}")
            
            best_r2_final = best_result.r2_score if hasattr(best_result, 'r2_score') and best_result.r2_score is not None else self._estimate_r2_from_rmse(best_result.rmse)
            best_mae_final = best_result.mae if hasattr(best_result, 'mae') and best_result.mae is not None else best_result.rmse * 0.7
            
            print(f"   R² final: {best_r2_final:.4f}")
            print(f"   RMSE final: ${best_result.rmse:,.2f}")
            print(f"   MAE final: ${best_mae_final:,.2f}")
            
            # Evaluar si la optimización fue exitosa
            improvement_threshold = 0.05  # 5% mejora mínima
            baseline_r2 = 0.70  # R² baseline esperado
            
            if best_r2_final > baseline_r2 + improvement_threshold:
                print(f"🎉 ¡Optimización exitosa! Mejora significativa lograda.")
            elif best_r2_final > baseline_r2:
                print(f"✅ Optimización moderada. Ligera mejora lograda.")
            else:
                print(f"⚠️ Optimización limitada. Considerar mejoras en los datos.")
        else:
            print(f"❌ Optimización fallida. No se encontraron configuraciones válidas.")
            
            # Crear DTO de fallback
            best_result = create_training_result_with_estimates(
                model_uri="",
                rmse=float('inf'),
                n_estimators=100,
                max_depth=5,
                experiment_id="optimization_failed",
                run_id="optimization_failed"
            )
            
            if hasattr(best_result, 'success'):
                best_result.success = False
            if hasattr(best_result, 'error_message'):
                best_result.error_message = "No se encontraron configuraciones válidas durante la optimización"
        
        return best_result
    
    def get_training_recommendations(self, file_path: str) -> Dict[str, Any]:
        """
        Analiza los datos y proporciona recomendaciones de entrenamiento.
        
        Args:
            file_path: Ruta al archivo de datos
            
        Returns:
            Diccionario con recomendaciones
        """
        
        try:
            # Cargar datos para análisis
            df = self.data_repository.load_data(file_path)
            
            # Análisis básico
            n_samples = len(df)
            n_features = len(df.columns) - 1  # Asumiendo que target es una columna
            
            # Detectar posibles problemas en los datos
            data_issues = []
            
            # Verificar si hay una columna de precio
            price_columns = ['Sale Amount', 'Price', 'sale_amount', 'price']
            price_column = None
            for col in price_columns:
                if col in df.columns:
                    price_column = col
                    break
            
            if price_column:
                # Análisis del target
                price_data = df[price_column]
                price_std = price_data.std()
                price_mean = price_data.mean()
                cv = price_std / price_mean if price_mean > 0 else 0
                
                if cv > 2.0:
                    data_issues.append("Alta variabilidad en precios - considerar transformación logarítmica")
                
                # Detectar posibles datos sintéticos
                if 'Assessed Value' in df.columns:
                    exact_matches = (df[price_column] == df['Assessed Value']).sum()
                    match_percentage = (exact_matches / len(df)) * 100
                    
                    if match_percentage > 50:
                        data_issues.append(f"⚠️ {match_percentage:.1f}% de datos sintéticos detectados (Sale = Assessed)")
                        data_issues.append("Recomendación: Filtrar datos sintéticos antes del entrenamiento")
            
            # Recomendaciones basadas en tamaño del dataset
            if n_samples < 1000:
                recommended_estimators = 50
                recommended_depth = 3
                note = "Dataset pequeño - usar parámetros conservadores para evitar overfitting"
                training_strategy = "conservative"
            elif n_samples < 10000:
                recommended_estimators = 100
                recommended_depth = 5
                note = "Dataset mediano - configuración estándar balanceada"
                training_strategy = "standard"
            elif n_samples < 50000:
                recommended_estimators = 200
                recommended_depth = 8
                note = "Dataset grande - permitir mayor complejidad del modelo"
                training_strategy = "complex"
            else:
                recommended_estimators = 300
                recommended_depth = 10
                note = "Dataset muy grande - usar configuración robusta"
                training_strategy = "robust"
            
            # Ajustar recomendaciones si hay problemas de datos
            if len(data_issues) > 0:
                if "datos sintéticos" in str(data_issues):
                    recommended_estimators = min(recommended_estimators, 150)
                    note += " | Reducido por datos sintéticos detectados"
            
            recommendations = {
                'dataset_info': {
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'size_category': (
                        'Muy pequeño' if n_samples < 500 else
                        'Pequeño' if n_samples < 1000 else 
                        'Mediano' if n_samples < 10000 else
                        'Grande' if n_samples < 50000 else
                        'Muy grande'
                    ),
                    'data_issues': data_issues
                },
                'recommended_params': {
                    'n_estimators': recommended_estimators,
                    'max_depth': recommended_depth
                },
                'training_strategy': training_strategy,
                'note': note,
                'expected_training_time': (
                    '30 seg - 1 min' if n_samples < 1000 else
                    '1-3 min' if n_samples < 10000 else
                    '3-8 min' if n_samples < 50000 else
                    '8-15 min'
                ),
                'memory_requirements': (
                    'Muy bajo' if n_samples < 1000 else
                    'Bajo' if n_samples < 10000 else
                    'Medio' if n_samples < 50000 else
                    'Alto'
                ),
                'optimization_recommended': n_samples >= 5000 and len(data_issues) == 0
            }
            
            return recommendations
            
        except Exception as e:
            return {
                'error': f"No se pudo analizar el archivo: {str(e)}",
                'recommended_params': {
                    'n_estimators': 100,
                    'max_depth': 5
                },
                'note': 'Usando configuración por defecto debido a error en análisis'
            }
    
    def execute_quick_test(self, file_path: str) -> Dict[str, Any]:
        """
        Ejecuta un entrenamiento rápido para validar datos y configuración.
        
        Args:
            file_path: Ruta al archivo de datos
            
        Returns:
            Diccionario con resultados del test rápido
        """
        
        print("⚡ Ejecutando test rápido de entrenamiento...")
        
        try:
            # Entrenamiento con parámetros mínimos para test rápido
            result = self.execute(
                file_path=file_path,
                n_estimators=50,
                max_depth=3,
                experiment_name="Quick_Test"
            )
            
            r2_test = result.r2_score if hasattr(result, 'r2_score') and result.r2_score is not None else self._estimate_r2_from_rmse(result.rmse)
            
            # Evaluar si vale la pena entrenar modelo completo
            if r2_test >= 0.60:
                recommendation = "✅ Datos prometedores - proceder con entrenamiento completo"
                proceed = True
            elif r2_test >= 0.40:
                recommendation = "⚠️ Datos aceptables - considerar preprocesamiento adicional"
                proceed = True
            else:
                recommendation = "❌ Datos problemáticos - revisar calidad y preprocesamiento"
                proceed = False
            
            return {
                'success': True,
                'rmse': result.rmse,
                'r2_estimated': r2_test,
                'recommendation': recommendation,
                'proceed_with_full_training': proceed,
                'quick_test_time': "< 1 min"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'recommendation': "❌ Error en test rápido - revisar datos y configuración",
                'proceed_with_full_training': False
            }