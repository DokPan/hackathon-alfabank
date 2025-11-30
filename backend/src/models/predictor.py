import pickle
import pandas as pd
from typing import Dict, Any
import logging

import sys
import os

try:
    from schemas.prediction import PredictionInput, PredictionOutput
except ImportError:
    pass

class IncomePredictor:
    def __init__(self, model_path: str):
        self.model = None
        self.preprocessor = None
        self.load_model(model_path)

    def load_model(self, model_path: str):
        """Загрузка модели и препроцессора"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.preprocessor = model_data['preprocessor']
                self.feature_names = model_data['feature_names']
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Предсказание дохода с объяснением"""
        try:
            # Преобразование в DataFrame
            input_df = pd.DataFrame([input_data])

            # Препроцессинг
            processed_data = self.preprocessor.transform(input_df)

            # Предсказание
            prediction = self.model.predict(processed_data)[0]
            probability = self.model.predict_proba(processed_data)[0]

            # Объяснение (SHAP/LIME)
            explanation = self._explain_prediction(processed_data)

            # Рекомендации
            recommendations = self._generate_recommendations(
                input_data, explanation
            )

            return {
                "prediction": float(prediction),
                "probability": float(probability[1]),  # вероятность класса 1
                "explanation": explanation,
                "recommendations": recommendations
            }

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            raise

    def _explain_prediction(self, processed_data) -> Dict[str, Any]:
        """Генерация объяснений предсказания"""
        # Используем SHAP или feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            return {
                "feature_importance": feature_importance,
                "top_factors": sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
        return {}

    def _generate_recommendations(self, input_data: Dict[str, Any],
                                  explanation: Dict[str, Any]) -> list:
        """Генерация рекомендаций на основе предсказания"""
        recommendations = []

        # Пример логики рекомендаций
        prediction = explanation.get("prediction", 0)
        if prediction < 50000:
            recommendations.append("Рассмотрите возможность повышения квалификации")
            recommendations.append("Исследуйте возможности карьерного роста")

        return recommendations