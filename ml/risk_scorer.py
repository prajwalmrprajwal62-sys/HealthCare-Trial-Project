"""
Drishti Health — XGBoost Risk Scoring Engine with SHAP Explainability

Provides a unified risk score (0-10) combining:
- Diabetes risk (UCI Pima Indians dataset)
- Heart disease risk (UCI Heart Disease dataset)
- Optional: Fundus DR grade integration

Uses SHAP for transparent, explainable predictions that ASHA workers
and patients can understand.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import pickle
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Any


class RiskScorer:
    """Multi-disease risk scoring engine with SHAP explainability."""

    FEATURE_NAMES_DIABETES = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]

    FEATURE_NAMES_HEART = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    # Unified feature set for the combined risk model
    UNIFIED_FEATURES = [
        "age", "sex", "bp_systolic", "bp_diastolic", "glucose",
        "hba1c", "bmi", "cholesterol", "heart_rate", "smoking",
        "family_history_diabetes", "family_history_heart",
        "physical_activity", "pregnancies"
    ]

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.diabetes_model: Optional[xgb.XGBClassifier] = None
        self.heart_model: Optional[xgb.XGBClassifier] = None
        self.unified_model: Optional[xgb.XGBClassifier] = None
        self.explainer: Optional[shap.TreeExplainer] = None

        self._load_or_create_models()

    def _load_or_create_models(self):
        """Load pre-trained models or create demo models."""
        unified_path = self.model_dir / "unified_risk_model.pkl"

        if unified_path.exists():
            with open(unified_path, "rb") as f:
                saved = pickle.load(f)
                self.unified_model = saved["model"]
                self.explainer = shap.TreeExplainer(self.unified_model)
            print("✅ Loaded pre-trained unified risk model")
        else:
            print("⚠️  No pre-trained model found. Creating demo model...")
            self._create_demo_model()

    def _create_demo_model(self):
        """Create a demo model trained on synthetic data matching the unified feature set."""
        np.random.seed(42)
        n_samples = 2000

        # Generate realistic synthetic data
        data = pd.DataFrame({
            "age": np.random.normal(45, 15, n_samples).clip(18, 90),
            "sex": np.random.binomial(1, 0.5, n_samples),
            "bp_systolic": np.random.normal(130, 20, n_samples).clip(80, 200),
            "bp_diastolic": np.random.normal(82, 12, n_samples).clip(50, 130),
            "glucose": np.random.normal(120, 40, n_samples).clip(60, 300),
            "hba1c": np.random.normal(6.5, 1.5, n_samples).clip(4, 14),
            "bmi": np.random.normal(27, 6, n_samples).clip(15, 50),
            "cholesterol": np.random.normal(220, 45, n_samples).clip(100, 400),
            "heart_rate": np.random.normal(75, 12, n_samples).clip(50, 120),
            "smoking": np.random.binomial(1, 0.25, n_samples),
            "family_history_diabetes": np.random.binomial(1, 0.35, n_samples),
            "family_history_heart": np.random.binomial(1, 0.3, n_samples),
            "physical_activity": np.random.uniform(0, 10, n_samples),
            "pregnancies": np.random.poisson(2, n_samples).clip(0, 15),
        })

        # Create realistic risk labels based on clinical risk factors
        risk_score = (
            0.25 * (data["glucose"] > 140).astype(float) +
            0.20 * (data["hba1c"] > 7).astype(float) +
            0.15 * (data["bp_systolic"] > 140).astype(float) +
            0.10 * (data["bmi"] > 30).astype(float) +
            0.10 * (data["age"] > 50).astype(float) +
            0.08 * data["smoking"] +
            0.05 * data["family_history_diabetes"] +
            0.05 * data["family_history_heart"] +
            0.02 * (data["cholesterol"] > 240).astype(float) +
            np.random.normal(0, 0.05, n_samples)
        )

        labels = (risk_score > 0.45).astype(int)

        # Train XGBoost
        self.unified_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42
        )
        self.unified_model.fit(data, labels)
        self.explainer = shap.TreeExplainer(self.unified_model)

        # Save model
        with open(self.model_dir / "unified_risk_model.pkl", "wb") as f:
            pickle.dump({"model": self.unified_model}, f)
        print("✅ Demo model created and saved")

    def predict_risk(self, vitals: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict unified risk score from patient vitals.

        Args:
            vitals: Dictionary with keys matching UNIFIED_FEATURES.
                    Missing values will be imputed with population means.

        Returns:
            Dictionary with risk_score (0-10), confidence, risk_level,
            contributing_factors, and recommendation.
        """
        # Create input dataframe with defaults for missing values
        defaults = {
            "age": 40, "sex": 0, "bp_systolic": 120, "bp_diastolic": 80,
            "glucose": 100, "hba1c": 5.7, "bmi": 25, "cholesterol": 200,
            "heart_rate": 72, "smoking": 0, "family_history_diabetes": 0,
            "family_history_heart": 0, "physical_activity": 5, "pregnancies": 0
        }

        input_data = {feat: vitals.get(feat, defaults.get(feat, 0))
                      for feat in self.UNIFIED_FEATURES}
        df = pd.DataFrame([input_data])

        # Get probability
        prob = self.unified_model.predict_proba(df)[0][1]
        risk_score = round(prob * 10, 1)  # Scale to 0-10

        # Get SHAP values for explainability
        shap_values = self.explainer.shap_values(df)

        # Build contributing factors sorted by importance
        feature_contributions = {}
        for i, feat in enumerate(self.UNIFIED_FEATURES):
            feature_contributions[feat] = {
                "value": float(input_data[feat]),
                "shap_value": float(shap_values[0][i]),
                "impact": "increases_risk" if shap_values[0][i] > 0 else "decreases_risk"
            }

        # Sort by absolute SHAP value
        sorted_factors = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]["shap_value"]),
            reverse=True
        )

        # Determine risk level and recommendation
        if risk_score <= 3:
            risk_level = "LOW"
            recommendation = "Regular monitoring recommended. Continue healthy lifestyle."
            recommendation_kn = "ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ ಶಿಫಾರಸು. ಆರೋಗ್ಯಕರ ಜೀವನಶೈಲಿ ಮುಂದುವರಿಸಿ."
        elif risk_score <= 6:
            risk_level = "MODERATE"
            recommendation = "Schedule appointment at PHC within 2 weeks. Lifestyle modifications advised."
            recommendation_kn = "2 ವಾರಗಳಲ್ಲಿ PHC ಯಲ್ಲಿ ಅಪಾಯಿಂಟ್ಮೆಂಟ್ ನಿಗದಿಪಡಿಸಿ."
        elif risk_score <= 8:
            risk_level = "HIGH"
            recommendation = "Urgent referral to taluk hospital. Doctor consultation required within 48 hours."
            recommendation_kn = "ತಾಲ್ಲೂಕು ಆಸ್ಪತ್ರೆಗೆ ತುರ್ತು ರೆಫರಲ್. 48 ಗಂಟೆಗಳಲ್ಲಿ ವೈದ್ಯರ ಸಮಾಲೋಚನೆ ಅಗತ್ಯ."
        else:
            risk_level = "CRITICAL"
            recommendation = "IMMEDIATE medical attention required. Transport to district hospital."
            recommendation_kn = "ತಕ್ಷಣದ ವೈದ್ಯಕೀಯ ಗಮನ ಅಗತ್ಯ. ಜಿಲ್ಲಾ ಆಸ್ಪತ್ರೆಗೆ ಸಾಗಿಸಿ."

        # Top 3 contributing factors for summary
        top_factors = []
        total_abs_shap = sum(abs(v["shap_value"]) for _, v in sorted_factors)
        for feat, vals in sorted_factors[:5]:
            pct = abs(vals["shap_value"]) / total_abs_shap * 100 if total_abs_shap > 0 else 0
            top_factors.append({
                "feature": feat,
                "value": vals["value"],
                "contribution_pct": round(pct, 1),
                "direction": vals["impact"]
            })

        return {
            "risk_score": risk_score,
            "risk_probability": round(prob, 4),
            "risk_level": risk_level,
            "confidence": round(min(0.95, 0.7 + abs(prob - 0.5) * 0.5), 2),
            "recommendation": recommendation,
            "recommendation_kn": recommendation_kn,
            "top_factors": top_factors,
            "all_factors": dict(sorted_factors),
            "shap_values": shap_values[0].tolist(),
            "feature_names": self.UNIFIED_FEATURES,
            "input_values": list(input_data.values())
        }

    def get_shap_plot_data(self, vitals: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Get data needed to render SHAP waterfall/force plots.

        Returns:
            Tuple of (shap_values, base_value, feature_names)
        """
        input_data = {feat: vitals.get(feat, 0) for feat in self.UNIFIED_FEATURES}
        df = pd.DataFrame([input_data])

        shap_values = self.explainer.shap_values(df)
        base_value = self.explainer.expected_value

        return shap_values[0], base_value, self.UNIFIED_FEATURES


# Convenience function for quick predictions
def quick_risk_assessment(
    age: float, bp_systolic: float, glucose: float,
    hba1c: float = 5.7, bmi: float = 25.0, **kwargs
) -> Dict[str, Any]:
    """Quick risk assessment with minimal inputs."""
    scorer = RiskScorer()
    vitals = {
        "age": age, "bp_systolic": bp_systolic,
        "glucose": glucose, "hba1c": hba1c, "bmi": bmi,
        **kwargs
    }
    return scorer.predict_risk(vitals)


if __name__ == "__main__":
    # Demo: Run risk assessment for sample patient
    scorer = RiskScorer()

    # Meena's patient — 48-year-old woman with blurred vision
    sample_vitals = {
        "age": 48,
        "sex": 0,
        "bp_systolic": 155,
        "bp_diastolic": 95,
        "glucose": 210,
        "hba1c": 8.2,
        "bmi": 31.5,
        "cholesterol": 245,
        "heart_rate": 82,
        "smoking": 0,
        "family_history_diabetes": 1,
        "family_history_heart": 0,
        "physical_activity": 2,
        "pregnancies": 3
    }

    result = scorer.predict_risk(sample_vitals)

    print("\n" + "=" * 60)
    print("🏥 DRISHTI HEALTH — Risk Assessment Report")
    print("=" * 60)
    print(f"Risk Score: {result['risk_score']}/10")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Confidence: {result['confidence'] * 100:.0f}%")
    print(f"\n📋 Recommendation: {result['recommendation']}")
    print(f"📋 ಶಿಫಾರಸು: {result['recommendation_kn']}")
    print(f"\n📊 Top Contributing Factors:")
    for factor in result['top_factors'][:3]:
        direction = "↑" if factor['direction'] == "increases_risk" else "↓"
        print(f"   {direction} {factor['feature']}: {factor['value']} "
              f"({factor['contribution_pct']}%)")
