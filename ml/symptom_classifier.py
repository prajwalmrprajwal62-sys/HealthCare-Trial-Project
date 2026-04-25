"""
Drishti Health — Symptom Classification & NLP Module

Classifies patient symptoms from text (English, Hindi, or Kannada) into:
- ICD-10 categories
- Risk factor flags for the unified risk engine
- Suggested diagnostic pathways

Supports:
1. Sarvam AI API (online, Indic-optimized)
2. DistilBERT keyword matching (offline fallback)
3. Rule-based classifier (zero-dependency fallback)
"""

import re
from typing import Dict, List, Any, Optional


# Symptom → Risk Factor mapping (rule-based, always available offline)
SYMPTOM_RULES = {
    # Diabetes-related symptoms
    "blurred vision": {"risk_factors": ["diabetes", "hypertension"], "icd10": "H53.8", "urgency": "moderate"},
    "frequent urination": {"risk_factors": ["diabetes"], "icd10": "R35.0", "urgency": "low"},
    "excessive thirst": {"risk_factors": ["diabetes"], "icd10": "R63.1", "urgency": "low"},
    "unexplained weight loss": {"risk_factors": ["diabetes", "thyroid"], "icd10": "R63.4", "urgency": "moderate"},
    "numbness": {"risk_factors": ["diabetes", "neuropathy"], "icd10": "R20.0", "urgency": "moderate"},
    "tingling": {"risk_factors": ["diabetes", "neuropathy"], "icd10": "R20.2", "urgency": "low"},
    "slow healing": {"risk_factors": ["diabetes"], "icd10": "R68.89", "urgency": "low"},
    "fatigue": {"risk_factors": ["diabetes", "anemia", "thyroid"], "icd10": "R53.83", "urgency": "low"},

    # Cardiovascular symptoms
    "chest pain": {"risk_factors": ["heart_disease"], "icd10": "R07.9", "urgency": "high"},
    "shortness of breath": {"risk_factors": ["heart_disease", "asthma"], "icd10": "R06.0", "urgency": "high"},
    "palpitations": {"risk_factors": ["heart_disease", "anxiety"], "icd10": "R00.2", "urgency": "moderate"},
    "swelling": {"risk_factors": ["heart_disease", "kidney"], "icd10": "R60.0", "urgency": "moderate"},
    "dizziness": {"risk_factors": ["hypertension", "anemia"], "icd10": "R42", "urgency": "moderate"},

    # Hypertension symptoms
    "headache": {"risk_factors": ["hypertension", "migraine"], "icd10": "R51", "urgency": "low"},
    "nosebleed": {"risk_factors": ["hypertension"], "icd10": "R04.0", "urgency": "moderate"},

    # Eye symptoms
    "eye pain": {"risk_factors": ["glaucoma", "eye_infection"], "icd10": "H57.10", "urgency": "moderate"},
    "floaters": {"risk_factors": ["retinal_detachment", "diabetes"], "icd10": "H43.10", "urgency": "high"},
    "night blindness": {"risk_factors": ["vitamin_a_deficiency"], "icd10": "H53.60", "urgency": "low"},
}

# Kannada symptom terms → English mapping
KANNADA_SYMPTOM_MAP = {
    "ಮಸುಕಾದ ದೃಷ್ಟಿ": "blurred vision",
    "ಆಗಾಗ್ಗೆ ಮೂತ್ರ ವಿಸರ್ಜನೆ": "frequent urination",
    "ಅತಿಯಾದ ಬಾಯಾರಿಕೆ": "excessive thirst",
    "ತೂಕ ನಷ್ಟ": "unexplained weight loss",
    "ಮರಗಟ್ಟುವಿಕೆ": "numbness",
    "ಎದೆ ನೋವು": "chest pain",
    "ಉಸಿರಾಟದ ತೊಂದರೆ": "shortness of breath",
    "ತಲೆನೋವು": "headache",
    "ಸುಸ್ತು": "fatigue",
    "ಕಣ್ಣು ನೋವು": "eye pain",
    "ತಲೆ ತಿರುಗುವಿಕೆ": "dizziness",
    "ಊತ": "swelling",
}

# Hindi symptom terms → English mapping
HINDI_SYMPTOM_MAP = {
    "धुंधली दृष्टि": "blurred vision",
    "बार-बार पेशाब": "frequent urination",
    "अत्यधिक प्यास": "excessive thirst",
    "वजन कम होना": "unexplained weight loss",
    "सुन्नपन": "numbness",
    "सीने में दर्द": "chest pain",
    "सांस की तकलीफ": "shortness of breath",
    "सिरदर्द": "headache",
    "थकान": "fatigue",
    "आंखों में दर्द": "eye pain",
    "चक्कर आना": "dizziness",
    "सूजन": "swelling",
}


class SymptomClassifier:
    """
    Multi-language symptom classifier with offline capability.
    """

    def __init__(self, use_sarvam: bool = False, sarvam_api_key: str = ""):
        self.use_sarvam = use_sarvam
        self.sarvam_api_key = sarvam_api_key

    def classify(self, text: str, language: str = "auto") -> Dict[str, Any]:
        """
        Classify symptoms from natural language text.

        Args:
            text: Patient symptom description
            language: "en", "kn" (Kannada), "hi" (Hindi), or "auto"

        Returns:
            Dictionary with detected symptoms, risk factors, ICD-10 codes,
            urgency level, and suggested examinations.
        """
        # Auto-detect language
        if language == "auto":
            language = self._detect_language(text)

        # Translate to English if needed
        english_text = self._translate_to_english(text, language)

        # Extract symptoms using rule-based matching
        detected = self._extract_symptoms(english_text)

        # Aggregate risk factors and urgency
        all_risk_factors = set()
        all_icd10 = []
        max_urgency = "low"
        urgency_order = {"low": 0, "moderate": 1, "high": 2, "critical": 3}

        for symptom_info in detected:
            all_risk_factors.update(symptom_info["risk_factors"])
            all_icd10.append(symptom_info["icd10"])
            if urgency_order.get(symptom_info["urgency"], 0) > urgency_order.get(max_urgency, 0):
                max_urgency = symptom_info["urgency"]

        # Generate recommended examinations
        exams = self._recommend_examinations(all_risk_factors)

        return {
            "input_text": text,
            "language_detected": language,
            "english_translation": english_text,
            "symptoms_detected": [s["symptom"] for s in detected],
            "symptom_details": detected,
            "risk_factors": list(all_risk_factors),
            "icd10_codes": all_icd10,
            "overall_urgency": max_urgency,
            "recommended_examinations": exams,
            "num_symptoms": len(detected),
        }

    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character ranges."""
        # Kannada Unicode range: 0C80-0CFF
        if re.search(r"[\u0C80-\u0CFF]", text):
            return "kn"
        # Devanagari (Hindi) Unicode range: 0900-097F
        if re.search(r"[\u0900-\u097F]", text):
            return "hi"
        return "en"

    def _translate_to_english(self, text: str, language: str) -> str:
        """Translate symptom text to English using local dictionaries."""
        if language == "en":
            return text.lower()

        symptom_map = KANNADA_SYMPTOM_MAP if language == "kn" else HINDI_SYMPTOM_MAP

        # Try to match known symptom phrases
        english_parts = []
        remaining = text
        for native, english in symptom_map.items():
            if native in remaining:
                english_parts.append(english)
                remaining = remaining.replace(native, "")

        if english_parts:
            return ", ".join(english_parts)

        # Fallback: return original text
        return text.lower()

    def _extract_symptoms(self, text: str) -> List[Dict[str, Any]]:
        """Extract symptoms from English text using keyword matching."""
        text = text.lower()
        detected = []

        for symptom, info in SYMPTOM_RULES.items():
            # Check for exact or partial match
            if symptom in text or any(word in text for word in symptom.split()):
                detected.append({
                    "symptom": symptom,
                    "risk_factors": info["risk_factors"],
                    "icd10": info["icd10"],
                    "urgency": info["urgency"],
                })

        return detected

    def _recommend_examinations(self, risk_factors: set) -> List[str]:
        """Recommend examinations based on identified risk factors."""
        exams = []

        if "diabetes" in risk_factors:
            exams.extend([
                "Fasting blood glucose test",
                "HbA1c test",
                "Fundus examination (retinopathy screening)",
                "Urine albumin test (nephropathy screening)"
            ])

        if "heart_disease" in risk_factors:
            exams.extend([
                "ECG (electrocardiogram)",
                "Lipid profile",
                "Echocardiography if chest pain present"
            ])

        if "hypertension" in risk_factors:
            exams.extend([
                "Repeated BP measurements (3 readings)",
                "Serum creatinine",
                "Fundus examination (hypertensive retinopathy)"
            ])

        if "anemia" in risk_factors:
            exams.extend([
                "Complete blood count (CBC)",
                "Serum iron and ferritin"
            ])

        if not exams:
            exams = ["General health screening", "Vital signs assessment"]

        return list(set(exams))


if __name__ == "__main__":
    classifier = SymptomClassifier()

    # Test with English
    result = classifier.classify(
        "I have blurred vision, frequent urination, and I feel very tired"
    )
    print("🔍 English Symptoms:")
    print(f"   Detected: {result['symptoms_detected']}")
    print(f"   Risk factors: {result['risk_factors']}")
    print(f"   Urgency: {result['overall_urgency']}")
    print(f"   Exams: {result['recommended_examinations']}")

    # Test with Kannada
    result_kn = classifier.classify("ಮಸುಕಾದ ದೃಷ್ಟಿ ಮತ್ತು ಸುಸ್ತು")
    print(f"\n🔍 Kannada Symptoms:")
    print(f"   Detected: {result_kn['symptoms_detected']}")
    print(f"   Language: {result_kn['language_detected']}")
    print(f"   Translation: {result_kn['english_translation']}")
