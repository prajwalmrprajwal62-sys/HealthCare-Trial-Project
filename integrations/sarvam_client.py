"""
Sarvam AI Integration — India's First Open-Source Indic LLM

Uses Sarvam-1/Sarvam-2B for generating patient-friendly health summaries
in Indian languages. Bengaluru-based, trained on 4 trillion tokens.
Selected under IndiaAI Mission by MeitY.

Falls back to rule-based template generation when API is unavailable.
"""

import os
import json
from typing import Optional

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


class SarvamClient:
    """Sarvam AI API client for Indian language health summaries."""

    BASE_URL = "https://api.sarvam.ai"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.available = bool(self.api_key) and HAS_HTTPX

    def generate_patient_summary(
        self,
        risk_result: dict,
        vitals: dict,
        symptom_result: dict = None,
        fundus_result: dict = None,
        language: str = "en",
    ) -> dict:
        """
        Generate a patient-friendly health summary from screening results.

        Args:
            risk_result: Risk scoring output with score, factors, recommendations
            vitals: Patient vitals dictionary
            symptom_result: Optional symptom classification result
            fundus_result: Optional fundus analysis result
            language: Target language ('en', 'kn', 'hi')

        Returns:
            dict with summary text, language, and method used
        """
        if self.available:
            try:
                return self._api_generate(risk_result, vitals, symptom_result, fundus_result, language)
            except Exception:
                pass

        # Fallback: rule-based template (works offline)
        return self._template_generate(risk_result, vitals, symptom_result, fundus_result, language)

    def _api_generate(self, risk_result, vitals, symptom_result, fundus_result, language) -> dict:
        """Generate summary using Sarvam AI API."""
        prompt = self._build_prompt(risk_result, vitals, symptom_result, fundus_result)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "sarvam-2b-v0.5",
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.3,
            "language": language,
        }

        with httpx.Client(timeout=10) as client:
            resp = client.post(f"{self.BASE_URL}/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return {
            "summary_en": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "summary_local": "",  # Would be translated via Sarvam translate API
            "language": language,
            "method": "Sarvam AI (sarvam-2b-v0.5)",
            "model": "sarvam-2b",
        }

    def _build_prompt(self, risk_result, vitals, symptom_result, fundus_result) -> str:
        """Build prompt for Sarvam AI."""
        prompt = f"""You are a healthcare assistant generating a simple, patient-friendly health summary.

Patient Details:
- Age: {vitals.get('age', 'N/A')} years
- Blood Pressure: {vitals.get('bp_systolic', 'N/A')}/{vitals.get('bp_diastolic', 'N/A')} mmHg
- Blood Glucose: {vitals.get('glucose', 'N/A')} mg/dL
- HbA1c: {vitals.get('hba1c', 'N/A')}%
- BMI: {vitals.get('bmi', 'N/A')}

Risk Assessment:
- Overall Risk Score: {risk_result.get('risk_score', 'N/A')}/10
- Risk Level: {risk_result.get('risk_level', 'N/A')}
- Top Risk Factors: {', '.join(f['feature'] for f in risk_result.get('top_factors', [])[:3])}
"""
        if fundus_result:
            prompt += f"""
Eye Examination:
- DR Grade: {fundus_result.get('dr_grade', 'N/A')} - {fundus_result.get('dr_label', 'N/A')}
"""
        prompt += """
Generate a brief, caring health summary that:
1. Explains the key findings in simple language
2. Lists 3 specific action items
3. Includes when to see a doctor
Keep it under 150 words. Use empathetic, non-alarming language."""
        return prompt

    def _template_generate(self, risk_result, vitals, symptom_result, fundus_result, language) -> dict:
        """Rule-based template generation (offline fallback)."""
        score = risk_result.get("risk_score", 0)
        level = risk_result.get("risk_level", "UNKNOWN")
        factors = risk_result.get("top_factors", [])
        top_factor_names = [f["feature"].replace("_", " ").title() for f in factors[:3]]

        # English summary
        if level == "CRITICAL":
            urgency = "requires immediate medical attention"
            action = "Please visit the nearest hospital or taluk medical center within 24 hours"
        elif level == "HIGH":
            urgency = "shows elevated health risks"
            action = "Please schedule a doctor's appointment within the next week"
        elif level == "MODERATE":
            urgency = "shows some areas that need attention"
            action = "Please consult a doctor during your next visit"
        else:
            urgency = "appears generally healthy"
            action = "Continue your current healthy lifestyle"

        summary_en = f"""Dear Patient,

Your health screening {urgency}. Your overall risk score is {score}/10 ({level}).

Key Findings:
- The main factors contributing to your risk are: {', '.join(top_factor_names)}
- Blood Pressure: {vitals.get('bp_systolic', 'N/A')}/{vitals.get('bp_diastolic', 'N/A')} mmHg {'(elevated)' if vitals.get('bp_systolic', 0) > 140 else '(normal range)'}
- Blood Glucose: {vitals.get('glucose', 'N/A')} mg/dL {'(high)' if vitals.get('glucose', 0) > 180 else '(acceptable)'}"""

        if fundus_result:
            dr_label = fundus_result.get("dr_label", "Not assessed")
            summary_en += f"\n- Eye Examination: {dr_label}"

        summary_en += f"""

Recommended Actions:
1. {action}
2. Monitor blood pressure and blood sugar daily
3. Maintain a balanced diet and walk for 30 minutes daily

This report was generated by Drishti Health AI and should be reviewed by a qualified doctor.

— Drishti Health, AI Diagnostic Co-Pilot"""

        # Kannada summary
        summary_kn = self._kannada_summary(score, level, top_factor_names, vitals, fundus_result)

        # Hindi summary
        summary_hi = self._hindi_summary(score, level, top_factor_names, vitals, fundus_result)

        result = {
            "summary_en": summary_en,
            "summary_kn": summary_kn,
            "summary_hi": summary_hi,
            "language": language,
            "method": "Rule-based Template (Offline)",
            "model": "template-v1",
        }

        # Return the requested language summary as primary
        lang_map = {"en": "summary_en", "kn": "summary_kn", "hi": "summary_hi"}
        result["summary_local"] = result.get(lang_map.get(language, "summary_en"), summary_en)

        return result

    def _kannada_summary(self, score, level, factors, vitals, fundus_result) -> str:
        """Generate Kannada health summary."""
        level_kn = {
            "CRITICAL": "ಅತ್ಯಂತ ಗಂಭೀರ",
            "HIGH": "ಹೆಚ್ಚಿನ ಅಪಾಯ",
            "MODERATE": "ಮಧ್ಯಮ ಅಪಾಯ",
            "LOW": "ಕಡಿಮೆ ಅಪಾಯ",
        }.get(level, level)

        summary = f"""ಆತ್ಮೀಯ ರೋಗಿಯವರೆ,

ನಿಮ್ಮ ಆರೋಗ್ಯ ತಪಾಸಣೆಯ ಫಲಿತಾಂಶ:
- ಒಟ್ಟಾರೆ ಅಪಾಯ ಅಂಕ: {score}/10 ({level_kn})
- ರಕ್ತದೊತ್ತಡ: {vitals.get('bp_systolic', 'N/A')}/{vitals.get('bp_diastolic', 'N/A')} mmHg
- ರಕ್ತದ ಗ್ಲೂಕೋಸ್: {vitals.get('glucose', 'N/A')} mg/dL

ಶಿಫಾರಸುಗಳು:
1. ದಯವಿಟ್ಟು ಸಮೀಪದ ಆಸ್ಪತ್ರೆಗೆ ಭೇಟಿ ನೀಡಿ
2. ಪ್ರತಿದಿನ ರಕ್ತದೊತ್ತಡ ಮತ್ತು ಸಕ್ಕರೆ ಪರಿಶೀಲಿಸಿ
3. ಸಮತೋಲಿತ ಆಹಾರ ಮತ್ತು 30 ನಿಮಿಷ ನಡಿಗೆ

— ದೃಷ್ಟಿ ಹೆಲ್ತ್, AI ರೋಗನಿರ್ಣಯ ಸಹಾಯಕ"""
        return summary

    def _hindi_summary(self, score, level, factors, vitals, fundus_result) -> str:
        """Generate Hindi health summary."""
        level_hi = {
            "CRITICAL": "अत्यंत गंभीर",
            "HIGH": "उच्च जोखिम",
            "MODERATE": "मध्यम जोखिम",
            "LOW": "कम जोखिम",
        }.get(level, level)

        summary = f"""प्रिय रोगी,

आपकी स्वास्थ्य जांच का परिणाम:
- समग्र जोखिम अंक: {score}/10 ({level_hi})
- रक्तचाप: {vitals.get('bp_systolic', 'N/A')}/{vitals.get('bp_diastolic', 'N/A')} mmHg
- रक्त ग्लूकोज: {vitals.get('glucose', 'N/A')} mg/dL

सिफारिशें:
1. कृपया निकटतम अस्पताल जाएं
2. प्रतिदिन रक्तचाप और शुगर की जांच करें
3. संतुलित आहार और 30 मिनट पैदल चलें

— दृष्टि हेल्थ, AI डायग्नोस्टिक सहायक"""
        return summary

    def translate_text(self, text: str, source_lang: str = "en", target_lang: str = "kn") -> str:
        """
        Translate text using Sarvam Translate API.
        Falls back to returning original text if API unavailable.
        """
        if not self.available:
            return text

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "input": text,
                "source_language_code": source_lang,
                "target_language_code": target_lang,
                "model": "sarvam-translate",
            }
            with httpx.Client(timeout=10) as client:
                resp = client.post(f"{self.BASE_URL}/translate", headers=headers, json=payload)
                resp.raise_for_status()
                return resp.json().get("translated_text", text)
        except Exception:
            return text
