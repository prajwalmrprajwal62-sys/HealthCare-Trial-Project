"""
Referral Report Generator — Drishti Health

Generates professional HTML referral reports with:
- Patient demographics and vitals
- Risk score with SHAP factor breakdown
- DR grade (if fundus analyzed)
- Recommendations and urgency
- ABHA ID placeholder
- QR code for report verification
"""

import base64
import hashlib
from datetime import datetime
from typing import Optional


class ReportGenerator:
    """Generate downloadable referral reports from screening results."""

    def generate_referral_report(
        self,
        risk_result: dict,
        vitals: dict,
        patient_name: str = "Patient",
        patient_age: int = 0,
        symptom_result: dict = None,
        fundus_result: dict = None,
        abha_id: str = "XXXX-XXXX-XXXX-XXXX",
    ) -> str:
        """
        Generate a complete HTML referral report.

        Returns:
            HTML string of the report, ready for rendering or download.
        """
        now = datetime.now()
        report_id = hashlib.md5(f"{patient_name}{now.isoformat()}".encode()).hexdigest()[:8].upper()

        score = risk_result.get("risk_score", 0)
        level = risk_result.get("risk_level", "UNKNOWN")
        factors = risk_result.get("top_factors", [])
        recommendation = risk_result.get("recommendation", "Consult a healthcare professional.")
        recommendation_kn = risk_result.get("recommendation_kn", "")

        # Color coding
        color_map = {"LOW": "#4CAF50", "MODERATE": "#FF9800", "HIGH": "#F44336", "CRITICAL": "#9C27B0"}
        color = color_map.get(level, "#888")

        # Urgency mapping
        urgency_map = {
            "LOW": "Routine follow-up",
            "MODERATE": "Schedule appointment within 2 weeks",
            "HIGH": "Urgent — consult within 48 hours",
            "CRITICAL": "EMERGENCY — Immediate referral to district hospital",
        }
        urgency = urgency_map.get(level, "Consult doctor")

        # Build factor rows
        factor_rows = ""
        for f in factors:
            direction = "⬆️ Increases Risk" if f.get("direction") == "increases_risk" else "⬇️ Decreases Risk"
            row_color = "#ffebee" if f.get("direction") == "increases_risk" else "#e8f5e9"
            factor_rows += f"""
                <tr style="background: {row_color};">
                    <td style="padding: 8px; border: 1px solid #ddd;">{f['feature'].replace('_', ' ').title()}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{f['value']}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{f['contribution_pct']}%</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{direction}</td>
                </tr>"""

        # Fundus section
        fundus_section = ""
        if fundus_result:
            dr_color = fundus_result.get("color", "#888")
            fundus_section = f"""
            <div style="margin-top: 20px; padding: 15px; border: 2px solid {dr_color}; border-radius: 8px;">
                <h3 style="color: {dr_color}; margin: 0;">🔬 Fundoscopy Results</h3>
                <table style="width: 100%; margin-top: 10px;">
                    <tr><td><b>DR Grade:</b></td><td>{fundus_result.get('dr_grade', 'N/A')} — {fundus_result.get('dr_label', 'N/A')}</td></tr>
                    <tr><td><b>Severity:</b></td><td>{fundus_result.get('severity', 'N/A')}</td></tr>
                    <tr><td><b>Confidence:</b></td><td>{fundus_result.get('confidence', 0)*100:.1f}%</td></tr>
                    <tr><td><b>Findings:</b></td><td>{', '.join(fundus_result.get('clinical_findings', []))}</td></tr>
                    <tr><td><b>Recommendation:</b></td><td>{fundus_result.get('recommendation', 'N/A')}</td></tr>
                </table>
            </div>"""

        # Symptom section
        symptom_section = ""
        if symptom_result:
            symptoms = ", ".join(symptom_result.get("symptoms_detected", []))
            risk_factors = ", ".join(symptom_result.get("risk_factors", []))
            symptom_section = f"""
            <div style="margin-top: 20px; padding: 15px; border: 1px solid #2196F3; border-radius: 8px;">
                <h3 style="color: #2196F3; margin: 0;">🗣️ Symptom Analysis</h3>
                <p><b>Symptoms Reported:</b> {symptoms}</p>
                <p><b>Associated Risk Factors:</b> {risk_factors}</p>
                <p><b>Urgency:</b> {symptom_result.get('overall_urgency', 'N/A')}</p>
            </div>"""

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Drishti Health — Referral Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; color: #333; }}
        .header {{ text-align: center; border-bottom: 3px solid #00BFA6; padding-bottom: 15px; margin-bottom: 20px; }}
        .header h1 {{ color: #00BFA6; margin: 0; font-size: 24px; }}
        .header p {{ color: #666; margin: 5px 0; }}
        .risk-badge {{ display: inline-block; background: {color}; color: white; padding: 10px 25px; border-radius: 25px; font-size: 20px; font-weight: bold; }}
        .section {{ margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 8px; }}
        .urgent {{ background: #fff3e0; border-left: 4px solid {color}; padding: 12px; margin: 15px 0; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th {{ background: #00BFA6; color: white; padding: 10px; text-align: left; }}
        .footer {{ text-align: center; margin-top: 30px; padding-top: 15px; border-top: 2px solid #eee; color: #888; font-size: 12px; }}
        .qr-placeholder {{ display: inline-block; width: 80px; height: 80px; border: 2px solid #ccc; border-radius: 8px; text-align: center; line-height: 80px; color: #aaa; font-size: 10px; }}
        @media print {{ body {{ margin: 0; }} .no-print {{ display: none; }} }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 DRISHTI HEALTH</h1>
        <p>AI-Assisted Diagnostic Screening Report</p>
        <p style="font-size: 12px; color: #999;">Report ID: DH-{report_id} | Generated: {now.strftime('%d %b %Y, %I:%M %p')}</p>
    </div>

    <!-- Patient Info -->
    <div class="section">
        <h3 style="margin-top: 0; color: #00BFA6;">👤 Patient Information</h3>
        <table>
            <tr><td style="width: 30%;"><b>Name:</b></td><td>{patient_name}</td><td style="width: 30%;"><b>Age:</b></td><td>{patient_age or vitals.get('age', 'N/A')} years</td></tr>
            <tr><td><b>ABHA ID:</b></td><td>{abha_id}</td><td><b>Date:</b></td><td>{now.strftime('%d-%m-%Y')}</td></tr>
            <tr><td><b>Screening Location:</b></td><td colspan="3">PHC / ASHA Worker Visit — Mandya District, Karnataka</td></tr>
        </table>
    </div>

    <!-- Risk Score -->
    <div style="text-align: center; margin: 25px 0;">
        <h2 style="margin-bottom: 10px;">Overall Risk Assessment</h2>
        <div class="risk-badge">{score}/10 — {level}</div>
        <div class="urgent">⚠️ {urgency}</div>
    </div>

    <!-- Vitals -->
    <div class="section">
        <h3 style="margin-top: 0; color: #00BFA6;">📋 Vitals Recorded</h3>
        <table>
            <tr><td><b>Blood Pressure:</b></td><td>{vitals.get('bp_systolic', 'N/A')}/{vitals.get('bp_diastolic', 'N/A')} mmHg</td>
                <td><b>Blood Glucose:</b></td><td>{vitals.get('glucose', 'N/A')} mg/dL</td></tr>
            <tr><td><b>HbA1c:</b></td><td>{vitals.get('hba1c', 'N/A')}%</td>
                <td><b>BMI:</b></td><td>{vitals.get('bmi', 'N/A')}</td></tr>
            <tr><td><b>Heart Rate:</b></td><td>{vitals.get('heart_rate', 'N/A')} bpm</td>
                <td><b>Cholesterol:</b></td><td>{vitals.get('cholesterol', 'N/A')} mg/dL</td></tr>
        </table>
    </div>

    <!-- SHAP Factors -->
    <div class="section">
        <h3 style="margin-top: 0; color: #00BFA6;">📊 Risk Factor Analysis (SHAP Explainability)</h3>
        <p style="color: #666; font-size: 13px;">AI-generated breakdown showing which factors contributed most to the risk score.</p>
        <table>
            <thead>
                <tr><th>Factor</th><th>Value</th><th>Contribution</th><th>Direction</th></tr>
            </thead>
            <tbody>
                {factor_rows}
            </tbody>
        </table>
    </div>

    {fundus_section}
    {symptom_section}

    <!-- Recommendation -->
    <div style="margin-top: 20px; padding: 20px; background: linear-gradient(135deg, #e8f5e9, #f9fbe7); border-radius: 12px; border: 1px solid #4CAF50;">
        <h3 style="color: #2E7D32; margin-top: 0;">✅ Recommendation</h3>
        <p style="font-size: 16px; font-weight: bold;">{recommendation}</p>
        <p style="font-style: italic; color: #555;">{recommendation_kn}</p>
    </div>

    <!-- Disclaimer -->
    <div style="margin-top: 20px; padding: 12px; background: #fff8e1; border-radius: 8px; font-size: 12px; color: #f57f17;">
        <b>⚠️ Disclaimer:</b> This report is AI-generated and intended for screening purposes only. It does not constitute a medical diagnosis.
        All findings must be verified by a qualified medical practitioner. Drishti Health AI is a decision-support tool, not a replacement for clinical judgment.
    </div>

    <!-- Footer -->
    <div class="footer">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="text-align: left;">
                <p><b>Screening by:</b> ASHA Worker (Drishti Health App)</p>
                <p><b>Technology:</b> XGBoost + SHAP | RETFound | Bhashini API</p>
            </div>
            <div class="qr-placeholder">QR Code<br>{report_id}</div>
        </div>
        <p style="margin-top: 10px;">Drishti Health — AI Diagnostic Co-Pilot for Rural India | Team Cognivex | Vibeathon Mysore 2026</p>
    </div>
</body>
</html>"""

        return html
