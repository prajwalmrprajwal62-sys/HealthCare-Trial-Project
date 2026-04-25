"""Full smoke test for Drishti Health V2."""
import sys
import os
from pathlib import Path

# Fix Windows unicode encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

# Shared test data
TEST_VITALS = {
    "age": 48, "sex": 0, "bp_systolic": 155, "bp_diastolic": 95,
    "glucose": 210, "hba1c": 8.2, "bmi": 31.5, "cholesterol": 245,
    "heart_rate": 82, "smoking": 0, "family_history_diabetes": 1,
    "family_history_heart": 0, "physical_activity": 2.0, "pregnancies": 3,
}


def test_all():
    passed = 0
    failed = 0
    risk_result = None
    symptom_result = None

    # 1. Risk Scorer
    try:
        from ml.risk_scorer import RiskScorer
        scorer = RiskScorer()
        risk_result = scorer.predict_risk(TEST_VITALS)
        assert risk_result["risk_score"] > 0
        assert risk_result["risk_level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")
        assert len(risk_result["top_factors"]) > 0
        print(f"[PASS] 1. Risk Scorer: {risk_result['risk_score']}/10 {risk_result['risk_level']}, {len(risk_result['top_factors'])} SHAP factors")
        passed += 1
    except Exception as e:
        print(f"[FAIL] 1. Risk Scorer: {e}")
        failed += 1

    # 2. Symptom Classifier
    try:
        from ml.symptom_classifier import SymptomClassifier
        c = SymptomClassifier()
        symptom_result = c.classify("blurred vision, frequent urination, fatigue")
        assert len(symptom_result["symptoms_detected"]) > 0
        print(f"[PASS] 2. Symptom Classifier: {len(symptom_result['symptoms_detected'])} symptoms, urgency={symptom_result['overall_urgency']}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] 2. Symptom Classifier: {e}")
        failed += 1

    # 3. Fundus Detector + Heatmap
    try:
        from ml.fundus_detector import FundusDetector
        from PIL import Image
        import numpy as np
        det = FundusDetector()
        img = Image.fromarray(np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8))
        fd = det.analyze(img)
        hm = det.generate_heatmap(img, grade=fd["dr_grade"])
        assert fd["dr_grade"] in range(5)
        assert hm.size == (512, 512)
        print(f"[PASS] 3. Fundus + Heatmap: Grade {fd['dr_grade']} ({fd['dr_label']}), heatmap={hm.size}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] 3. Fundus + Heatmap: {e}")
        failed += 1

    # 4. Camera PPG
    try:
        from ml.camera_ppg import CameraPPG
        ppg = CameraPPG()
        pr = ppg.measure_demo()
        assert 40 < pr["heart_rate_bpm"] < 120
        assert "waveform" in pr
        print(f"[PASS] 4. Camera PPG: {pr['heart_rate_bpm']} bpm, quality={pr['signal_quality']}, SpO2={pr.get('spo2_estimate', 'N/A')}%")
        passed += 1
    except Exception as e:
        print(f"[FAIL] 4. Camera PPG: {e}")
        failed += 1

    # 5. Database Auto-Save
    try:
        from backend.database import DrishtiDB
        db = DrishtiDB()
        save = db.save_screening_result(
            vitals=TEST_VITALS,
            risk_result=risk_result or {"risk_score": 7.0, "risk_level": "HIGH", "recommendation": "Test", "recommendation_kn": "Test", "top_factors": []},
            symptom_result=symptom_result,
            patient_name="Test Patient"
        )
        assert "patient_id" in save
        assert "screening_id" in save
        stats = db.get_statistics()
        print(f"[PASS] 5. DB Auto-Save: patient={save['patient_id']}, {stats['total_patients']} total patients")
        passed += 1
    except Exception as e:
        print(f"[FAIL] 5. DB Auto-Save: {e}")
        failed += 1

    # 6. Sarvam AI Summary
    try:
        from integrations.sarvam_client import SarvamClient
        sarvam = SarvamClient()
        rr = risk_result or {"risk_score": 7.0, "risk_level": "HIGH", "recommendation": "Test", "top_factors": []}
        summary = sarvam.generate_patient_summary(risk_result=rr, vitals=TEST_VITALS, language="kn")
        assert len(summary["summary_en"]) > 50
        assert len(summary["summary_kn"]) > 50
        print(f"[PASS] 6. Sarvam Summary: method={summary['method']}, EN={len(summary['summary_en'])}c, KN={len(summary['summary_kn'])}c")
        passed += 1
    except Exception as e:
        print(f"[FAIL] 6. Sarvam Summary: {e}")
        failed += 1

    # 7. Report Generator
    try:
        from ml.report_generator import ReportGenerator
        gen = ReportGenerator()
        rr = risk_result or {"risk_score": 7.0, "risk_level": "HIGH", "recommendation": "Test", "recommendation_kn": "Test", "top_factors": []}
        html = gen.generate_referral_report(
            risk_result=rr, vitals=TEST_VITALS, patient_name="Test", symptom_result=symptom_result
        )
        assert "DRISHTI HEALTH" in html
        assert "SHAP" in html
        assert len(html) > 2000
        print(f"[PASS] 7. Report Generator: {len(html)} chars of HTML")
        passed += 1
    except Exception as e:
        print(f"[FAIL] 7. Report Generator: {e}")
        failed += 1

    # 8. Trained Models
    try:
        models = list(Path("models").glob("*.pkl"))
        assert len(models) >= 3
        total_size = sum(m.stat().st_size for m in models)
        print(f"[PASS] 8. Trained Models: {len(models)} models ({total_size//1024}KB)")
        passed += 1
    except Exception as e:
        print(f"[FAIL] 8. Trained Models: {e}")
        failed += 1

    # 9. Voice Stubs
    try:
        from voice.bhashini_client import BhashiniClient
        from voice.vosk_offline import VoskOfflineSTT
        print(f"[PASS] 9. Voice: Bhashini + Vosk loaded")
        passed += 1
    except Exception as e:
        print(f"[FAIL] 9. Voice: {e}")
        failed += 1

    # 10. ABHA Integration
    try:
        from integrations.abha_client import ABHAClient
        print(f"[PASS] 10. ABHA Client loaded")
        passed += 1
    except Exception as e:
        print(f"[FAIL] 10. ABHA Client: {e}")
        failed += 1

    # Summary
    print()
    print("=" * 60)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED out of 10 modules")
    if failed == 0:
        print("ALL MODULES WORKING -- HACKATHON READY!")
    else:
        print("SOME MODULES NEED FIXING")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)
