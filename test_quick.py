"""Quick smoke test for all Drishti Health modules."""
import sys
sys.path.insert(0, ".")

print("=" * 50)
print("DRISHTI HEALTH — Quick Smoke Test")
print("=" * 50)

# Test 1: Risk Scorer
print("\n[1/4] Testing Risk Scorer...")
from ml.risk_scorer import RiskScorer
scorer = RiskScorer()
result = scorer.predict_risk({
    "age": 48, "bp_systolic": 155, "glucose": 210,
    "hba1c": 8.2, "bmi": 31.5, "family_history_diabetes": 1
})
print(f"  ✅ Risk Score: {result['risk_score']}/10 ({result['risk_level']})")
print(f"  ✅ Top factor: {result['top_factors'][0]['feature']} ({result['top_factors'][0]['contribution_pct']}%)")

# Test 2: Fundus Detector
print("\n[2/4] Testing Fundus Detector...")
from ml.fundus_detector import FundusDetector
from PIL import Image
import numpy as np
detector = FundusDetector()
test_img = Image.fromarray(np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8))
dr_result = detector.analyze(test_img)
print(f"  ✅ DR Grade: {dr_result['dr_grade']} — {dr_result['dr_label']}")
print(f"  ✅ Confidence: {dr_result['confidence']*100:.1f}%")

# Test 3: Symptom Classifier
print("\n[3/4] Testing Symptom Classifier...")
from ml.symptom_classifier import SymptomClassifier
classifier = SymptomClassifier()
sym_result = classifier.classify("blurred vision, frequent urination, fatigue")
print(f"  ✅ Symptoms: {sym_result['symptoms_detected']}")
print(f"  ✅ Risk factors: {sym_result['risk_factors']}")

# Test Kannada
sym_kn = classifier.classify("ಮಸುಕಾದ ದೃಷ್ಟಿ, ಸುಸ್ತು")
print(f"  ✅ Kannada: {sym_kn['language_detected']} → {sym_kn['english_translation']}")

# Test 4: Database
print("\n[4/4] Testing Database...")
from backend.database import DrishtiDB
db = DrishtiDB()
db.seed_demo_data()
stats = db.get_statistics()
print(f"  ✅ Patients: {stats['total_patients']}")
print(f"  ✅ Screenings: {stats['total_screenings']}")
print(f"  ✅ Avg Risk: {stats['average_risk_score']}/10")

print("\n" + "=" * 50)
print("ALL TESTS PASSED ✅")
print("=" * 50)
