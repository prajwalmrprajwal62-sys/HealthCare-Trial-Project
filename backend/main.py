"""
Drishti Health — FastAPI Backend

REST API for:
- Fundus image analysis
- Risk score computation with SHAP
- Symptom classification
- Patient records (CRUD)
- ABHA sync stub
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import json
from datetime import datetime

from backend.models import (
    PatientVitals, SymptomInput, PatientRecord,
    RiskResult, FundusResult, SymptomResult, ScreeningResult
)
from backend.database import DrishtiDB
from ml.risk_scorer import RiskScorer
from ml.fundus_detector import FundusDetector
from ml.symptom_classifier import SymptomClassifier

# ── App Setup ───────────────────────────────────────────

app = FastAPI(
    title="Drishti Health API",
    description="Offline-capable multi-disease diagnostic co-pilot for ASHA workers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize Services ────────────────────────────────

db = DrishtiDB()
risk_scorer = RiskScorer()
fundus_detector = FundusDetector()
symptom_classifier = SymptomClassifier()


# ── Health Check ────────────────────────────────────────

@app.get("/")
def root():
    return {
        "app": "Drishti Health API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/api/risk-score",
            "/api/analyze-fundus",
            "/api/classify-symptoms",
            "/api/patients",
            "/api/screenings",
            "/api/statistics",
        ]
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ── Risk Score Endpoint ─────────────────────────────────

@app.post("/api/risk-score")
def compute_risk_score(vitals: PatientVitals):
    """Compute unified risk score from patient vitals with SHAP explainability."""
    try:
        vitals_dict = vitals.model_dump()
        result = risk_scorer.predict_risk(vitals_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk scoring failed: {str(e)}")


# ── Fundus Analysis Endpoint ───────────────────────────

@app.post("/api/analyze-fundus")
async def analyze_fundus(file: UploadFile = File(...)):
    """Analyze retinal fundus image for diabetic retinopathy."""
    try:
        # Validate file type
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        result = fundus_detector.analyze(image)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fundus analysis failed: {str(e)}")


# ── Symptom Classification ─────────────────────────────

@app.post("/api/classify-symptoms")
def classify_symptoms(symptom_input: SymptomInput):
    """Classify patient symptoms from text (supports Kannada, Hindi, English)."""
    try:
        result = symptom_classifier.classify(
            symptom_input.text,
            symptom_input.language.value
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Symptom classification failed: {str(e)}")


# ── Combined Screening ─────────────────────────────────

@app.post("/api/screen")
async def full_screening(
    vitals: PatientVitals,
    symptoms: str = "",
    patient_id: str = None,
):
    """Run full screening: vitals → risk + symptoms → unified result."""
    try:
        # Risk scoring
        risk_result = risk_scorer.predict_risk(vitals.model_dump())

        # Symptom classification (if provided)
        symptom_result = None
        if symptoms:
            symptom_result = symptom_classifier.classify(symptoms)

        # Compute unified risk (combine risk model + symptom urgency)
        unified_score = risk_result["risk_score"]
        if symptom_result and symptom_result["overall_urgency"] in ["high", "critical"]:
            unified_score = min(10, unified_score + 1.5)

        # Determine recommendation
        referral_needed = unified_score > 6

        # Save to database if patient_id provided
        screening_id = None
        if patient_id:
            screening_id = db.save_screening(patient_id, {
                "screening_type": "combined",
                "risk_score": unified_score,
                "risk_level": risk_result["risk_level"],
                "symptoms": symptom_result["symptoms_detected"] if symptom_result else [],
                "vitals": vitals.model_dump(),
                "recommendation": risk_result["recommendation"],
                "referral_needed": referral_needed,
            })

        return {
            "screening_id": screening_id,
            "unified_risk_score": round(unified_score, 1),
            "risk_result": risk_result,
            "symptom_result": symptom_result,
            "referral_needed": referral_needed,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screening failed: {str(e)}")


# ── Patient Records ─────────────────────────────────────

@app.post("/api/patients")
def create_patient(patient: PatientRecord):
    """Create a new patient record."""
    patient_id = db.create_patient(patient.model_dump())
    return {"patient_id": patient_id, "message": "Patient created successfully"}


@app.get("/api/patients")
def list_patients():
    """List all patients."""
    return db.get_all_patients()


@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: str):
    """Get patient by ID."""
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@app.get("/api/patients/{patient_id}/screenings")
def get_patient_screenings(patient_id: str):
    """Get all screenings for a patient."""
    return db.get_patient_screenings(patient_id)


# ── Statistics ──────────────────────────────────────────

@app.get("/api/statistics")
def get_statistics():
    """Get screening statistics for dashboard."""
    return db.get_statistics()


# ── ABHA Sync Stub ──────────────────────────────────────

@app.post("/api/abha-sync/{patient_id}")
def sync_to_abha(patient_id: str):
    """
    Stub: Sync patient records to ABHA (Ayushman Bharat Health Account).
    In production, this would call ABDM Sandbox APIs (FHIR R4).
    """
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Create FHIR R4 resource (stub)
    fhir_patient = {
        "resourceType": "Patient",
        "identifier": [{"system": "https://healthid.abdm.gov.in", "value": patient.get("abha_id", "pending")}],
        "name": [{"text": patient["name"]}],
        "gender": "female" if patient.get("sex") == "F" else "male",
        "birthDate": str(2026 - (patient.get("age") or 30)),
    }

    # Get latest screening
    screenings = db.get_patient_screenings(patient_id)
    fhir_observation = None
    if screenings:
        latest = screenings[0]
        fhir_observation = {
            "resourceType": "Observation",
            "status": "final",
            "code": {"coding": [{"system": "http://loinc.org", "code": "85354-9", "display": "Risk Assessment"}]},
            "valueQuantity": {"value": latest.get("risk_score", 0), "unit": "score"},
        }

    # Add to sync queue
    db.add_to_sync_queue(
        record_type="patient",
        record_id=patient_id,
        abha_id=patient.get("abha_id", "pending"),
        fhir_resource={"patient": fhir_patient, "observation": fhir_observation}
    )

    return {
        "status": "queued",
        "message": f"Patient {patient_id} queued for ABHA sync",
        "fhir_patient": fhir_patient,
        "fhir_observation": fhir_observation,
        "note": "In production, this calls ABDM Sandbox at sandbox.abdm.gov.in"
    }


# ── Seed Demo Data ──────────────────────────────────────

@app.post("/api/seed-demo")
def seed_demo_data():
    """Seed database with demo data for presentation."""
    db.seed_demo_data()
    return {"status": "success", "message": "Demo data seeded (25 patients)"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
