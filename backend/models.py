"""
Drishti Health — Pydantic Models

Data models for API requests/responses and database records.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Language(str, Enum):
    ENGLISH = "en"
    KANNADA = "kn"
    HINDI = "hi"


class Urgency(str, Enum):
    ROUTINE = "routine"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ── Request Models ──────────────────────────────────────

class PatientVitals(BaseModel):
    """Patient vital signs for risk assessment."""
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    sex: int = Field(0, ge=0, le=1, description="0=Female, 1=Male")
    bp_systolic: float = Field(..., ge=50, le=250, description="Systolic BP in mmHg")
    bp_diastolic: float = Field(80, ge=30, le=150, description="Diastolic BP in mmHg")
    glucose: float = Field(..., ge=30, le=500, description="Blood glucose in mg/dL")
    hba1c: float = Field(5.7, ge=3, le=20, description="HbA1c percentage")
    bmi: float = Field(25, ge=10, le=60, description="Body Mass Index")
    cholesterol: float = Field(200, ge=50, le=500, description="Total cholesterol mg/dL")
    heart_rate: float = Field(72, ge=30, le=200, description="Heart rate bpm")
    smoking: int = Field(0, ge=0, le=1, description="Smoking status: 0=No, 1=Yes")
    family_history_diabetes: int = Field(0, ge=0, le=1)
    family_history_heart: int = Field(0, ge=0, le=1)
    physical_activity: float = Field(5, ge=0, le=10, description="Activity score 0-10")
    pregnancies: int = Field(0, ge=0, le=20)


class SymptomInput(BaseModel):
    """Patient symptom text input."""
    text: str = Field(..., min_length=1, description="Symptom description")
    language: Language = Field(Language.ENGLISH, description="Input language")


class PatientRecord(BaseModel):
    """Full patient record for storage."""
    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=120)
    sex: str = Field(..., pattern="^(M|F|Other)$")
    abha_id: Optional[str] = Field(None, description="ABHA Health ID")
    phone: Optional[str] = None
    village: Optional[str] = None
    district: Optional[str] = None
    asha_worker_id: Optional[str] = None


# ── Response Models ─────────────────────────────────────

class ContributingFactor(BaseModel):
    """A single risk contributing factor with SHAP value."""
    feature: str
    value: float
    contribution_pct: float
    direction: str  # "increases_risk" or "decreases_risk"


class RiskResult(BaseModel):
    """Risk assessment result."""
    risk_score: float = Field(..., ge=0, le=10)
    risk_probability: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    confidence: float = Field(..., ge=0, le=1)
    recommendation: str
    recommendation_kn: str
    top_factors: List[ContributingFactor]
    shap_values: Optional[List[float]] = None
    feature_names: Optional[List[str]] = None
    input_values: Optional[List[float]] = None


class FundusResult(BaseModel):
    """Fundus DR detection result."""
    dr_grade: int = Field(..., ge=0, le=4)
    dr_label: str
    severity: str
    color: str
    confidence: float = Field(..., ge=0, le=1)
    probabilities: Dict[str, float]
    clinical_findings: List[str]
    recommendation: str
    recommendation_kn: str
    referral_urgency: Urgency
    model_used: str
    image_size: str


class SymptomResult(BaseModel):
    """Symptom classification result."""
    symptoms_detected: List[str]
    risk_factors: List[str]
    icd10_codes: List[str]
    overall_urgency: str
    recommended_examinations: List[str]
    language_detected: str
    english_translation: str


class ScreeningResult(BaseModel):
    """Combined screening result."""
    patient_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    risk_result: Optional[RiskResult] = None
    fundus_result: Optional[FundusResult] = None
    symptom_result: Optional[SymptomResult] = None
    unified_risk_score: float = Field(0, ge=0, le=10)
    unified_recommendation: str = ""
    referral_needed: bool = False


class ABHARecord(BaseModel):
    """ABHA health record sync data."""
    abha_id: str
    screening_id: str
    fhir_resource_type: str = "DiagnosticReport"
    fhir_bundle: Optional[Dict[str, Any]] = None
    sync_status: str = "pending"  # pending, synced, failed
    synced_at: Optional[str] = None
