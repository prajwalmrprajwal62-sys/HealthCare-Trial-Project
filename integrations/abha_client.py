"""
Drishti Health — ABHA (Ayushman Bharat Health Account) Integration

Integrates with ABDM (Ayushman Bharat Digital Mission) Sandbox APIs
for health record syncing using FHIR R4 standard.

Sandbox: https://sandbox.abdm.gov.in
Docs: https://sandbox.abdm.gov.in/docs/

This module handles:
- ABHA ID verification
- Health record creation (FHIR R4)
- Diagnostic report sync
- Patient consent management (stub)
"""

import os
import json
import httpx
from datetime import datetime
from typing import Dict, Any, Optional


class ABHAClient:
    """
    Client for ABDM Sandbox APIs.

    In production, this connects to:
    - Gateway: https://dev.abdm.gov.in/gateway
    - Health Info: https://dev.abdm.gov.in/cm
    """

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        base_url: str = ""
    ):
        self.client_id = client_id or os.getenv("ABDM_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("ABDM_CLIENT_SECRET", "")
        self.base_url = base_url or os.getenv(
            "ABDM_BASE_URL", "https://dev.abdm.gov.in/gateway"
        )
        self.access_token = None
        self.available = bool(self.client_id and self.client_secret)

        if not self.available:
            print("⚠️  ABDM credentials not set. ABHA sync will use demo mode.")
            print("   Register at: https://sandbox.abdm.gov.in")

    async def authenticate(self) -> bool:
        """Get access token from ABDM gateway."""
        if not self.available:
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v0.5/sessions",
                    json={
                        "clientId": self.client_id,
                        "clientSecret": self.client_secret,
                    },
                    timeout=30,
                )
                data = response.json()
                self.access_token = data.get("accessToken")
                return bool(self.access_token)
        except Exception as e:
            print(f"ABDM auth failed: {e}")
            return False

    def create_fhir_patient(self, patient_data: Dict[str, Any]) -> Dict:
        """
        Create FHIR R4 Patient resource.

        Args:
            patient_data: Patient info dict

        Returns:
            FHIR R4 Patient resource
        """
        return {
            "resourceType": "Patient",
            "meta": {
                "profile": ["https://nrces.in/ndhm/fhir/r4/StructureDefinition/Patient"]
            },
            "identifier": [{
                "system": "https://healthid.abdm.gov.in",
                "value": patient_data.get("abha_id", "pending"),
                "type": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                        "code": "MR",
                        "display": "Medical record number"
                    }]
                }
            }],
            "name": [{
                "text": patient_data.get("name", "Unknown"),
                "family": patient_data.get("name", "").split()[-1] if patient_data.get("name") else "",
                "given": patient_data.get("name", "").split()[:-1] if patient_data.get("name") else [],
            }],
            "gender": self._map_gender(patient_data.get("sex", "")),
            "birthDate": str(datetime.now().year - patient_data.get("age", 30)),
            "address": [{
                "district": patient_data.get("district", ""),
                "state": "Karnataka",
                "country": "IN",
                "text": patient_data.get("village", ""),
            }],
        }

    def create_fhir_observation(
        self, risk_score: float, risk_level: str, vitals: Dict
    ) -> Dict:
        """
        Create FHIR R4 Observation resource for risk assessment.
        """
        return {
            "resourceType": "Observation",
            "meta": {
                "profile": ["https://nrces.in/ndhm/fhir/r4/StructureDefinition/Observation"]
            },
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "survey",
                    "display": "Survey"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "85354-9",
                    "display": "Multi-disease Risk Assessment"
                }],
                "text": "Drishti Health Risk Score"
            },
            "valueQuantity": {
                "value": risk_score,
                "unit": "score",
                "system": "http://unitsofmeasure.org",
            },
            "interpretation": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                    "code": "H" if risk_score > 6 else "N",
                    "display": risk_level
                }]
            }],
            "component": self._vitals_to_fhir_components(vitals),
            "effectiveDateTime": datetime.now().isoformat(),
        }

    def create_fhir_diagnostic_report(
        self, patient_id: str, dr_grade: int, dr_label: str,
        confidence: float, findings: list
    ) -> Dict:
        """
        Create FHIR R4 Diagnostic Report for DR screening.
        """
        return {
            "resourceType": "DiagnosticReport",
            "meta": {
                "profile": ["https://nrces.in/ndhm/fhir/r4/StructureDefinition/DiagnosticReport"]
            },
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                    "code": "IMG",
                    "display": "Diagnostic Imaging"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "134395001",
                    "display": "Diabetic retinopathy screening"
                }],
                "text": "Diabetic Retinopathy AI Screening"
            },
            "conclusion": f"DR Grade {dr_grade}: {dr_label} (Confidence: {confidence*100:.1f}%)",
            "conclusionCode": [{
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": self._dr_grade_to_snomed(dr_grade),
                    "display": dr_label
                }]
            }],
            "effectiveDateTime": datetime.now().isoformat(),
            "presentedForm": [{
                "contentType": "text/plain",
                "data": None,
                "title": "Clinical Findings",
            }],
        }

    def create_fhir_bundle(
        self, patient: Dict, observation: Dict,
        diagnostic_report: Optional[Dict] = None
    ) -> Dict:
        """Create a FHIR R4 Bundle containing all resources."""
        entries = [
            {"resource": patient, "request": {"method": "POST", "url": "Patient"}},
            {"resource": observation, "request": {"method": "POST", "url": "Observation"}},
        ]
        if diagnostic_report:
            entries.append({
                "resource": diagnostic_report,
                "request": {"method": "POST", "url": "DiagnosticReport"}
            })

        return {
            "resourceType": "Bundle",
            "type": "transaction",
            "timestamp": datetime.now().isoformat(),
            "entry": entries,
        }

    async def sync_to_abha(self, bundle: Dict) -> Dict[str, Any]:
        """
        Sync FHIR bundle to ABHA (demo/stub).

        In production, this would:
        1. Get patient consent
        2. Encrypt data per ABDM specs
        3. Push to Health Information Provider (HIP)
        """
        if not self.available:
            return {
                "status": "demo",
                "message": "ABHA sync simulated (no credentials)",
                "bundle_size": len(json.dumps(bundle)),
                "resources_count": len(bundle.get("entry", [])),
                "timestamp": datetime.now().isoformat(),
                "note": "In production, connects to ABDM Sandbox at sandbox.abdm.gov.in"
            }

        # Real sync would go here
        try:
            if not self.access_token:
                await self.authenticate()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v0.5/health-information/hip/request",
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json",
                    },
                    json=bundle,
                    timeout=30,
                )
                return {
                    "status": "synced",
                    "response_code": response.status_code,
                    "timestamp": datetime.now().isoformat(),
                }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _map_gender(self, sex: str) -> str:
        mapping = {"M": "male", "F": "female", "Male": "male", "Female": "female"}
        return mapping.get(sex, "unknown")

    def _dr_grade_to_snomed(self, grade: int) -> str:
        snomed_map = {
            0: "312903003",  # No retinopathy
            1: "312904009",  # Mild nonproliferative
            2: "312905005",  # Moderate nonproliferative
            3: "312906006",  # Severe nonproliferative
            4: "312907002",  # Proliferative
        }
        return snomed_map.get(grade, "312903003")

    def _vitals_to_fhir_components(self, vitals: Dict) -> list:
        components = []
        vital_loinc = {
            "bp_systolic": ("8480-6", "Systolic blood pressure", "mmHg"),
            "bp_diastolic": ("8462-4", "Diastolic blood pressure", "mmHg"),
            "glucose": ("2339-0", "Blood glucose", "mg/dL"),
            "hba1c": ("4548-4", "Hemoglobin A1c", "%"),
            "bmi": ("39156-5", "Body mass index", "kg/m2"),
            "heart_rate": ("8867-4", "Heart rate", "beats/min"),
            "cholesterol": ("2093-3", "Total cholesterol", "mg/dL"),
        }

        for key, (code, display, unit) in vital_loinc.items():
            if key in vitals:
                components.append({
                    "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}]},
                    "valueQuantity": {"value": vitals[key], "unit": unit},
                })

        return components


if __name__ == "__main__":
    client = ABHAClient()
    print(f"ABHA API available: {client.available}")

    # Demo: Create FHIR resources
    patient = client.create_fhir_patient({
        "name": "Meena K.",
        "age": 48,
        "sex": "F",
        "village": "Mandya",
        "district": "Mandya",
        "abha_id": "12-3456-7890-1234",
    })

    observation = client.create_fhir_observation(
        risk_score=7.2,
        risk_level="HIGH",
        vitals={"bp_systolic": 155, "glucose": 210, "hba1c": 8.2, "bmi": 31.5}
    )

    dr_report = client.create_fhir_diagnostic_report(
        patient_id="12345",
        dr_grade=2,
        dr_label="Moderate NPDR",
        confidence=0.78,
        findings=["Multiple microaneurysms", "Hard exudates"]
    )

    bundle = client.create_fhir_bundle(patient, observation, dr_report)
    print(json.dumps(bundle, indent=2)[:500])
