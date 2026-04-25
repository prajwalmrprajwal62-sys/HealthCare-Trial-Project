"""
Drishti Health — SQLite Database Layer

Offline-first database for patient records, screenings, and referrals.
Syncs to ABHA when connectivity is available.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


class DrishtiDB:
    """SQLite database manager for offline-first patient data."""

    def __init__(self, db_path: str = "data/drishti.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS patients (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    age INTEGER,
                    sex TEXT,
                    abha_id TEXT,
                    phone TEXT,
                    village TEXT,
                    district TEXT,
                    asha_worker_id TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS screenings (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    screening_type TEXT,  -- 'fundus', 'vitals', 'symptoms', 'combined'
                    risk_score REAL,
                    risk_level TEXT,
                    dr_grade INTEGER,
                    symptoms TEXT,  -- JSON
                    vitals TEXT,  -- JSON
                    result_data TEXT,  -- Full result JSON
                    recommendation TEXT,
                    referral_needed INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (patient_id) REFERENCES patients(id)
                );

                CREATE TABLE IF NOT EXISTS referrals (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    screening_id TEXT NOT NULL,
                    referred_to TEXT,
                    urgency TEXT,
                    status TEXT DEFAULT 'pending',  -- pending, completed, cancelled
                    notes TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    completed_at TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients(id),
                    FOREIGN KEY (screening_id) REFERENCES screenings(id)
                );

                CREATE TABLE IF NOT EXISTS sync_queue (
                    id TEXT PRIMARY KEY,
                    record_type TEXT,  -- 'patient', 'screening', 'referral'
                    record_id TEXT,
                    abha_id TEXT,
                    fhir_resource TEXT,  -- JSON
                    status TEXT DEFAULT 'pending',
                    attempts INTEGER DEFAULT 0,
                    last_attempt TEXT,
                    synced_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_screenings_patient ON screenings(patient_id);
                CREATE INDEX IF NOT EXISTS idx_referrals_patient ON referrals(patient_id);
                CREATE INDEX IF NOT EXISTS idx_sync_status ON sync_queue(status);
            """)

    def create_patient(self, patient_data: Dict[str, Any]) -> str:
        """Create a new patient record. Returns patient ID."""
        patient_id = str(uuid.uuid4())[:8]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO patients (id, name, age, sex, abha_id, phone, village, district, asha_worker_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    patient_id,
                    patient_data.get("name", "Unknown"),
                    patient_data.get("age"),
                    patient_data.get("sex"),
                    patient_data.get("abha_id"),
                    patient_data.get("phone"),
                    patient_data.get("village"),
                    patient_data.get("district"),
                    patient_data.get("asha_worker_id"),
                )
            )
        return patient_id

    def save_screening(self, patient_id: str, screening_data: Dict[str, Any]) -> str:
        """Save a screening result. Returns screening ID."""
        screening_id = str(uuid.uuid4())[:8]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO screenings
                   (id, patient_id, screening_type, risk_score, risk_level, dr_grade,
                    symptoms, vitals, result_data, recommendation, referral_needed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    screening_id,
                    patient_id,
                    screening_data.get("screening_type", "combined"),
                    screening_data.get("risk_score"),
                    screening_data.get("risk_level"),
                    screening_data.get("dr_grade"),
                    json.dumps(screening_data.get("symptoms", [])),
                    json.dumps(screening_data.get("vitals", {})),
                    json.dumps(screening_data),
                    screening_data.get("recommendation"),
                    1 if screening_data.get("referral_needed") else 0,
                )
            )
        return screening_id

    def create_referral(
        self, patient_id: str, screening_id: str,
        referred_to: str, urgency: str, notes: str = ""
    ) -> str:
        """Create a referral record. Returns referral ID."""
        referral_id = str(uuid.uuid4())[:8]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO referrals (id, patient_id, screening_id, referred_to, urgency, notes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (referral_id, patient_id, screening_id, referred_to, urgency, notes)
            )
        return referral_id

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get patient record by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
            return dict(row) if row else None

    def get_patient_screenings(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get all screenings for a patient."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM screenings WHERE patient_id = ? ORDER BY created_at DESC",
                (patient_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_all_patients(self) -> List[Dict[str, Any]]:
        """Get all patient records."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    def get_statistics(self) -> Dict[str, Any]:
        """Get screening statistics for dashboard."""
        with sqlite3.connect(self.db_path) as conn:
            total_patients = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
            total_screenings = conn.execute("SELECT COUNT(*) FROM screenings").fetchone()[0]
            total_referrals = conn.execute("SELECT COUNT(*) FROM referrals").fetchone()[0]
            pending_referrals = conn.execute(
                "SELECT COUNT(*) FROM referrals WHERE status = 'pending'"
            ).fetchone()[0]

            # Risk distribution
            risk_dist = {}
            for level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]:
                count = conn.execute(
                    "SELECT COUNT(*) FROM screenings WHERE risk_level = ?", (level,)
                ).fetchone()[0]
                risk_dist[level] = count

            # Average risk score
            avg_risk = conn.execute(
                "SELECT AVG(risk_score) FROM screenings WHERE risk_score IS NOT NULL"
            ).fetchone()[0]

            # DR grade distribution
            dr_dist = {}
            for grade in range(5):
                count = conn.execute(
                    "SELECT COUNT(*) FROM screenings WHERE dr_grade = ?", (grade,)
                ).fetchone()[0]
                dr_dist[f"Grade {grade}"] = count

        return {
            "total_patients": total_patients,
            "total_screenings": total_screenings,
            "total_referrals": total_referrals,
            "pending_referrals": pending_referrals,
            "risk_distribution": risk_dist,
            "average_risk_score": round(avg_risk, 1) if avg_risk else 0,
            "dr_distribution": dr_dist,
        }

    def add_to_sync_queue(self, record_type: str, record_id: str, abha_id: str, fhir_resource: Dict):
        """Add a record to the ABHA sync queue."""
        sync_id = str(uuid.uuid4())[:8]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO sync_queue (id, record_type, record_id, abha_id, fhir_resource)
                   VALUES (?, ?, ?, ?, ?)""",
                (sync_id, record_type, record_id, abha_id, json.dumps(fhir_resource))
            )

    def get_pending_syncs(self) -> List[Dict[str, Any]]:
        """Get all pending sync records."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sync_queue WHERE status = 'pending' ORDER BY ROWID"
            ).fetchall()
            return [dict(row) for row in rows]

    def seed_demo_data(self):
        """Seed database with demo data for presentation."""
        import random
        random.seed(42)

        villages = ["Mandya", "Srirangapatna", "Maddur", "Nagamangala", "Pandavapura"]
        names_f = ["Meena", "Lakshmi", "Savitri", "Radha", "Kamala", "Anita", "Suma", "Geetha"]
        names_m = ["Raju", "Kumar", "Suresh", "Ramesh", "Prakash", "Venkatesh", "Basavaraj"]

        for i in range(25):
            sex = random.choice(["M", "F"])
            name = random.choice(names_m if sex == "M" else names_f)
            age = random.randint(25, 72)

            pid = self.create_patient({
                "name": f"{name} {'ABCDEFGH'[i % 8]}.",
                "age": age,
                "sex": sex,
                "village": random.choice(villages),
                "district": "Mandya",
                "asha_worker_id": f"ASHA-{random.randint(100, 999)}",
            })

            # Random screening
            risk_score = round(random.uniform(1, 9.5), 1)
            risk_level = (
                "LOW" if risk_score <= 3 else
                "MODERATE" if risk_score <= 6 else
                "HIGH" if risk_score <= 8 else "CRITICAL"
            )

            sid = self.save_screening(pid, {
                "screening_type": "combined",
                "risk_score": risk_score,
                "risk_level": risk_level,
                "dr_grade": random.choice([0, 0, 1, 1, 2, 2, 3, 4]),
                "recommendation": f"Risk Level: {risk_level}",
                "referral_needed": risk_score > 6,
            })

            if risk_score > 6:
                self.create_referral(
                    pid, sid,
                    referred_to="Mandya District Hospital",
                    urgency="high" if risk_score > 8 else "moderate",
                    notes=f"Risk score {risk_score}/10"
                )

        print(f"✅ Seeded {25} demo patients with screenings")


if __name__ == "__main__":
    db = DrishtiDB()
    db.seed_demo_data()
    stats = db.get_statistics()
    print(json.dumps(stats, indent=2))
