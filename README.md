# 🏥 Drishti Health

> **Turning any Android phone into an offline multi-disease diagnostic co-pilot for ASHA workers**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)

## The Problem

- **77M diabetics** in India (2nd globally), **138M hypertensives**
- **0.7 doctors** per 1,000 in rural India (WHO minimum: 1:1,000)
- Time to diagnose diabetes in rural areas: **4.5 years** vs 6 months urban
- First point of contact is **ASHA workers**, not doctors

## The Solution

Drishti Health provides:
- 🔬 **Retinopathy detection** in 3 seconds (RETFound + IDRiD dataset)
- 🗣️ **Vernacular voice input** (Kannada/Hindi via Bhashini + Vosk offline)
- 📊 **Explainable risk scoring** (XGBoost + SHAP)
- 🔗 **ABHA-linked** health records (ABDM FHIR R4)
- 📴 **Fully offline** capability (SQLite + TFLite)

## Architecture

```
ANDROID PHONE (or Raspberry Pi 4)
┌────────────────────────────────────────────────────────┐
│  Camera Feed      Voice Input (Kannada)   Manual Vitals │
│  (OpenCV)         (Bhashini STT API)      (BP, Glucose) │
│      ↓                   ↓                     ↓        │
│  YOLOv8-n         Sarvam / DistilBERT     XGBoost Risk  │
│  + RETFound       Symptom Classifier      Ensemble Model │
│  (Fundus DR)      [Offline-capable]       (UCI + IDRiD)  │
│      ↓                   ↓                     ↓        │
│         Unified Risk Score Engine (SHAP explainability)  │
│                          ↓                              │
│  Local SQLite (offline) → ABHA API Sync (when online)   │
│  Bhashini TTS → Patient summary in Kannada/Hindi        │
└────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd drishti-health
pip install -r requirements.txt

# 2. Copy environment variables
cp .env.example .env

# 3. Train risk model (optional — pre-trained model included)
python ml/train_risk_model.py

# 4. Launch dashboard
streamlit run app.py

# 5. Launch API server (separate terminal)
uvicorn backend.main:app --reload --port 8000
```

## Team Cognivex

| Member | Role | Focus |
|---|---|---|
| Prajwal | Captain + Data/ML | Model pipeline, datasets, metrics |
| Harshith | Integration + Demo | Real-time feed, demo script, hardware |
| Hemanth | Pitch + UX | Trust narrative, impact numbers |
| Tejas | Backend + API | FastAPI, Streamlit UI |

## Tech Stack

| Component | Technology |
|---|---|
| Fundus DR Detection | RETFound (Nature 2023) fine-tuned on IDRiD |
| Risk Scoring | XGBoost + SHAP explainability |
| Voice Input | Bhashini API (online) + Vosk (offline) |
| Indian LLM | Sarvam-1 (Bengaluru) |
| Dashboard | Streamlit |
| API | FastAPI |
| Database | SQLite (offline) → ABHA sync |
| Medical Framework | MONAI (PyTorch) |

## Key References

- Zhou et al., *Nature 2023* — RETFound: pre-trained on 1.6M retinal images
- IDRiD Dataset — Indian Diabetic Retinopathy Image Dataset
- Bhashini — Government of India's Language AI platform
- Sarvam AI — India's first open-source Indic LLM

---

*Built with ❤️ for Vibeathon Mysore by Team Cognivex*
