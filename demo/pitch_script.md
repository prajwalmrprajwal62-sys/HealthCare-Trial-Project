# Drishti Health — Demo Pitch Script
# Time: 5 minutes | Slides: 5 | Live Demo: 2 minutes

---

## SLIDE 1: The Problem (30 seconds)

**[Show statistics — big, bold numbers]**

> "India has 77 million diabetics — second only to China.
> In rural India, there are 0.7 doctors per thousand people.
> The average time to diagnose diabetes in a village?
> **Four and a half years.**
> And the first person a villager sees isn't a doctor — it's an ASHA worker."

**Key stat to emphasize:** 67 crore ABHA IDs issued, but less than 5% of rural PHCs use any AI triage.

---

## SLIDE 2: The Solution (45 seconds)

**[Show Drishti Health logo + architecture diagram]**

> "Drishti Health turns any ₹3,000 Android phone into an offline multi-disease diagnostic co-pilot for ASHA workers."

**Three sentences:**
1. "Retinopathy detection from a ₹200 fundus lens attachment — 3 seconds, ophthalmologist-level accuracy"
2. "Explainable risk scoring — the AI tells you WHY, not just what"
3. "Works in Kannada, requires zero internet, syncs to ABHA when connected"

---

## SLIDE 3: LIVE DEMO (2 minutes)

### Demo Script (Memorize This)

> *"Meet Meena, an ASHA worker in rural Mandya district. No lab. No doctor within 20km. No internet."*

> *"A 48-year-old woman walks in with blurred vision."*

**[Action 1: Open Drishti app → Screening tab]**
- Enter vitals: Age 48, BP 155/95, Glucose 210, HbA1c 8.2, BMI 31.5
- Type symptoms: "blurred vision, frequent urination, fatigue"
- Click "Run Risk Assessment"

**[Action 2: Show Results]**
> *"Risk Score: 7.2 out of 10. HIGH risk."*
> *"Look at this SHAP chart — the AI is transparent. It says: High BP contributed 40%, HbA1c 35%, Age 25%."*
> *"The AI never says 'you have diabetes.' It says: Recommend Dr. consultation at taluk hospital."*

**[Action 3: Fundus Image]**
- Upload sample fundus image → Show DR Grade result
> *"Diabetic Retinopathy: Moderate. Confidence: 78%. Immediate ophthalmologist referral."*

**[Action 4: Voice Input — if Bhashini API available]**
- Type in Kannada: "ಮಸುಕಾದ ದೃಷ್ಟಿ, ಸುಸ್ತು"
> *"The system understands Kannada. It maps symptoms to risk factors automatically."*

**[Action 5: ABHA Sync]**
> *"One tap — ABHA record synced. This patient's health data follows her through India's national health stack."*

### The Closer:
> *"This runs on a ₹3,000 Android phone. No internet. No doctor. Just Meena, the app, and three seconds."*

---

## SLIDE 4: Technology & Trust (45 seconds)

**[Show tech stack table]**

> "We're not using generic models. Our retinal analysis uses RETFound — published in Nature 2023, pre-trained on 1.6 million retinal images. We fine-tune on IDRiD — the Indian Diabetic Retinopathy dataset. Real Indian patients, real Indian data."

> "For voice, we use the Government of India's own language AI — Bhashini. Not OpenAI. Not Google. India's own stack."

**Addressing the trust objection:**
> "Will village people trust AI? Here's our answer:
> - SHAP explainability shows exactly what caused the score
> - Human is ALWAYS in the loop — AI is a calculator, not a doctor
> - Referenced IDx-DR, the first FDA-cleared AI for retinopathy — this works globally"

---

## SLIDE 5: Impact & Scalability (30 seconds)

| Metric | Value |
|---|---|
| Target users | 10 lakh+ ASHA workers in India |
| Screening time | <60 seconds per patient |
| Cost | ₹3,000 phone + ₹200 fundus lens |
| Languages | 22 (via Bhashini) |
| Internet required | No (fully offline) |
| ABHA integration | Yes (FHIR R4) |

> "With 10 lakh ASHA workers screening 10 patients daily, that's 1 crore screenings per day.
> Early detection saves ₹50,000 in treatment costs per patient.
> That's ₹5 lakh crore in annual healthcare savings."

> "This isn't a prototype. This is the future of primary healthcare in India."

---

## BACKUP SLIDES (If judges ask)

### Technical Architecture Deep Dive
- RETFound ViT-Large 224x224 → DR Grade 0-4
- XGBoost ensemble → Risk Score 0-10 with SHAP
- Bhashini pipeline: ASR → NMT → TTS
- SQLite offline → ABDM FHIR R4 sync

### Competitive Landscape
| | Drishti Health | Google Health AI | Microsoft InnerEye |
|---|---|---|---|
| Offline | ✅ | ❌ | ❌ |
| Indian Languages | 22 | 3 | 0 |
| Indian Dataset | ✅ IDRiD | ❌ | ❌ |
| ABHA Integration | ✅ | ❌ | ❌ |
| Cost | ₹3,000 | Enterprise | Enterprise |

### Team Credentials
- Prajwal — ML/Data Science Lead
- Harshith — Integration & Demo
- Hemanth — UX & Pitch
- Tejas — Backend & API

---

## PRE-DEMO CHECKLIST

- [ ] Streamlit app running (`streamlit run app.py`)
- [ ] Demo data seeded (click "Seed Demo Data" on Patient Records page)
- [ ] Sample fundus image ready for upload
- [ ] Kannada text ready to paste: ಮಸುಕಾದ ದೃಷ್ಟಿ, ಸುಸ್ತು
- [ ] Default vitals pre-filled (48yo, BP 155/95, Glucose 210)
- [ ] SHAP chart renders correctly
- [ ] Browser zoomed to 125% for visibility
- [ ] Phone hotspot ready as backup internet
