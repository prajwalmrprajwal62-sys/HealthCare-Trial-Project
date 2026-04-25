"""
Drishti Health — Streamlit Dashboard (V2)

Multi-page diagnostic co-pilot for ASHA workers.
Upgraded with: Camera PPG, Auto-Save, Demo Mode, Sarvam AI,
Referral Reports, Grad-CAM Heatmaps, and Quick Demo Landing.

Launch: streamlit run app.py
"""

import streamlit as st
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="Drishti Health — AI Diagnostic Co-Pilot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom CSS ─────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar Navigation ─────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding: 1rem 0;">
    <h1 style="color: #00BFA6; font-size: 1.8rem; margin-bottom: 0;">🏥 Drishti Health</h1>
    <p style="color: #888; font-size: 0.85rem; margin-top: 0.2rem;">AI Diagnostic Co-Pilot v2.0</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

# Navigation pages
NAV_PAGES = ["🏠 Home", "🔬 Screening", "📊 Results", "📋 Patient Records", "📈 Dashboard"]

# Handle programmatic navigation
default_idx = 0
if "_go_to_page" in st.session_state:
    _target = st.session_state.pop("_go_to_page")
    if _target in NAV_PAGES:
        default_idx = NAV_PAGES.index(_target)

page = st.sidebar.radio(
    "Navigate",
    NAV_PAGES,
    index=default_idx,
    label_visibility="collapsed",
)

st.sidebar.divider()

# Connectivity indicator
st.sidebar.markdown("### ⚙️ System Status")
offline_mode = st.sidebar.toggle("Offline Mode", value=True)
if offline_mode:
    st.sidebar.success("📴 Offline — All data stored locally")
else:
    st.sidebar.info("🌐 Online — ABHA sync available")

# Screening counter
total_screened = st.session_state.get("_total_screened", 0)
st.sidebar.metric("Patients Screened Today", total_screened)

st.sidebar.markdown("""
<div style="position: fixed; bottom: 1rem; padding: 0.5rem; font-size: 0.7rem; color: #666;">
    Built by Team Cognivex<br>
    Vibeathon Mysore 2026
</div>
""", unsafe_allow_html=True)


# ── Page Router ─────────────────────────────────────────

if page == "🏠 Home":
    # ── HOME PAGE — QUICK DEMO LANDING ──────────────────
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="background: linear-gradient(135deg, #00BFA6 0%, #0288D1 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;">
            Drishti Health
        </h1>
        <p style="font-size: 1.2rem; color: #888; max-width: 700px; margin: 0 auto;">
            Turning any Android phone into an offline multi-disease diagnostic co-pilot for ASHA workers
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🇮🇳 Diabetics in India", "77M", "2nd globally")
    with col2:
        st.metric("👨‍⚕️ Doctors/1000", "0.7", "WHO min: 1.0", delta_color="inverse")
    with col3:
        st.metric("⏱️ Rural Diagnosis Delay", "4.5 years", "vs 6 months urban")
    with col4:
        st.metric("🆔 ABHA IDs Issued", "67 Cr", "<5% rural AI use")

    st.markdown("---")

    # Tech stack badges
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0;">
        <span style="display: inline-block; background: #1a1a2e; border: 1px solid #00BFA6; padding: 5px 12px; border-radius: 20px; margin: 3px; color: #00BFA6; font-size: 0.8rem;">🧠 XGBoost + SHAP</span>
        <span style="display: inline-block; background: #1a1a2e; border: 1px solid #FF9800; padding: 5px 12px; border-radius: 20px; margin: 3px; color: #FF9800; font-size: 0.8rem;">🔬 RETFound (Nature 2023)</span>
        <span style="display: inline-block; background: #1a1a2e; border: 1px solid #E040FB; padding: 5px 12px; border-radius: 20px; margin: 3px; color: #E040FB; font-size: 0.8rem;">🗣️ Bhashini API</span>
        <span style="display: inline-block; background: #1a1a2e; border: 1px solid #2196F3; padding: 5px 12px; border-radius: 20px; margin: 3px; color: #2196F3; font-size: 0.8rem;">🤖 Sarvam AI</span>
        <span style="display: inline-block; background: #1a1a2e; border: 1px solid #4CAF50; padding: 5px 12px; border-radius: 20px; margin: 3px; color: #4CAF50; font-size: 0.8rem;">🆔 ABHA/ABDM</span>
        <span style="display: inline-block; background: #1a1a2e; border: 1px solid #F44336; padding: 5px 12px; border-radius: 20px; margin: 3px; color: #F44336; font-size: 0.8rem;">💓 Camera PPG</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Feature cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
                    padding: 1.5rem; border-radius: 12px; border: 1px solid #0f3460;
                    height: 200px;">
            <h3 style="color: #00BFA6;">🔬 Retinopathy Detection</h3>
            <p style="color: #ccc;">RETFound foundation model fine-tuned on IDRiD Indian dataset.
            Ophthalmologist-level DR grading in 3 seconds.</p>
            <p style="color: #00BFA6; font-weight: bold;">Nature 2023 · 1.6M images</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
                    padding: 1.5rem; border-radius: 12px; border: 1px solid #0f3460;
                    height: 200px;">
            <h3 style="color: #FF9800;">📊 Explainable Risk Scoring</h3>
            <p style="color: #ccc;">XGBoost ensemble with SHAP explainability.
            Shows exactly WHY the AI flagged high risk.</p>
            <p style="color: #FF9800; font-weight: bold;">Trust through transparency</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
                    padding: 1.5rem; border-radius: 12px; border: 1px solid #0f3460;
                    height: 200px;">
            <h3 style="color: #E040FB;">🗣️ Vernacular Voice Input</h3>
            <p style="color: #ccc;">Bhashini API for Kannada/Hindi.
            Vosk offline fallback. Full TTS patient summaries.</p>
            <p style="color: #E040FB; font-weight: bold;">Govt of India Language AI</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── ONE-CLICK DEMO MODE ─────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #0d1117, #161b22);
                    border-radius: 16px; border: 2px solid #00BFA6;">
            <h3 style="color: #00BFA6; margin: 0;">🎬 Live Demo Mode</h3>
            <p style="color: #888; font-size: 0.9rem;">Run the complete Meena demo scenario in one click</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("▶️  Run Live Demo — Meet Meena", width="stretch", type="primary"):
            with st.spinner("Running Meena's screening..."):
                from ml.risk_scorer import RiskScorer
                from ml.symptom_classifier import SymptomClassifier

                scorer = RiskScorer()
                classifier = SymptomClassifier()

                # Meena's patient vitals
                demo_vitals = {
                    "age": 48, "sex": 0, "bp_systolic": 155, "bp_diastolic": 95,
                    "glucose": 210, "hba1c": 8.2, "bmi": 31.5, "cholesterol": 245,
                    "heart_rate": 82, "smoking": 0, "family_history_diabetes": 1,
                    "family_history_heart": 0, "physical_activity": 2.0, "pregnancies": 3,
                }

                risk_result = scorer.predict_risk(demo_vitals)
                symptom_result = classifier.classify("blurred vision, frequent urination, fatigue")

                # Auto-save to database
                from backend.database import DrishtiDB
                db = DrishtiDB()
                save_result = db.save_screening_result(
                    vitals=demo_vitals, risk_result=risk_result,
                    symptom_result=symptom_result, patient_name="Meena S. (Demo)"
                )

                # Store in session state
                st.session_state["risk_result"] = risk_result
                st.session_state["symptom_result"] = symptom_result
                st.session_state["vitals"] = demo_vitals
                st.session_state["_total_screened"] = st.session_state.get("_total_screened", 0) + 1
                st.session_state["_last_save"] = save_result

            st.session_state["_go_to_page"] = "📊 Results"
            st.rerun()

        if st.button("🚀 Start New Screening", width="stretch"):
            st.session_state["_go_to_page"] = "🔬 Screening"
            st.rerun()

    # Show last screening summary if available
    if "risk_result" in st.session_state:
        st.markdown("---")
        rr = st.session_state["risk_result"]
        color = {"LOW": "#4CAF50", "MODERATE": "#FF9800", "HIGH": "#F44336", "CRITICAL": "#9C27B0"}.get(rr["risk_level"], "#888")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0d1117, #161b22);
                    padding: 1rem; border-radius: 12px; border: 1px solid {color}; text-align: center;">
            <p style="color: #888; margin: 0;">Last Screening Result</p>
            <h2 style="color: {color}; margin: 0.3rem 0;">{rr['risk_score']}/10 — {rr['risk_level']}</h2>
            <p style="color: #ccc; font-size: 0.85rem; margin: 0;">{rr['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)


elif page == "🔬 Screening":
    # ── SCREENING PAGE ──────────────────────────────────
    st.markdown("""
    <h2 style="color: #00BFA6;">🔬 Patient Screening</h2>
    <p style="color: #888;">Enter patient vitals and upload fundus images for comprehensive risk assessment.</p>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Vitals Input", "📷 Fundus Image", "💓 Heart Rate (Camera)", "🗣️ Voice Input"])

    with tab1:
        st.markdown("### Patient Vitals")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Demographics")
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=48, key="age")
            sex = st.selectbox("Sex", ["Female", "Male"], key="sex")
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=3, key="preg")

            st.markdown("#### Blood Pressure")
            bp_sys = st.slider("Systolic BP (mmHg)", 80, 220, 155, key="bps")
            bp_dia = st.slider("Diastolic BP (mmHg)", 40, 140, 95, key="bpd")

            st.markdown("#### Lifestyle")
            smoking = st.selectbox("Smoking", ["No", "Yes"], key="smoke")
            activity = st.slider("Physical Activity (0-10)", 0.0, 10.0, 2.0, key="activity")

        with col2:
            st.markdown("#### Blood Tests")
            glucose = st.number_input("Blood Glucose (mg/dL)", 30, 500, 210, key="glucose")
            hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 8.2, step=0.1, key="hba1c")
            cholesterol = st.number_input("Cholesterol (mg/dL)", 50, 500, 245, key="chol")

            st.markdown("#### Body Measurements")
            bmi = st.number_input("BMI", 10.0, 60.0, 31.5, step=0.1, key="bmi")
            heart_rate = st.number_input("Heart Rate (bpm)", 30, 200, 82, key="hr")

            st.markdown("#### Family History")
            fam_diabetes = st.selectbox("Family History — Diabetes", ["No", "Yes"], index=1, key="fam_d")
            fam_heart = st.selectbox("Family History — Heart Disease", ["No", "Yes"], key="fam_h")

        st.markdown("---")

        # Patient Name for record saving
        patient_name = st.text_input("Patient Name", value="Walk-in Patient", key="patient_name")

        # Symptom text input
        symptoms_text = st.text_area(
            "Symptoms (English, Kannada, or Hindi)",
            value="blurred vision, frequent urination, fatigue",
            placeholder="ಮಸುಕಾದ ದೃಷ್ಟಿ, ಸುಸ್ತು (or type in English/Hindi)",
            key="symptoms"
        )

        if st.button("🔍 Run Risk Assessment", type="primary", width="stretch"):
            with st.spinner("Analyzing patient data..."):
                from ml.risk_scorer import RiskScorer
                from ml.symptom_classifier import SymptomClassifier
                from backend.database import DrishtiDB

                scorer = RiskScorer()
                classifier = SymptomClassifier()
                db = DrishtiDB()

                vitals = {
                    "age": age,
                    "sex": 1 if sex == "Male" else 0,
                    "bp_systolic": bp_sys,
                    "bp_diastolic": bp_dia,
                    "glucose": glucose,
                    "hba1c": hba1c,
                    "bmi": bmi,
                    "cholesterol": cholesterol,
                    "heart_rate": heart_rate,
                    "smoking": 1 if smoking == "Yes" else 0,
                    "family_history_diabetes": 1 if fam_diabetes == "Yes" else 0,
                    "family_history_heart": 1 if fam_heart == "Yes" else 0,
                    "physical_activity": activity,
                    "pregnancies": pregnancies,
                }

                risk_result = scorer.predict_risk(vitals)
                symptom_result = classifier.classify(symptoms_text)

                # Auto-save to database
                save_result = db.save_screening_result(
                    vitals=vitals,
                    risk_result=risk_result,
                    symptom_result=symptom_result,
                    patient_name=patient_name,
                )

                # Store results in session state for Results page
                st.session_state["risk_result"] = risk_result
                st.session_state["symptom_result"] = symptom_result
                st.session_state["vitals"] = vitals
                st.session_state["_total_screened"] = st.session_state.get("_total_screened", 0) + 1
                st.session_state["_last_save"] = save_result

            st.success(f"✅ Screening saved! Patient ID: {save_result['patient_id']} — Navigate to Results for details.")

            # Quick result preview
            score = risk_result["risk_score"]
            level = risk_result["risk_level"]
            color = {"LOW": "#4CAF50", "MODERATE": "#FF9800", "HIGH": "#F44336", "CRITICAL": "#9C27B0"}.get(level, "#888")

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
                        padding: 2rem; border-radius: 16px; border: 2px solid {color};
                        text-align: center; margin: 1rem 0;">
                <h1 style="color: {color}; font-size: 3rem; margin: 0;">{score}/10</h1>
                <h3 style="color: {color}; margin: 0.5rem 0;">Risk Level: {level}</h3>
                <p style="color: #ccc;">{risk_result['recommendation']}</p>
                <p style="color: #aaa; font-style: italic;">{risk_result['recommendation_kn']}</p>
            </div>
            """, unsafe_allow_html=True)

            if save_result.get("referral_id"):
                st.warning(f"🚨 Referral auto-generated (ID: {save_result['referral_id']}) — Mandya District Hospital")

    with tab2:
        st.markdown("### 📷 Fundus Image Analysis")
        st.info("Upload a retinal fundus photograph for Diabetic Retinopathy screening.")

        upload_method = st.radio("Input Method", ["Upload Image", "Camera Capture"], horizontal=True)

        if upload_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload fundus image",
                type=["jpg", "jpeg", "png", "bmp"],
                key="fundus_upload"
            )
        else:
            uploaded_file = st.camera_input("Capture fundus image", key="fundus_camera")

        if uploaded_file:
            from PIL import Image
            from ml.fundus_detector import FundusDetector

            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption="Input Fundus Image", width=None)

            with col2:
                with st.spinner("Analyzing retinal image..."):
                    detector = FundusDetector()
                    result = detector.analyze(image)
                    heatmap = detector.generate_heatmap(image, grade=result["dr_grade"])

                st.session_state["fundus_result"] = result

                color = result["color"]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
                            padding: 1.5rem; border-radius: 12px; border: 2px solid {color};">
                    <h3 style="color: {color}; margin: 0;">DR Grade: {result['dr_grade']} — {result['dr_label']}</h3>
                    <p style="color: #ccc;"><b>Severity:</b> {result['severity']}</p>
                    <p style="color: #ccc;"><b>Confidence:</b> {result['confidence']*100:.1f}%</p>
                    <p style="color: #ccc;"><b>Model:</b> {result['model_used']}</p>
                    <hr style="border-color: #333;">
                    <p style="color: #fff;"><b>📋 {result['recommendation']}</b></p>
                    <p style="color: #aaa; font-style: italic;">📋 {result['recommendation_kn']}</p>
                </div>
                """, unsafe_allow_html=True)

            # Grad-CAM Heatmap
            st.markdown("### 🔥 Grad-CAM Attention Heatmap")
            st.caption("Red regions = AI attention (pathology detected). Shows WHERE the AI found issues.")
            cols = st.columns(2)
            with cols[0]:
                st.image(image.resize((512, 512)), caption="Original Fundus")
            with cols[1]:
                st.image(heatmap, caption="Grad-CAM Heatmap Overlay")

    with tab3:
        # ── CAMERA PPG TAB ──────────────────────────────
        st.markdown("### 💓 Heart Rate — Camera PPG")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
                    padding: 1.5rem; border-radius: 12px; border: 1px solid #F44336;">
            <h4 style="color: #F44336;">No Hardware Needed!</h4>
            <p style="color: #ccc;">Measures heart rate from the phone camera using photoplethysmography (PPG).
            Place your fingertip on the camera lens under steady lighting.</p>
            <p style="color: #888; font-size: 0.8rem;">
                <b>How it works:</b> Detects blood volume changes through skin color variations in the green channel.
                FFT extracts the dominant frequency → heart rate in BPM.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        ppg_mode = st.radio("PPG Mode", ["📱 Demo (Simulated)", "📷 Camera (Live)"], horizontal=True)

        if ppg_mode == "📱 Demo (Simulated)":
            if st.button("💓 Measure Heart Rate (Demo)", type="primary"):
                from ml.camera_ppg import CameraPPG
                import plotly.graph_objects as go

                ppg = CameraPPG()
                result = ppg.measure_demo()

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div style="text-align: center; background: linear-gradient(135deg, #1a1a2e, #16213e);
                                padding: 2rem; border-radius: 16px; border: 2px solid #F44336;">
                        <h1 style="color: #F44336; font-size: 4rem; margin: 0;">❤️ {result['heart_rate_bpm']}</h1>
                        <h3 style="color: #F44336; margin: 0;">BPM</h3>
                        <p style="color: #ccc;">Signal Quality: {result['signal_quality']} ({result['confidence']}%)</p>
                        <p style="color: #888;">SpO2 Estimate: {result.get('spo2_estimate', 'N/A')}%</p>
                        <p style="color: #666; font-size: 0.8rem;">Method: {result['method']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    waveform = result["waveform"]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=waveform["time"][:300],
                        y=waveform["amplitude"][:300],
                        mode="lines",
                        line=dict(color="#F44336", width=2),
                        name="PPG Signal"
                    ))
                    fig.update_layout(
                        title="PPG Waveform",
                        xaxis_title="Time (s)",
                        yaxis_title="Amplitude",
                        height=300,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#ccc"),
                    )
                    st.plotly_chart(fig, width="stretch")
        else:
            st.info("📷 Point your camera at your fingertip (cover the lens). Results appear after 5 seconds.")
            cam_input = st.camera_input("Place fingertip on camera", key="ppg_camera")
            if cam_input:
                st.info("💓 Processing... (In hackathon, live PPG would process continuous frames)")

    with tab4:
        st.markdown("### 🗣️ Voice Input")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
                    padding: 1.5rem; border-radius: 12px; border: 1px solid #E040FB;">
            <h4 style="color: #E040FB;">Supported Languages</h4>
            <p style="color: #ccc;">🇮🇳 Kannada (ಕನ್ನಡ) · Hindi (हिन्दी) · English</p>
            <p style="color: #888; font-size: 0.85rem;">
                <b>Online:</b> Bhashini API (Govt of India) — 22 languages<br>
                <b>Offline:</b> Vosk STT — Hindi & Kannada models
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        voice_text = st.text_input(
            "Type or speak symptoms",
            placeholder="ಮಸುಕಾದ ದೃಷ್ಟಿ, ಆಗಾಗ್ಗೆ ಮೂತ್ರ ವಿಸರ್ಜನೆ",
            key="voice_text"
        )

        if voice_text:
            from ml.symptom_classifier import SymptomClassifier
            classifier = SymptomClassifier()
            result = classifier.classify(voice_text)

            st.markdown(f"""
            **Language Detected:** {result['language_detected']}
            **English Translation:** {result['english_translation']}
            **Symptoms Found:** {', '.join(result['symptoms_detected'])}
            **Risk Factors:** {', '.join(result['risk_factors'])}
            **Urgency:** {result['overall_urgency']}
            """)


elif page == "📊 Results":
    # ── RESULTS PAGE ────────────────────────────────────
    st.markdown("""
    <h2 style="color: #00BFA6;">📊 Screening Results</h2>
    """, unsafe_allow_html=True)

    risk_result = st.session_state.get("risk_result")
    fundus_result = st.session_state.get("fundus_result")
    symptom_result = st.session_state.get("symptom_result")

    if not risk_result and not fundus_result:
        st.warning("No screening results yet. Go to 🔬 Screening to run an assessment.")
    else:
        # ── Demo narrative ──────────────────────────────
        if risk_result:
            st.markdown("""
            > *"Meet Meena, an ASHA worker in rural Mandya district. No lab. No doctor within 20km.
            > A 48-year-old woman walks in with blurred vision. Three seconds later..."*
            """)

        # ── Risk Score Gauge ────────────────────────────
        if risk_result:
            score = risk_result["risk_score"]
            level = risk_result["risk_level"]
            color = {"LOW": "#4CAF50", "MODERATE": "#FF9800", "HIGH": "#F44336", "CRITICAL": "#9C27B0"}.get(level, "#888")

            st.markdown(f"""
            <div style="text-align: center; background: linear-gradient(135deg, #0d1117, #161b22);
                        padding: 2rem; border-radius: 20px; border: 2px solid {color}; margin-bottom: 1rem;">
                <h1 style="color: {color}; font-size: 4rem; margin: 0; font-weight: 800;">{score}/10</h1>
                <h2 style="color: {color}; margin: 0.5rem 0 0 0; letter-spacing: 3px;">{level} RISK</h2>
                <p style="color: #ccc; margin-top: 1rem; font-size: 1.1rem;">{risk_result['recommendation']}</p>
                <p style="color: #999; font-style: italic;">{risk_result['recommendation_kn']}</p>
            </div>
            """, unsafe_allow_html=True)

            # ── SHAP Waterfall Chart ────────────────────
            st.markdown("### 📊 Why This Score? (SHAP Explainability)")
            st.caption("Each bar shows how much a factor pushed the risk UP or DOWN.")

            import plotly.graph_objects as go

            factors = risk_result["top_factors"]
            if factors:
                features = [f["feature"].replace("_", " ").title() for f in factors]
                values = [f["contribution_pct"] * (1 if f["direction"] == "increases_risk" else -1) for f in factors]
                colors = ["#F44336" if v > 0 else "#4CAF50" for v in values]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=features[::-1],
                    x=values[::-1],
                    orientation="h",
                    marker_color=colors[::-1],
                    text=[f"{abs(v):.1f}%" for v in values[::-1]],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Contribution: %{x:.1f}%<extra></extra>"
                ))
                fig.update_layout(
                    height=max(300, len(factors) * 50),
                    xaxis_title="Contribution to Risk Score (%)",
                    yaxis_title="",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ccc"),
                    margin=dict(l=20, r=20, t=10, b=40),
                    xaxis=dict(gridcolor="#333", zeroline=True, zerolinecolor="#666"),
                    yaxis=dict(gridcolor="#333"),
                )
                st.plotly_chart(fig, width="stretch")

                # Factor details table
                st.markdown("#### Factor Details")
                import pandas as pd
                df = pd.DataFrame(factors)
                df["direction"] = df["direction"].map({
                    "increases_risk": "⬆️ Increases Risk",
                    "decreases_risk": "⬇️ Decreases Risk"
                })
                df.columns = ["Feature", "Value", "Contribution %", "Direction"]
                st.dataframe(df, width="stretch", hide_index=True)

        # ── Fundus Result ───────────────────────────────
        if fundus_result:
            st.markdown("---")
            st.markdown("### 🔬 Fundus Analysis Result")

            color = fundus_result["color"]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0d1117, #161b22);
                        padding: 1.5rem; border-radius: 16px; border: 2px solid {color};">
                <h2 style="color: {color};">DR Grade {fundus_result['dr_grade']}: {fundus_result['dr_label']}</h2>
                <p style="color: #ccc;"><b>Confidence:</b> {fundus_result['confidence']*100:.1f}%</p>
                <p style="color: #ccc;"><b>Urgency:</b> {fundus_result['referral_urgency']}</p>
            </div>
            """, unsafe_allow_html=True)

            if fundus_result.get("probabilities"):
                import plotly.graph_objects as go
                probs = fundus_result["probabilities"]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(probs.keys()),
                    y=list(probs.values()),
                    marker_color=["#4CAF50", "#8BC34A", "#FF9800", "#F44336", "#9C27B0"],
                    text=[f"{v*100:.1f}%" for v in probs.values()],
                    textposition="outside",
                ))
                fig.update_layout(
                    height=300,
                    xaxis_title="DR Grade",
                    yaxis_title="Probability",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ccc"),
                    margin=dict(l=20, r=20, t=10, b=40),
                )
                st.plotly_chart(fig, width="stretch")

        # ── Symptom Result ──────────────────────────────
        if symptom_result:
            st.markdown("---")
            st.markdown("### 🗣️ Symptom Analysis")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Symptoms Detected:** {', '.join(symptom_result['symptoms_detected'])}")
                st.markdown(f"**Risk Factors:** {', '.join(symptom_result['risk_factors'])}")
                st.markdown(f"**Urgency:** {symptom_result['overall_urgency']}")
            with col2:
                st.markdown("**Recommended Examinations:**")
                for exam in symptom_result['recommended_examinations']:
                    st.markdown(f"- {exam}")

        # ── ACTION BUTTONS ──────────────────────────────
        if risk_result:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                # Generate Patient Summary (Sarvam AI)
                if st.button("📝 Generate Patient Summary", width="stretch"):
                    from integrations.sarvam_client import SarvamClient
                    sarvam = SarvamClient()
                    vitals = st.session_state.get("vitals", {})
                    summary = sarvam.generate_patient_summary(
                        risk_result=risk_result, vitals=vitals,
                        symptom_result=symptom_result, fundus_result=fundus_result,
                        language="kn"
                    )
                    st.session_state["patient_summary"] = summary

                if "patient_summary" in st.session_state:
                    summary = st.session_state["patient_summary"]
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a1a2e, #16213e);
                                padding: 1.5rem; border-radius: 12px; border: 1px solid #2196F3;">
                        <h4 style="color: #2196F3;">📝 Patient Summary ({summary['method']})</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.text_area("English", summary.get("summary_en", ""), height=200, disabled=True)
                    st.text_area("ಕನ್ನಡ (Kannada)", summary.get("summary_kn", ""), height=150, disabled=True)

            with col2:
                # Download Referral Report
                if st.button("📄 Download Referral Report", width="stretch"):
                    from ml.report_generator import ReportGenerator
                    gen = ReportGenerator()
                    vitals = st.session_state.get("vitals", {})
                    html_report = gen.generate_referral_report(
                        risk_result=risk_result, vitals=vitals,
                        patient_name=st.session_state.get("patient_name", "Patient"),
                        symptom_result=symptom_result, fundus_result=fundus_result,
                    )
                    st.session_state["referral_report"] = html_report

                if "referral_report" in st.session_state:
                    st.download_button(
                        "⬇️ Download HTML Report",
                        data=st.session_state["referral_report"],
                        file_name=f"drishti_referral_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                        mime="text/html",
                    )
                    st.components.v1.html(st.session_state["referral_report"], height=500, scrolling=True)

            with col3:
                if st.button("🔬 New Screening", width="stretch"):
                    st.session_state["_go_to_page"] = "🔬 Screening"
                    st.rerun()


elif page == "📋 Patient Records":
    # ── PATIENT RECORDS ─────────────────────────────────
    st.markdown("""
    <h2 style="color: #00BFA6;">📋 Patient Records</h2>
    """, unsafe_allow_html=True)

    from backend.database import DrishtiDB
    db = DrishtiDB()

    # Seed demo data button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🌱 Seed Demo Data"):
            db.seed_demo_data()
            st.success("Seeded 25 demo patients!")
            st.rerun()

    patients = db.get_all_patients()

    if not patients:
        st.info("No patient records yet. Click 'Seed Demo Data' to populate sample data.")
    else:
        import pandas as pd
        df = pd.DataFrame(patients)
        display_cols = [c for c in ["id", "name", "age", "sex", "village", "district", "created_at"] if c in df.columns]
        st.dataframe(df[display_cols], width="stretch", hide_index=True)

        # Show screening count
        st.info(f"📊 Total patients: {len(patients)} | Use Dashboard page for analytics")


elif page == "📈 Dashboard":
    # ── ANALYTICS DASHBOARD ─────────────────────────────
    st.markdown("""
    <h2 style="color: #00BFA6;">📈 Population Health Dashboard</h2>
    <p style="color: #888;">Screening statistics and health trends for Mandya District.</p>
    """, unsafe_allow_html=True)

    from backend.database import DrishtiDB
    db = DrishtiDB()
    stats = db.get_statistics()

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", stats["total_patients"])
    with col2:
        st.metric("Total Screenings", stats["total_screenings"])
    with col3:
        st.metric("Avg Risk Score", f"{stats['average_risk_score']}/10")
    with col4:
        st.metric("Pending Referrals", stats["pending_referrals"])

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Risk Distribution")
        import plotly.graph_objects as go

        risk_data = stats["risk_distribution"]
        if any(v > 0 for v in risk_data.values()):
            fig = go.Figure(data=[go.Pie(
                labels=list(risk_data.keys()),
                values=list(risk_data.values()),
                marker=dict(colors=["#4CAF50", "#FF9800", "#F44336", "#9C27B0"]),
                hole=0.4,
                textinfo="label+value",
            )])
            fig.update_layout(
                height=350,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No screening data yet. Run screenings to see distribution.")

    with col2:
        st.markdown("### DR Grade Distribution")
        dr_data = stats["dr_distribution"]
        if any(v > 0 for v in dr_data.values()):
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(dr_data.keys()),
                y=list(dr_data.values()),
                marker_color=["#4CAF50", "#8BC34A", "#FF9800", "#F44336", "#9C27B0"],
            ))
            fig.update_layout(
                height=350,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
                xaxis_title="DR Grade",
                yaxis_title="Count",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No DR screening data yet.")

    # Trust section
    st.markdown("---")
    st.markdown("""
    ### 🤝 Trust & Transparency
    > *"The AI never says 'you have diabetes.' It says: 'Risk score: 7.2/10. Recommend Dr. consultation at taluk hospital.'"*

    **How Drishti Health builds trust:**
    - ✅ **SHAP explainability** — Shows exactly which factors drove the risk score
    - ✅ **Human-in-the-loop** — AI is a calculator, not a doctor. ASHA worker always decides.
    - ✅ **Vernacular output** — Results in Kannada/Hindi, not English jargon
    - ✅ **Camera PPG** — Heart rate from phone camera, zero additional hardware
    - ✅ **Reference standard** — Based on IDx-DR (US FDA-cleared), ada Health (12M+ users globally)
    """)
