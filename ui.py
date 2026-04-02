import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from PIL import Image
import io
import os
from utils.ui_helpers import list_local_models, get_local_stats, generate_pdf_report

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Skin Disease AI Assistant", layout="wide")

# Sidebar - System Status & Model Selection
st.sidebar.title("⚙️ System Control")

# 1. Consolidated Health Status
try:
    health_resp = requests.get(f"{API_BASE_URL}/health")
    if health_resp.status_code == 200:
        health = health_resp.json()
        status_color = "🟢" if health["status"] == "healthy" else "🟡"
        st.sidebar.success(f"{status_color} System Status: {health['status'].upper()}")
        
        llm_info = health["services"].get("llm", {})
        if llm_info.get("provider") == "Ollama":
            st.sidebar.info(f"Ollama: {llm_info.get('status')} ({llm_info.get('model')})")
        
        with st.sidebar.expander("🔍 Service Details"):
            st.json(health["services"])
    else:
        st.sidebar.error("🔴 API Offline")
except Exception:
    st.sidebar.error("🔴 Cannot connect to API")

# 2. Model Selection
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Selection")
available_models = list_local_models()

if available_models:
    selected_model = st.sidebar.selectbox(
        "Available Weight Files:",
        options=available_models
    )
    if st.sidebar.button("🔄 Reload Backend Model", width="stretch"):
        with st.spinner(f"Switching to {selected_model}..."):
            try:
                resp = requests.post(f"{API_BASE_URL}/models/select", params={"model_name": selected_model})
                if resp.status_code == 200:
                    st.sidebar.success(f"Backend now using: {selected_model}")
                    st.rerun()
                else:
                    st.sidebar.error("Failed to switch model.")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
else:
    st.sidebar.warning("No model weights found in models/weights/")

# Main UI Tabs
tab1, tab2, tab3 = st.tabs(["🩺 Analysis & Diagnosis", "📂 Patient History", "📊 Disease Insights"])

# --- TAB 1: Analysis ---
with tab1:
    st.title("Skin Disease AI Analysis")
    st.write("Enter patient details and upload an image for AI analysis.")
    
    col_input, col_img = st.columns([1, 1])
    
    with col_input:
        st.subheader("👤 Patient Information")
        # Use a form to prevent rerunning on every keystroke
        with st.form("patient_analysis_form"):
            user_id = st.text_input("Unique Patient ID", placeholder="e.g. PATIENT-123")
            patient_name = st.text_input("Patient Name", placeholder="e.g. John Doe")
            age = st.number_input("Patient Age", min_value=0, max_value=120, value=25)
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            submit_button = st.form_submit_button("🚀 Start Analysis", width="stretch")

    with col_img:
        if submit_button:
            if uploaded_file and user_id and patient_name:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width="stretch")
                
                with st.spinner("Analyzing..."):
                    try:
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='JPEG')
                        
                        data = {
                            "user_id": user_id, 
                            "patient_name": patient_name,
                            "age": str(age)
                        }
                        files = {"file": ("image.jpg", img_byte_arr.getvalue(), "image/jpeg")}
                        
                        # Call the unified single API (POST) with streaming enabled
                        response = requests.post(f"{API_BASE_URL}/analyze_skin", data=data, files=files, stream=True)
                        
                        if response.status_code == 200:
                            # 1. Capture the Metadata chunk
                            metadata_json = ""
                            remaining_chunk = ""
                            stream_content = response.iter_content(chunk_size=None, decode_unicode=True)
                            
                            # Read the first chunk which contains our metadata
                            for chunk in stream_content:
                                if "||METADATA_END||" in chunk:
                                    parts = chunk.split("||METADATA_END||")
                                    metadata_json = parts[0]
                                    remaining_chunk = parts[1] if len(parts) > 1 else ""
                                    break
                            
                            import json
                            result = json.loads(metadata_json)
                            st.subheader(f"Patient: **{result.get('patient_name', 'N/A')}** ({result['user_id']})")
                            st.subheader(f"Prediction: **{result['prediction']}**")
                            st.progress(result["accuracy"], text=f"Confidence: {result['accuracy']:.2%}")
                            st.markdown("---")
                            
                            st.subheader("💡 AI Recommendation")
                            
                            # 2. Consume the remaining chunks as LLM tokens
                            def stream_generator(initial_token):
                                if initial_token: yield initial_token
                                for token in stream_content:
                                    if token: yield token

                            full_recommendation = st.write_stream(stream_generator(remaining_chunk))
                            
                            def get_pdf():
                                return bytes(generate_pdf_report(
                                    img_byte_arr.getvalue(),
                                    result["prediction"],
                                    result["accuracy"],
                                    full_recommendation
                                ))

                            st.download_button(
                                label="📄 Download PDF Report",
                                data=get_pdf(),
                                file_name=f"Report_{result.get('patient_name', 'Patient')}_{result['user_id']}_{result['prediction']}.pdf",
                                mime="application/pdf",
                                width="stretch"
                            )
                        else:
                            st.error(f"Analysis failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error connecting to backend: {str(e)}")
        elif (not user_id or not patient_name) and uploaded_file:
            st.warning("Please enter both Patient ID and Patient Name to proceed.")
        else:
            st.info("Fill in patient details and upload an image to begin.")

# --- TAB 2: Patient History ---
with tab2:
    st.title("📂 Patient Scan History")
    st.write("Retrieve and review all previous scans for a specific Patient ID.")
    
    search_id = st.text_input("Enter Patient ID to Search", placeholder="e.g. PATIENT-123", key="history_search_id")
    
    if search_id:
        try:
            with st.spinner(f"Fetching history for {search_id}..."):
                hist_resp = requests.get(f"{API_BASE_URL}/history/{search_id}")
                
            if hist_resp.status_code == 200:
                history = hist_resp.json()
                if history:
                    st.success(f"Found {len(history)} records for **{search_id}**")
                    st.markdown("---")
                    
                    for record in history:
                        date_str = record['created_at'][:10]
                        time_str = record['created_at'][11:16]
                        
                        with st.expander(f"📅 {date_str} at {time_str} - {record['prediction']}"):
                            c1, c2 = st.columns([1, 2])
                            
                            with c1:
                                if os.path.exists(record['image_path']):
                                    st.image(record['image_path'], caption="Scan Image", width="stretch")
                                else:
                                    st.warning("Original image file not found on server.")
                            
                            with c2:
                                st.write(f"**Patient Name:** {record.get('patient_name', 'N/A')}")
                                st.write(f"**Age at Scan:** {record['age']}")
                                st.write(f"**AI Confidence:** {record['accuracy']:.2%}")
                                st.write(f"**LLM Provider:** {record['llm_provider']}")
                                st.markdown("### Recommendation")
                                st.markdown(record['llm_recommendation'])
                                
                                # PDF Generation for History
                                if st.button(f"Generate PDF Report (Scan #{record['id']})", key=f"hist_pdf_{record['id']}"):
                                    if os.path.exists(record['image_path']):
                                        with open(record['image_path'], "rb") as f:
                                            img_data = f.read()
                                        
                                        pdf_bytes = generate_pdf_report(
                                            img_data, 
                                            record['prediction'], 
                                            record['accuracy'], 
                                            record['llm_recommendation']
                                        )
                                        
                                        st.download_button(
                                            label="📥 Click to Download PDF",
                                            data=bytes(pdf_bytes),
                                            file_name=f"History_{record.get('patient_name', 'N/A')}_{search_id}_{date_str}.pdf",
                                            mime="application/pdf",
                                            key=f"dl_link_{record['id']}"
                                        )
                                    else:
                                        st.error("Cannot generate PDF: Original image is missing.")
                else:
                    st.info(f"No previous analysis records found for Patient ID: **{search_id}**")
            else:
                st.error(f"Error fetching history: {hist_resp.text}")
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
    else:
        st.info("Enter a Patient ID above to view their medical scan history.")

# --- TAB 3: Disease Insights ---
with tab3:
    st.title("📊 Disease Occurrence Insights")
    st.write("Aggregated statistics of all conditions detected across the entire system.")
    
    # Using the local helper function to get stats directly from DB
    stats_data = get_local_stats()
    
    if stats_data:
        df = pd.DataFrame(list(stats_data.items()), columns=["Disease", "Count"])
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Condition Distribution")
            fig_pie = px.pie(df, values='Count', names='Disease', hole=0.4,
                           color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig_pie, width="stretch")
            
        with col2:
            st.subheader("Detection Frequency")
            fig_bar = px.bar(df, x='Disease', y='Count', color='Disease',
                           text_auto=True, color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig_bar, width="stretch")
            
        st.markdown("---")
        st.subheader("System-wide Summary Table")
        st.dataframe(df, width="stretch")
    else:
        st.info("The system hasn't recorded any scan data yet. Complete an analysis to see insights!")

# Footer
st.markdown("---")
# Safe footer display
llm_name = "Ollama/Groq"
if 'health' in locals() and isinstance(health, dict):
    llm_name = health.get("services", {}).get("llm", {}).get("provider", llm_name)

st.caption(f"Skin Disease AI Assistant v1.0 | Backend: FastAPI | LLM: {llm_name}")
