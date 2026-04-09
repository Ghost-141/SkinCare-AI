import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from PIL import Image
import io
import os
import json
from datetime import datetime, timezone
from utils.ui_helpers import generate_pdf_report, clean_llm_markdown

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Skin Care AI Assistant", layout="wide")

# Sidebar - System Status & Model Selection
st.sidebar.title("⚙️ System Control")


# --- Cached API Calls ---
@st.cache_data(ttl=2)  # Short TTL for health check to recover quickly from offline state
def fetch_health_status():
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=5)  # Shorter TTL for models list
def fetch_available_models():
    try:
        resp = requests.get(f"{API_BASE_URL}/models", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# --- Sidebar Fragment ---
@st.fragment
def render_sidebar():
    st.title("⚙️ System Control")

    # 1. Health Status (Using Cache)
    health = fetch_health_status()
    if health:
        status_color = "🟢" if health["status"] == "healthy" else "🟡"
        st.success(f"{status_color} System Status: {health['status'].upper()}")

        llm_info = health["services"].get("llm", {})
        if llm_info:
            status_text = llm_info.get("status", "unknown")
            provider = llm_info.get("provider", "LLM")

            # Show model name if success, otherwise show error
            if any(
                err in status_text.lower()
                for err in ["error", "unreachable", "missing_key"]
            ):
                st.error(f"{provider}: {status_text}")
            else:
                st.info(f"{provider}: {status_text}")

        with st.expander("Service Details"):
            st.json(health["services"])
    else:
        st.error("API Offline")

    # 2. Model Selection (Using Cache)
    st.markdown("---")
    st.subheader("Model Selection")

    model_data = fetch_available_models()
    available_models = []
    active_model_name = ""

    if model_data:
        available_models = model_data.get("available_models", [])
        active_model_path = model_data.get("active_model", "")
        active_model_name = os.path.basename(active_model_path)

    if available_models:
        default_index = 0
        if active_model_name in available_models:
            default_index = available_models.index(active_model_name)

        selected_model = st.selectbox(
            "Available Weight Files:", options=available_models, index=default_index
        )

        if st.button("Reload Backend Model", width="stretch"):
            with st.spinner(f"Switching..."):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/models/select",
                        params={"model_name": selected_model},
                    )
                    if resp.status_code == 200:
                        fetch_available_models.clear()
                        fetch_health_status.clear()
                        st.success(f"Model Updated!")
                    else:
                        st.error("Failed to switch model.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.warning("No model weights found in models/weights/")


# Render the sidebar isolated from main app reruns
with st.sidebar:
    render_sidebar()


# --- Main Content Fragment ---
@st.fragment
def render_main_content():
    # Main UI Tabs
    tab1, tab2, tab3 = st.tabs(
        ["🩺 Analysis & Diagnosis", "📂 Patient History", "📊 Disease Insights"]
    )

    # --- TAB 1: Analysis ---
    with tab1:
        st.title("Skin Care AI Analysis")
        st.write("Enter patient details and upload an image for AI analysis.")

        col_input, col_img = st.columns([1, 1])

        with col_input:
            st.subheader("👤 Patient Information")
            # Form for taking patient info input
            with st.form("patient_analysis_form"):
                user_id = st.text_input(
                    "Unique Patient ID (Numeric)", placeholder="e.g. 1001"
                )
                patient_name = st.text_input(
                    "Patient Name", placeholder="e.g. John Doe"
                )
                age = st.number_input(
                    "Patient Age", min_value=0, max_value=120, value=25
                )
                uploaded_file = st.file_uploader(
                    "Choose an image...", type=["jpg", "jpeg", "png"]
                )
                submit_button = st.form_submit_button("Start Analysis", width="stretch")

        with col_img:
            if submit_button:
                if uploaded_file and user_id and patient_name:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", width=300)

                    with st.spinner("Analyzing..."):
                        try:
                            img_byte_arr = io.BytesIO()
                            image.save(img_byte_arr, format="JPEG")

                            data = {
                                "user_id": str(user_id),
                                "patient_name": patient_name,
                                "age": str(age),
                            }
                            files = {
                                "file": (
                                    "image.jpg",
                                    img_byte_arr.getvalue(),
                                    "image/jpeg",
                                )
                            }

                            # Call the unified single API (POST) with streaming enabled
                            response = requests.post(
                                f"{API_BASE_URL}/analyze_skin",
                                data=data,
                                files=files,
                                stream=True,
                            )

                            if response.status_code == 200:
                                # 1. Capture the Metadata chunk with a buffer to handle split chunks
                                metadata_json = ""
                                remaining_chunk = ""
                                buffer = ""
                                delimiter = "||METADATA_END||"
                                
                                # Use chunk_size=None for the entire stream for maximum efficiency
                                stream_content = response.iter_content(
                                    chunk_size=1, decode_unicode=True
                                )

                                for chunk in stream_content:
                                    buffer += chunk
                                    if delimiter in buffer:
                                        parts = buffer.split(delimiter)
                                        metadata_json = parts[0]
                                        remaining_chunk = parts[1] if len(parts) > 1 else ""
                                        break

                                if not metadata_json:
                                    st.error("Failed to parse metadata from stream.")
                                    return

                                result = json.loads(metadata_json)
                                st.subheader(
                                    f"Patient: **{result.get('patient_name', 'N/A')}** ({result['user_id']})"
                                )
                                st.subheader(f"Prediction: **{result['prediction']}**")
                                st.progress(
                                    result["accuracy"],
                                    text=f"Confidence: {result['accuracy']:.2%}",
                                )

                                # Display side-by-side comparison
                                res_col1, res_col2 = st.columns(2)
                                with res_col1:
                                    st.image(
                                        uploaded_file,
                                        caption="Original Image",
                                        use_container_width=True,
                                    )
                                with res_col2:
                                    if "heatmap_path" in result and os.path.exists(
                                        result["heatmap_path"]
                                    ):
                                        st.image(
                                            result["heatmap_path"],
                                            caption="(Grad-CAM Heatmap)",
                                            use_container_width=True,
                                        )
                                    else:
                                        st.warning("Heatmap not available")

                                st.markdown("---")

                                st.subheader("💡 AI Recommendation")

                                # 2. Consume the remaining chunks as LLM tokens
                                def stream_generator(initial_token):
                                    if initial_token:
                                        # Only clean the very first chunk if it's messy
                                        yield clean_llm_markdown(initial_token)
                                    # Continue yielding raw chunks to preserve natural spacing
                                    for chunk in stream_content:
                                        if chunk:
                                            yield chunk

                                full_recommendation = st.write_stream(
                                    stream_generator(remaining_chunk)
                                )

                                def get_pdf():
                                    h_bytes = None
                                    if "heatmap_path" in result and os.path.exists(
                                        result["heatmap_path"]
                                    ):
                                        with open(result["heatmap_path"], "rb") as f:
                                            h_bytes = f.read()

                                    return bytes(
                                        generate_pdf_report(
                                            img_byte_arr.getvalue(),
                                            result["prediction"],
                                            result["accuracy"],
                                            full_recommendation,
                                            heatmap_bytes=h_bytes,
                                        )
                                    )

                                st.download_button(
                                    label="📄 Download PDF Report",
                                    data=get_pdf(),
                                    file_name=f"Report_{result.get('patient_name', 'Patient')}_{result['user_id']}_{result['prediction']}.pdf",
                                    mime="application/pdf",
                                    width="stretch",
                                )
                            else:
                                st.error(f"Analysis failed: {response.text}")
                        except Exception as e:
                            st.error(f"Error connecting to backend: {str(e)}")
                elif (not user_id or not patient_name) and uploaded_file:
                    st.warning(
                        "Please enter both Patient ID and Patient Name to proceed."
                    )
                else:
                    st.info("Fill in patient details and upload an image to begin.")

    # --- TAB 2: Patient History ---
    with tab2:
        st.title("📂 Patient Scan History")
        st.write("Retrieve and review all previous scans for a specific Patient ID.")

        search_id = st.number_input(
            "Enter Patient ID to Search",
            min_value=1,
            step=1,
            value=1001,
            key="history_search_id",
        )

        if search_id:
            try:
                with st.spinner(f"Fetching history for {search_id}..."):
                    hist_resp = requests.get(f"{API_BASE_URL}/history/{str(search_id)}")

                if hist_resp.status_code == 200:
                    history = hist_resp.json()
                    if history:
                        st.success(f"Found {len(history)} records for **{search_id}**")
                        st.markdown("---")

                        for record in history:
                            # Convert ISO format to datetime object
                            raw_date = record["created_at"]
                            if isinstance(raw_date, str):
                                if raw_date.endswith("Z"):
                                    raw_date = raw_date.replace("Z", "+00:00")
                                dt_obj = datetime.fromisoformat(raw_date)
                            else:
                                dt_obj = raw_date

                            # If naive, assume UTC (since it's stored in UTC)
                            if dt_obj.tzinfo is None:
                                dt_obj = dt_obj.replace(tzinfo=timezone.utc)

                            # Convert to local time
                            local_dt = dt_obj.astimezone()

                            date_str = local_dt.strftime("%Y-%m-%d")
                            time_str = local_dt.strftime("%I:%M %p")

                            with st.expander(
                                f"📅 {date_str} at {time_str} - {record['prediction']}"
                            ):
                                c1, c2 = st.columns([1, 2])

                                with c1:
                                    if os.path.exists(record["image_path"]):
                                        st.image(
                                            record["image_path"],
                                            caption="Scan Image",
                                            use_container_width=True,
                                        )
                                        # Try to show heatmap if it exists
                                        heatmap_path = os.path.join(
                                            os.path.dirname(record["image_path"]),
                                            f"heatmap_{os.path.basename(record['image_path'])}",
                                        )
                                        if os.path.exists(heatmap_path):
                                            st.image(
                                                heatmap_path,
                                                caption="Analysis Heatmap",
                                                use_container_width=True,
                                            )
                                    else:
                                        st.warning(
                                            "Original image file not found on server."
                                        )

                                with c2:
                                    st.write(
                                        f"**Patient Name:** {record.get('patient_name', 'N/A')}"
                                    )
                                    st.write(f"**Age at Scan:** {record['age']}")
                                    st.write(
                                        f"**Confidence:** {record['accuracy']:.2%}"
                                    )
                                    st.write(
                                        f"**LLM Provider:** {record['llm_provider']}"
                                    )
                                    st.markdown("### Recommendation")
                                    st.markdown(record["llm_recommendation"])

                                    # PDF Generation for History
                                    if st.button(
                                        f"Generate PDF Report (Scan #{record['id']})",
                                        key=f"hist_pdf_{record['id']}",
                                    ):
                                        if os.path.exists(record["image_path"]):
                                            with open(record["image_path"], "rb") as f:
                                                img_data = f.read()

                                            # load heatmap if it exists
                                            h_data = None
                                            h_path = os.path.join(
                                                os.path.dirname(record["image_path"]),
                                                f"heatmap_{os.path.basename(record['image_path'])}",
                                            )
                                            if os.path.exists(h_path):
                                                with open(h_path, "rb") as f:
                                                    h_data = f.read()

                                            pdf_bytes = generate_pdf_report(
                                                img_data,
                                                record["prediction"],
                                                record["accuracy"],
                                                record["llm_recommendation"],
                                                heatmap_bytes=h_data,
                                            )

                                            st.download_button(
                                                label="📥 Click to Download PDF",
                                                data=bytes(pdf_bytes),
                                                file_name=f"History_{record.get('patient_name', 'N/A')}_{search_id}_{date_str}.pdf",
                                                mime="application/pdf",
                                                key=f"dl_link_{record['id']}",
                                            )
                                        else:
                                            st.error(
                                                "Cannot generate PDF: Original image is missing."
                                            )
                    else:
                        st.info(
                            f"No previous analysis records found for Patient ID: **{search_id}**"
                        )
                else:
                    st.error(f"Error fetching history: {hist_resp.text}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")
        else:
            st.info("Enter a Patient ID above to view their medical scan history.")

    # --- TAB 3: Disease Insights ---
    with tab3:
        st.title("📊 Disease Occurrence Insights")
        st.write(
            "Aggregated statistics of all conditions detected across the entire system."
        )

        # Fetch stats from API
        try:
            stats_resp = requests.get(f"{API_BASE_URL}/stats")
            if stats_resp.status_code == 200:
                stats_data = stats_resp.json()
            else:
                stats_data = {}
                st.error("Failed to fetch statistics from API.")
        except Exception as e:
            stats_data = {}
            st.error(f"Error connecting to API: {str(e)}")

        if stats_data:
            df = pd.DataFrame(list(stats_data.items()), columns=["Disease", "Count"])

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Condition Distribution")
                fig_pie = px.pie(
                    df,
                    values="Count",
                    names="Disease",
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Plasma,
                )
                st.plotly_chart(fig_pie, width="stretch")

            with col2:
                st.subheader("Detection Frequency")
                fig_bar = px.bar(
                    df,
                    x="Disease",
                    y="Count",
                    color="Disease",
                    text_auto=True,
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                st.plotly_chart(fig_bar, width="stretch")

            st.markdown("---")
            st.subheader("System-wide Summary Table")
            st.dataframe(df, width="stretch")
        else:
            st.info(
                "The system hasn't recorded any scan data yet. Complete an analysis to see insights!"
            )


# Execute main content fragment
render_main_content()

# Footer
st.markdown("---")
# Safe footer display
llm_name = "Ollama/Groq"
# Fetch health status for footer if not cached, or use a default
health = fetch_health_status()
if health:
    llm_name = health.get("services", {}).get("llm", {}).get("provider", llm_name)

st.caption(f"Skin Care AI Assistant v1.0 | Backend: FastAPI | LLM: {llm_name}")
