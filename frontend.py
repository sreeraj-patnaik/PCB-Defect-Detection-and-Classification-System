import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("🔍 PCB Defect Detection (Template Comparison)")

st.write("Upload a clean template PCB and a defective PCB image.")

template_file = st.file_uploader("Upload Template PCB", type=["png","jpg","jpeg"], key="template")
defect_file = st.file_uploader("Upload Defective PCB", type=["png","jpg","jpeg"], key="defect")

if template_file and defect_file:

    st.image(template_file, caption="Template PCB", use_column_width=True)
    st.image(defect_file, caption="Defective PCB", use_column_width=True)

    if st.button("Detect Defects"):

        files = {
            "template": template_file,
            "defect": defect_file
        }

        response = requests.post(API_URL, files=files)

        if response.status_code == 200:

            result = response.json()

            st.success(f"Detected {result['total_rois_detected']} defects")

            if result["total_rois_detected"] == 0:
                st.info("No defects found.")
            else:
                for i, item in enumerate(result["results"]):

                    st.write(f"### Defect {i+1}")
                    st.write(f"Type: **{item['prediction']}**")
                    st.write(f"Confidence: **{item['confidence']}%**")
                    st.write("---")

        else:
            st.error("Error connecting to backend.")