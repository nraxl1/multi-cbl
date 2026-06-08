import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from predict import load_model, predict_file
from dataset import parse_spectrum_file

#extra facts about polymers (can be changed to anything)
POLYMER_INFO = {
    "PET": "PET is commonly used in drink bottles and food packaging.",
    "HDPE": "HDPE is used in milk bottles, detergent bottles, and pipes.",
    "PP": "PP is used in food containers, caps, packaging, and textiles.",
    "PS": "PS is used in disposable cups, packaging foam, and trays.",
    "PVC": "PVC is used in pipes, flooring, and cable insulation.",
    "LDPE": "LDPE is used in plastic bags, films, and flexible packaging.",
}

#functional groups differenciating polymers
POLYMER_FUNCTIONAL_GROUPS = {
    "PET": [
        (1450, 1600, "Aromatic Ring"),
        (1000, 1300, "Ester Linkage"),
    ],
    "PS": [
        (1450, 1600, "Aromatic Ring"),
    ],
    "PP": [
        (1370, 1470, "Methyl Bransh"),
    ],
    "HDPE": [
        (2800, 3000, "C-H stretch"),
        (1450, 1475, "CH₂ bending"),
        (700, 730, "CH₂ rocking"),
    ],
    "LDPE": [
        (2800, 3000, "C-H stretch"),
        (1450, 1475, "CH₂ bending"),
        (1370, 1385, "CH₃ branching"),
    ],
    "PVC": [
        (600, 700, "C-Cl chlorine"),
    ],
}

#website setup
st.set_page_config(
    page_title="FTIR Polymer Classifier",
    layout="wide"
)
#title and 'subtitle'
st.title("FTIR Polymer Classifier")
st.write("Upload one CSV file or multiple CSV files to predict the polymer type.")

#directory to the trained model 
model_dir = st.sidebar.text_input("Model folder", "./output")
architecture = "CNN"

#uploading the file
uploaded_files = st.file_uploader(
    "Upload FTIR CSV file(s)",
    type=["csv"],
    #multiple files can be uploaded
    accept_multiple_files=True
)

#adjusting the confidence treshhold (how 'strict do we want to be with the results') - defaoult 60%
unknown_threshold = st.sidebar.slider(
    "Confidence threshold for 'Other'",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.05,
    key="unknown_treshold_slider"
)

#loading the model
@st.cache_resource
def cached_load_model(model_dir, architecture):
    return load_model(model_dir, architecture)

#plotting the spectrum and highlighting the functional groups differenciating different polymers
def plot_spectrum(csv_path, predicted_polymer):
    wn, tr = parse_spectrum_file(csv_path)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wn, tr)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Transmittance")
    ax.set_title(f"Input IR Spectrum and relevant groups for {predicted_polymer}")

    groups = POLYMER_FUNCTIONAL_GROUPS.get(predicted_polymer, [])

    #highlighting relevant group regions
    for start, end, label in groups:
        ax.axvspan(start, end, alpha=0.15)
        ax.text(
            (start + end) / 2,
            max(tr),
            label,
            rotation=90,
            fontsize=8,
            ha="center",
            va="top"
        )
    #reversing the x-axis (IR spectra are shown from high to low wavenumber)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


#stuff that needs to be done after the file is uploaded
if uploaded_files:
    try:
        #load the model
        model, scaler, class_names = cached_load_model(model_dir, architecture)
    except Exception as e:
        #if loading failed throw those exeptions
        st.error(f"Could not load model: {e}")
        st.stop()

    #place for predictions
    results = []

    #go through all of the uploaded files
    for uploaded_file in uploaded_files:

        #temporarly save the files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.divider()
        st.subheader(uploaded_file.name)
        
        #predicting the polymer (predict.py) and saving to result
        result = predict_file(tmp_path, model, scaler, class_names)


       # """
       # example of result
       # {
       #     "prediction": "PET",
       #     "confidence": 0.92,
       #     "all_probabilities": {
       #         "PET": 0.92,
       #         "HDPE": 0.03,
       #         "PP": 0.02
       #     }
       # }
       # """

        if "error" in result:
            st.error(result["error"])
            continue

        #getting the prediction and confidence
        prediction = result["prediction"]
        confidence = result["confidence"]

        #checking if the confidence is high enough for a given treshold
        if confidence < st.session_state.unknown_treshold_slider:
            final_prediction = "Other"
        else:
            final_prediction = prediction

        #col1 - predicted polymer and confidence col2 - IR diagram
        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Predicted polymer", final_prediction)
            st.metric("Confidence", f"{confidence:.2%}")

            if final_prediction != "Other":
                st.info(POLYMER_INFO.get(final_prediction, "No extra information available."))
            else:
                st.warning("The model confidence is low - this sample is categorized as Other.")

            #listing all of the probabilities 
            probs_df = pd.DataFrame(
                list(result["all_probabilities"].items()),
                columns=["Polymer", "Probability"]
            ).sort_values("Probability", ascending=False)

            st.write("Probability breakdown")
            st.dataframe(probs_df, use_container_width=True)

        with col2:
            plot_spectrum(tmp_path, final_prediction)
        #adding the results to the list of all of the other results
        results.append({
            "file": uploaded_file.name,
            "prediction": final_prediction,
            "model_prediction": prediction,
            "confidence": confidence
        })

    #creating a summary if there is more than one cvs file
    if len(results) > 1:
        st.divider()
        st.subheader("Batch summary")

        summary_df = pd.DataFrame(results)
        st.dataframe(summary_df, use_container_width=True)

        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results CSV",
            csv,
            "polymer_predictions.csv",
            "text/csv"
        )

#if nothing is uploaded
else:
    st.info("Upload one or more CSV files to start.")