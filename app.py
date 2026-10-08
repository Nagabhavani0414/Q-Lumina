import streamlit as st
import pickle
import numpy as np

from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Q-Lumina",
    page_icon="⚛️",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title("⚛️ Q-Lumina")
st.subheader("Quantum-Assisted Breast Cancer Detection")

st.write(
    "A research prototype using Quantum Machine Learning "
    "for breast cancer classification."
)

st.divider()


# ==========================================
# LOAD SCALER
# ==========================================

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# ==========================================
# RECREATE QUANTUM MODEL
# ==========================================

num_features = 4
from qiskit_machine_learning.algorithms import VQC

vqc = VQC.from_dill("q_lumina_vqc.dill")

# ==========================================
# PATIENT INPUT
# ==========================================

st.markdown("### 🧬 Enter Patient Features")

mean_radius = st.number_input(
    "Mean Radius",
    min_value=0.0,
    value=14.0
)

mean_texture = st.number_input(
    "Mean Texture",
    min_value=0.0,
    value=20.0
)

mean_perimeter = st.number_input(
    "Mean Perimeter",
    min_value=0.0,
    value=90.0
)

mean_area = st.number_input(
    "Mean Area",
    min_value=0.0,
    value=600.0
)


st.divider()


# ==========================================
# QUANTUM PREDICTION
# ==========================================

if st.button("⚛️ Run Quantum Prediction"):

    input_data = np.array([[
        mean_radius,
        mean_texture,
        mean_perimeter,
        mean_area
    ]])

    # Apply same scaling used during training
    input_scaled = scaler.transform(input_data)

    # Quantum prediction
    prediction = vqc.predict(input_scaled)

    result = int(prediction)

    st.divider()

    if result == 0:
        st.error("⚠️ Prediction: Malignant")
        st.warning(
            "The quantum model classified this sample as malignant."
        )

    else:
        st.success("✅ Prediction: Benign")
        st.info(
            "The quantum model classified this sample as benign."
        )

    st.caption(
        "Research prototype only — not a medical diagnosis."
    )