# Q-Lumina
## Quantum-Assisted Early Cancer Detection

### Problem Statement
VNQFF-01 – Quantum Assisted Early Cancer Detection

### Team
Q-Lumina

### Project Overview
Q-Lumina is a quantum-assisted machine learning prototype
designed for breast cancer classification.

The project combines classical data preprocessing with
a quantum machine learning model implemented using Qiskit.

### Dataset
We use the Breast Cancer Wisconsin dataset available
through Scikit-learn.

### How It Works

1. Load the breast cancer dataset.
2. Preprocess the data.
3. Select relevant features.
4. Scale the selected features.
5. Encode the features into a quantum circuit.
6. Use a 4-qubit Variational Quantum Circuit (VQC).
7. Train the quantum model.
8. Classify the input as Benign or Malignant.
9. Display the prediction through the application.

### Quantum Computing
Q-Lumina uses IBM Qiskit for the quantum computing
component.

A 4-qubit quantum circuit is used for feature encoding
and Variational Quantum Classification.

### Technologies Used

- Python
- Qiskit
- Qiskit Aer
- Scikit-learn
- NumPy
- Pandas
- Streamlit
- Matplotlib

### Results

The trained quantum model was integrated into the
application and tested for cancer classification.

A confusion matrix is included in this repository
to show the model evaluation results.

### Repository Contents

- `app.py` – Application interface
- `cancer_dataset.py` – Dataset loading and preprocessing
- `q_cure_quantum.py` – Quantum model implementation
- `scaler.pkl` – Trained data scaler
- `vqc_weights.npy` – Trained VQC weights
- `confusion_matrix.png` – Model evaluation result

### Future Scope

- Improve model performance.
- Test with larger datasets.
- Explore IBM Quantum hardware.
- Investigate advanced quantum machine learning models.

### References

- IBM Qiskit
- Scikit-learn
- Breast Cancer Wisconsin Dataset
  
