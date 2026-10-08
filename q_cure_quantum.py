from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load Breast Cancer Wisconsin dataset
data = load_breast_cancer()

# Convert dataset into DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df["target"] = data.target

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nTarget classes:")
print(df["target"].value_counts())

print("\nFeature names:")
print(data.feature_names)# Select 4 features for quantum model
selected_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area"
]

X = df[selected_features]
y = df["target"]

print("\nSelected 4 features:")
print(X.head())

print("\nSelected feature names:")
print(selected_features)

print("\nInput shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features to the range [0, 1]
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nData preprocessing completed!")

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nFirst 5 scaled training samples:")
print(X_train[:5])

print("\nFirst 5 training labels:")
print(y_train.iloc[:5].values)
import pennylane as qml
import numpy as np

# Create a 4-qubit quantum device
dev = qml.device("default.qubit", wires=4)

# Quantum circuit for encoding 4 features
@qml.qnode(dev)
def quantum_circuit(features):
    for i in range(4):
        qml.RY(features[i] * np.pi, wires=i)

    return qml.probs(wires=range(4))

# Test the quantum circuit with the first training sample
sample = X_train[0]

probabilities = quantum_circuit(sample)

print("\nQuantum circuit executed successfully!")

print("\nInput sample:")
print(sample)

print("\nQuantum measurement probabilities:")
print(probabilities)
# ==========================================
# QUANTUM VARIATIONAL CLASSIFIER
# ==========================================
# ==========================================
# QUANTUM VARIATIONAL CLASSIFIER
# ==========================================

from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler

# Number of features = number of qubits
num_features = 4

# Feature map: encodes our 4 cancer features
feature_map = ZZFeatureMap(
    feature_dimension=num_features,
    reps=2,
    entanglement="linear"
)

# Trainable variational circuit
ansatz = TwoLocal(
    num_qubits=num_features,
    rotation_blocks="ry",
    entanglement_blocks="cz",
    reps=2,
    entanglement="full"
)

# Quantum sampler
sampler = StatevectorSampler()

# Classical optimizer
optimizer = COBYLA(maxiter=50)

# Create VQC
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer
)

print("\nVQC model configured successfully!")
print("Features:", num_features)
print("Qubits:", num_features)
print("Feature map: ZZFeatureMap")
print("Ansatz: TwoLocal")
print("Optimizer: COBYLA")
# ==========================================
# TRAIN THE QUANTUM MODEL
# ==========================================

print("\nStarting VQC training...")
print("Training samples:", len(X_train))
print("Please wait...")

# Convert Pandas Series to NumPy array
X_train_np = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train
y_train_np = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train

# Train the quantum model
vqc.fit(X_train, y_train_np)

import numpy as np

np.save("vqc_weights.npy", vqc._fit_result.x)

print("VQC weights saved as vqc_weights.npy")

# Save complete trained VQC model
vqc.to_dill("q_lumina_vqc.dill")

print(" Trained Q-Lumina VQC saved as q_lumina_vqc.dill")



# TEST THE QUANTUM MODEL
# ==========================================

print("\nTesting the quantum model...")

# Convert test data to NumPy arrays
X_test_np = X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test
y_test_np = y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test

# Predict test samples
y_pred = vqc.predict(X_test_np)

print("\nPrediction completed successfully!")

print("Number of test samples:", len(X_test_np))

print("\nActual labels:")
print(y_test_np[:10])

print("\nPredicted labels:")
print(y_pred[:10])
# ==========================================
# MODEL PERFORMANCE
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Calculate accuracy
accuracy = accuracy_score(y_test_np, y_pred)

print("\n========================================")
print("QUANTUM MODEL PERFORMANCE")
print("========================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test_np,
    y_pred,
    target_names=["Malignant", "Benign"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_np, y_pred))
# ==========================================
# VISUALIZE MODEL PERFORMANCE
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

# Create confusion matrix
cm = confusion_matrix(y_test_np, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Malignant", "Benign"],
    yticklabels=["Malignant", "Benign"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Quantum Model - Confusion Matrix")

plt.tight_layout()

# Save the figure
plt.savefig("confusion_matrix.png", dpi=300)

# ==========================================
# SAVE TRAINED MODEL AND SCALER
# ==========================================

import pickle


with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("\nVQC model saved as vqc_model.pkl")
print("Scaler saved as scaler.pkl")

print("\nConfusion matrix saved as confusion_matrix.png")

plt.show()