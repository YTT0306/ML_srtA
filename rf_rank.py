# Virtual Screening: Processes new SMILES and ranks them using the trained RF model.
import os
import csv
import joblib
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors


# Configuration Parameters

ECFP_BITS = 2048
ECFP_RADIUS = 2

INPUT_CSV = "compound.csv"
MODEL_FILE = "results/models/RF_sortaseA_model.pkl"
FEATURES_REF_CSV = "results/features/sortaseA_features_ecfp4_rdkit.csv"

OUT_DIR = "results/features"
PRED_OUT = "RF_scores.csv"
FEATURE_OUT_NPY = os.path.join(OUT_DIR, "RF_inference_features.npy")
SMILES_OUT = os.path.join(OUT_DIR, "RF_inference_smiles.csv")


# Utility Functions

def load_final_feature_columns(path):
    """
    Identifies the exact feature set used during training.
    Filters out metadata columns to ensure the inference input matches the model's expected shape.
    """
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    forbidden = {"SMILES", "label", "split", "scaffold"}
    return [c for c in header if c not in forbidden]


# Initialization

print("[INFO] Loading training feature definition")
final_feature_names = load_final_feature_columns(FEATURES_REF_CSV)
rdkit_desc = [c for c in final_feature_names if not c.startswith("ECFP4_")]

# Initialize RDKit generators based on training parameters
fpgen = rdFingerprintGenerator.GetMorganGenerator(
    radius=ECFP_RADIUS, fpSize=ECFP_BITS
)
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(rdkit_desc)

os.makedirs(OUT_DIR, exist_ok=True)

# Feature Generation

def featurize(smiles):
    """
    Converts a SMILES string into the hybrid feature vector (ECFP4 + RDKit Descriptors).
    Uses reindexing to ensure the feature order exactly matches the training phase.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Part A: Circular Fingerprints
    ecfp = fpgen.GetFingerprint(mol)
    ecfp_arr = np.zeros(ECFP_BITS, dtype=int)
    DataStructs.ConvertToNumpyArray(ecfp, ecfp_arr)

    # Part B: Physical-Chemical Descriptors
    desc = np.array(calculator.CalcDescriptors(mol), dtype=float)
    desc[~np.isfinite(desc)] = np.nan # Handle potential infinity values

    full_cols = (
        [f"ECFP4_{i}" for i in range(ECFP_BITS)]
        + list(calculator.GetDescriptorNames())
    )
    vec = np.concatenate([ecfp_arr, desc])

    # Reindex to match training feature selection (handles dropped/highly correlated features)
    df = pd.DataFrame([vec], columns=full_cols)
    return df.reindex(columns=final_feature_names, fill_value=np.nan).values[0]


# Main Execution Pipeline

print(f"[INFO] Reading input molecules: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

features = []
valid_idx = []

for i, smi in enumerate(df["SMILES"]):
    v = featurize(smi)
    if v is not None:
        features.append(v)
        valid_idx.append(i)
    if (i + 1) % 200 == 0:
        print(f"  processed {i + 1}/{len(df)}", end="\r")

print("\n[INFO] Feature calculation finished")

X = np.array(features)
df_valid = df.iloc[valid_idx].reset_index(drop=True)

# Cache generated features to avoid re-calculation in future steps
np.save(FEATURE_OUT_NPY, X)
df_valid[["SMILES"]].to_csv(SMILES_OUT, index=False)

print(f"[INFO] Feature matrix saved: {X.shape}")


# Prediction & Ranking

print("[INFO] Loading model")
model = joblib.load(MODEL_FILE)

print("[INFO] Predicting probabilities")
# Predict probability of being 'Active' (Class 1)
proba = model.predict_proba(X)[:, 1]
df_valid["Prob_Active"] = proba

# Rank compounds from highest to lowest probability
df_valid = df_valid.sort_values("Prob_Active", ascending=False)
df_valid.to_csv(PRED_OUT, index=False)

print(f"[DONE] Prediction finished → {PRED_OUT}")