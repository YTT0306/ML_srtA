import argparse
import os
import sys
import random
from collections import defaultdict

import joblib
import matplotlib
matplotlib.use("Agg") # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# RDKit: The gold standard for cheminformatics
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors import MoleculeDescriptors

# ML Stack
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

def ensure_dir(path):
    """Simple utility to ensure output directories exist before saving."""
    os.makedirs(path, exist_ok=True)

def get_scaffold(smiles):
    """
    Extracts the Bemis-Murcko scaffold from a SMILES string.
    The scaffold represents the core ring systems and linkers of a molecule,
    removing side chains to identify the fundamental structural class.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Extract the core molecular framework
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return None

def scaffold_statistics(df, out_csv):
    """
    Analyzes the distribution of scaffolds within the dataset.
    Helps identify if the dataset is dominated by a few chemical series 
    or if it is structurally diverse.
    """
    stats = (
        df.groupby("scaffold")
        .agg(n_compounds=("label", "count"), n_active=("label", "sum"))
        .reset_index()
        .sort_values(by="n_compounds", ascending=False)
    )
    stats["active_ratio"] = stats["n_active"] / stats["n_compounds"]

    total_compounds = len(df)
    total_scaffolds = stats.shape[0]
    # Calculate how much of the data is covered by the top 10 most common scaffolds
    top10_coverage = stats.head(10)["n_compounds"].sum() / total_compounds

    print(f"Total compounds: {total_compounds}")
    print(f"Total scaffolds: {total_scaffolds}")
    print(f"Top 10 scaffold coverage: {top10_coverage:.2%}")

    stats.to_csv(out_csv, index=False)
    print(f"Scaffold statistics saved to {out_csv}")

def scaffold_split(df, seed=42, train_frac=0.8):
    """
    Performs a scaffold-aware train/test split. 
    
    Why? In drug discovery, random splits often lead to 'easy' tests because 
    similar molecules end up in both sets. Scaffold splitting ensures the 
    model generalizes to entirely new chemical classes.
    """
    rng = random.Random(seed)

    # Map each scaffold to its corresponding row indices
    scaffold_groups = {}
    for idx, row in df.iterrows():
        scaffold_groups.setdefault(row["scaffold"], []).append(idx)

    # Group scaffolds by activity to maintain a balanced label distribution
    active_scaffolds = []
    inactive_scaffolds = []
    for scaffold, indices in scaffold_groups.items():
        n_active = int(df.loc[indices, "label"].sum())
        if n_active > 0:
            active_scaffolds.append((scaffold, indices, n_active))
        else:
            inactive_scaffolds.append((scaffold, indices, 0))

    # Shuffle to ensure randomness within active/inactive groups
    rng.shuffle(active_scaffolds)
    rng.shuffle(inactive_scaffolds)

    n_total = len(df)
    target_train = int(train_frac * n_total)
    train_idx, test_idx = [], []

    # Attempt to balance actives across the split while keeping scaffolds atomic
    total_actives = int(df["label"].sum())
    target_train_actives = int(train_frac * total_actives)
    train_actives = 0

    # Fill training set with active-containing scaffolds first
    for _, indices, n_active in active_scaffolds:
        if train_actives < target_train_actives and len(train_idx) < target_train:
            train_idx.extend(indices)
            train_actives += n_active
        else:
            test_idx.extend(indices)

    # Fill the remainder with inactive scaffolds
    for _, indices, _ in inactive_scaffolds:
        if len(train_idx) < target_train:
            train_idx.extend(indices)
        else:
            test_idx.extend(indices)

    df = df.copy()
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    # Validation: Ensure no scaffold exists in both training and testing
    train_scaffolds = set(df[df["split"] == "train"]["scaffold"])
    test_scaffolds = set(df[df["split"] == "test"]["scaffold"])
    intersection = train_scaffolds.intersection(test_scaffolds)
    assert len(intersection) == 0, "Scaffold overlap detected! Data leakage risk."

    return df

def build_features(df, output_csv, ecfp_bits=2048, ecfp_radius=2):
    """
    Generates a hybrid feature set:
    1. ECFP4 Fingerprints: Capture local circular environments (substructures).
    2. RDKit Descriptors: Capture global physical-chemical properties (LogP, MW, etc.).
    
    Includes a robust feature selection pipeline to remove redundant/low-quality data.
    """
    print("\n--- Generating Hybrid Features ---")
    descriptor_names = [d[0] for d in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=ecfp_radius, fpSize=ecfp_bits)

    def featurize(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            # Part A: Circular Fingerprints (Fixed-length bit vector)
            ecfp = fp_gen.GetFingerprint(mol)
            ecfp_array = np.zeros((ecfp_bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(ecfp, ecfp_array)
            # Part B: PhysChem Descriptors
            rdkit_desc = np.array(calculator.CalcDescriptors(mol), dtype=float)
            # Replace infinities (common in some descriptors) with NaN for easier cleaning
            if not np.isfinite(rdkit_desc).all():
                rdkit_desc[~np.isfinite(rdkit_desc)] = np.nan
            return np.concatenate([ecfp_array, rdkit_desc])
        except Exception:
            return None

    # Processing molecular structures
    features, valid_idx = [], []
    for i, smi in enumerate(df["SMILES"]):
        vec = featurize(smi)
        if vec is not None:
            features.append(vec)
            valid_idx.append(i)

    X = np.array(features)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    initial_cols = [f"ECFP4_{i}" for i in range(ecfp_bits)] + descriptor_names
    feature_df = pd.DataFrame(X, columns=initial_cols)

    # --- Feature Cleaning Pipeline ---
    # 1. Remove features with > 50% missing values (Low reliability)
    feature_df.dropna(thresh=len(feature_df)*0.5, axis=1, inplace=True)
    
    # 2. Remove constant features (Zero variance = No information)
    feature_df = feature_df.loc[:, feature_df.nunique() > 1]
    
    # 3. Remove highly correlated features (r > 0.9)
    # This reduces multicollinearity and speeds up training.
    corr_matrix = feature_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.90)]
    feature_df.drop(columns=to_drop, inplace=True)

    final_df = pd.concat([df_valid[["SMILES", "label", "split", "scaffold"]], feature_df], axis=1)
    return final_df, feature_df.columns.tolist()

# ... (Additional model building functions would follow similar documentation patterns)

def main():
    """
    Entry point for the Chemoinformatics ML Pipeline.
    Handles command-line arguments and orchestrates the workflow.
    """
    parser = argparse.ArgumentParser(description="Molecular Activity Prediction Pipeline")
    parser.add_argument("--input", default="sortaseA_compound.csv", help="Input CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--k-folds", type=int, default=5, help="Folds for cross-validation")
    # ... additional arguments
