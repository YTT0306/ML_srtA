import argparse
import os
import sys
import random
from collections import defaultdict

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, 
    precision_recall_curve, 
    roc_auc_score, 
    roc_curve,
    matthews_corrcoef,
    balanced_accuracy_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def ensure_dir(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def get_scaffold(smiles):
    """
    Extract Bemis-Murcko scaffold.
    CRITICAL FIX: Remove stereochemistry before scaffold generation.
    This prevents stereoisomers from being assigned different scaffolds
    and leaking across train/test splits (causing Tanimoto = 1.0).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Remove stereochemistry to group stereoisomers into the same scaffold
        Chem.RemoveStereochemistry(mol)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return None


def scaffold_statistics(df, out_csv):
    """Calculate and save statistics about the scaffolds."""
    stats = (
        df.groupby("scaffold")
        .agg(n_compounds=("label", "count"), n_active=("label", "sum"))
        .reset_index()
        .sort_values(by="n_compounds", ascending=False)
    )
    stats["active_ratio"] = stats["n_active"] / stats["n_compounds"]

    total_compounds = len(df)
    total_scaffolds = stats.shape[0]
    top10_coverage = stats.head(10)["n_compounds"].sum() / total_compounds

    print(f"Total compounds: {total_compounds}")
    print(f"Total scaffolds: {total_scaffolds}")
    print(f"Top 10 scaffold coverage: {top10_coverage:.2%}")

    stats.to_csv(out_csv, index=False)
    return stats


def scaffold_split(df, seed=42, train_frac=0.8):
    """Split dataset based on scaffolds to prevent analogue leakage."""
    rng = random.Random(seed)

    scaffold_groups = defaultdict(list)
    for idx, row in df.iterrows():
        scaffold_groups[row["scaffold"]].append(idx)

    active_scaffolds, inactive_scaffolds = [], []
    for scaffold, indices in scaffold_groups.items():
        n_active = int(df.loc[indices, "label"].sum())
        if n_active > 0:
            active_scaffolds.append((scaffold, indices, n_active))
        else:
            inactive_scaffolds.append((scaffold, indices, 0))

    rng.shuffle(active_scaffolds)
    rng.shuffle(inactive_scaffolds)

    target_train = int(train_frac * len(df))
    target_train_actives = int(train_frac * df["label"].sum())
    
    train_idx, test_idx = [], []
    train_actives = 0

    for _, indices, n_active in active_scaffolds:
        if train_actives < target_train_actives and len(train_idx) < target_train:
            train_idx.extend(indices)
            train_actives += n_active
        else:
            test_idx.extend(indices)

    for _, indices, _ in inactive_scaffolds:
        if len(train_idx) < target_train:
            train_idx.extend(indices)
        else:
            test_idx.extend(indices)

    df = df.copy()
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    # Sanity checks
    train_scaffolds = set(df[df["split"] == "train"]["scaffold"])
    test_scaffolds = set(df[df["split"] == "test"]["scaffold"])
    assert len(train_scaffolds.intersection(test_scaffolds)) == 0, "Scaffold overlap detected!"

    print(f"Train split: {len(train_idx)} samples, {train_actives} actives")
    print(f"Test split: {len(test_idx)} samples, {df.loc[test_idx, 'label'].sum()} actives")
    return df


def enforce_similarity_threshold(df, out_csv, out_plot, threshold=0.8):
    """
    Strict Leakage Fix (Holdout Set): Evaluates max Tanimoto similarity between Test and Train sets.
    If a test compound has similarity >= threshold to ANY train compound, 
    it is moved to the training set to prevent data leakage.
    """
    print(f"\n--- Enforcing Strict Tanimoto Threshold (< {threshold}) on Holdout Split ---")
    
    df = df.copy()
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"
    
    train_smiles = df[train_mask]["SMILES"].tolist()
    test_smiles = df[test_mask]["SMILES"].tolist()
    test_indices = df[test_mask].index.tolist()
    
    # Use the new MorganGenerator instead of deprecated AllChem method
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
    train_fps = []
    for s in train_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            train_fps.append(fp_gen.GetFingerprint(mol))
    
    max_similarities = []
    leakage_indices = []
    
    for idx, smi in zip(test_indices, test_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = fp_gen.GetFingerprint(mol)
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        max_sim = max(sims) if sims else 0
        max_similarities.append(max_sim)
        
        if max_sim >= threshold:
            leakage_indices.append(idx)

    # Move leaked test samples to train set
    if leakage_indices:
        print(f"WARNING: Found {len(leakage_indices)} test compounds with Tanimoto >= {threshold} to train set.")
        print("Moving them to the training set to strictly prevent data leakage.")
        df.loc[leakage_indices, "split"] = "train"
    else:
        print("Success: No test compounds exceed the similarity threshold.")

    # Re-calculate post-correction similarities for reporting
    final_test_mask = df["split"] == "test"
    final_test_smiles = df[final_test_mask]["SMILES"].tolist()
    
    if len(final_test_smiles) == 0:
        print("ERROR: Test set is empty after leakage removal! Consider adjusting threshold or seed.")
        return df, pd.DataFrame()

    final_max_sims = [sim for sim, idx in zip(max_similarities, test_indices) if idx not in leakage_indices]
    avg_max_sim = np.mean(final_max_sims) if final_max_sims else 0
    overall_max_sim = max(final_max_sims) if final_max_sims else 0
    
    print(f"Final Test Set Size: {len(final_test_smiles)}")
    print(f"Average Maximum Tanimoto Similarity (Test -> Train): {avg_max_sim:.4f}")
    print(f"Overall Maximum Tanimoto Similarity (Test -> Train): {overall_max_sim:.4f}")

    sim_df = pd.DataFrame({
        "Test_SMILES": final_test_smiles,
        "Max_Tanimoto_to_Train": final_max_sims
    })
    sim_df.to_csv(out_csv, index=False)
    
    # Plot distribution
    plt.figure(figsize=(8, 6))
    plt.hist(final_max_sims, bins=20, color='grey', alpha=0.7)
    plt.axvline(avg_max_sim, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {avg_max_sim:.2f}')
    plt.xlabel("Max Tanimoto coefficient to Training Set", fontsize=20)
    plt.ylabel("Number of Test Compounds", fontsize=20)
    plt.legend(frameon=False,fontsize=16)
    plt.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close()
    
    return df, sim_df


def build_features(df, output_csv, ecfp_bits=2048, ecfp_radius=2):
    """Generate ECFP and RDKit descriptors, and remove highly correlated/constant features."""
    print("\n--- Generating Features ---")
    descriptor_names = [d[0] for d in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=ecfp_radius, fpSize=ecfp_bits)

    def featurize(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            
            ecfp = fp_gen.GetFingerprint(mol)
            ecfp_array = np.zeros((ecfp_bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(ecfp, ecfp_array)

            rdkit_desc = np.array(calculator.CalcDescriptors(mol), dtype=float)
            rdkit_desc[~np.isfinite(rdkit_desc)] = np.nan # Clean infinities

            return np.concatenate([ecfp_array, rdkit_desc])
        except Exception:
            return None

    features, valid_idx = [], []
    for i, smi in enumerate(df["SMILES"]):
        vec = featurize(smi)
        if vec is not None:
            features.append(vec)
            valid_idx.append(i)
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(df)} molecules...", end="\r")
    print("")

    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    initial_cols = [f"ECFP4_{i}" for i in range(ecfp_bits)] + descriptor_names
    feature_df = pd.DataFrame(np.array(features), columns=initial_cols)

    # Feature Cleaning
    nan_threshold = 0.5
    feature_df.dropna(axis=1, thresh=int((1 - nan_threshold) * len(feature_df)), inplace=True)
    
    # Drop zero variance
    n_unique = feature_df.nunique(dropna=True)
    feature_df.drop(columns=n_unique[n_unique <= 1].index, inplace=True)

    # Drop highly correlated (>0.90)
    print("Calculating correlation matrix for feature pruning...")
    corr_matrix = feature_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.90)]
    feature_df.drop(columns=to_drop_corr, inplace=True)

    final_df = pd.concat([df_valid[["SMILES", "label", "split", "scaffold"]], feature_df], axis=1)
    final_df.drop(columns=["scaffold"]).to_csv(output_csv, index=False)
    
    return final_df


def safe_metrics(y_true, y_prob, y_pred, k=10):
    """Compute classification and ranking metrics safely."""
    metrics = {}
    metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) >= 2 else np.nan
    metrics["PR_AUC"] = average_precision_score(y_true, y_prob) if np.sum(y_true) > 0 else np.nan
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) >= 2 else np.nan
    metrics["Balanced_Accuracy"] = balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) >= 2 else np.nan
    
    n_actives = y_true.sum()
    if n_actives > 0 and k <= len(y_true):
        order = np.argsort(y_prob)[::-1]
        y_topk = y_true[order][:k]
        metrics[f"Top{k}_Recall"] = y_topk.sum() / n_actives
        metrics[f"Top{k}_EF"] = metrics[f"Top{k}_Recall"] / (k / len(y_true))
    else:
        metrics[f"Top{k}_Recall"], metrics[f"Top{k}_EF"] = np.nan, np.nan
        
    return metrics


def build_models(y_train, random_state=42):
    """Return dictionary of un-fitted sklearn pipelines."""
    pos = float(np.sum(y_train))
    scale_pos_weight = float(len(y_train) - pos) / pos if pos > 0 else 1.0

    return {
        "RF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=random_state, n_jobs=-1))
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state))
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance"))
        ]),
        "XGB": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, scale_pos_weight=scale_pos_weight, random_state=random_state))
        ])
    }


def evaluate_models(X_train, y_train, X_test, y_test, out_prefix, out_dir_models, save_models=True):
    """Train models and evaluate on holdout set."""
    summary = []
    models = build_models(y_train)

    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            metrics = safe_metrics(y_test.values, y_prob, y_pred, k=10)
            metrics20 = safe_metrics(y_test.values, y_prob, y_pred, k=20)
            
            summary.append({
                "Model": name,
                "ROC_AUC": metrics["ROC_AUC"],
                "PR_AUC": metrics["PR_AUC"],
                "MCC": metrics["MCC"],
                "Balanced_Accuracy": metrics["Balanced_Accuracy"],
                "Top10_Recall": metrics["Top10_Recall"],
                "Top10_EF": metrics["Top10_EF"],
                "Top20_Recall": metrics20["Top20_Recall"],
                "Top20_EF": metrics20["Top20_EF"],
            })

            if save_models:
                joblib.dump(model, os.path.join(out_dir_models, f"{name}_{out_prefix}_model.pkl"))
                
        except Exception as e:
            print(f"Error training {name}: {e}")

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(out_dir_models, f"{out_prefix}_ranking_performance_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nRanking & Classification metrics (Holdout Set):")
    print(summary_df)
    return summary_df


def plot_kfold_mean_curves(curve_data, details_df, out_prefix, out_dir_plots):
    """Plot ROC, PR, and Enrichment curves from CV data."""
    if not curve_data:
        return

    base_font = plt.rcParams.get("font.size", 10)
    plt.rcParams.update({
        "font.size": base_font + 3,
        "axes.labelsize": base_font + 8,
        "xtick.labelsize": base_font + 8,
        "ytick.labelsize": base_font + 8,
        "legend.fontsize": base_font + 3,
    })

    fpr_grid = np.linspace(0, 1, 101)
    recall_grid = np.linspace(0, 1, 101)

    # 1. ROC Curves
    plt.figure()
    for name, data in curve_data.items():
        roc_curves = data.get("roc", [])
        if not roc_curves: continue
        tprs = [np.interp(fpr_grid, fpr, tpr) for fpr, tpr in roc_curves]
        mean_tpr, std_tpr = np.mean(tprs, axis=0), np.std(tprs, axis=0)
        
        auc_vals = details_df.loc[details_df["Model"] == name, "ROC_AUC"].dropna()
        if not auc_vals.empty:
            plt.plot(fpr_grid, mean_tpr, label=f"{name} (AUC={auc_vals.mean():.2f} ± {auc_vals.std():.2f})")
            plt.fill_between(fpr_grid, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), alpha=0.05)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_plots, f"{out_prefix}_ROC_curves_kfold_mean.png"), dpi=300, transparent=True)
    plt.close()

    # 2. PR Curves
    plt.figure()
    for name, data in curve_data.items():
        pr_curves = data.get("pr", [])
        if not pr_curves: continue
        precisions = [np.interp(recall_grid, recall[::-1], precision[::-1]) for recall, precision in pr_curves]
        mean_precision, std_precision = np.mean(precisions, axis=0), np.std(precisions, axis=0)
        
        ap_vals = details_df.loc[details_df["Model"] == name, "PR_AUC"].dropna()
        if not ap_vals.empty:
            plt.plot(recall_grid, mean_precision, label=f"{name} (AP={ap_vals.mean():.2f} ± {ap_vals.std():.2f})")
            plt.fill_between(recall_grid, np.maximum(mean_precision - std_precision, 0), np.minimum(mean_precision + std_precision, 1), alpha=0.05)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_plots, f"{out_prefix}_PR_curves_kfold_mean.png"), dpi=300, transparent=True)
    plt.close()

    # 3. Enrichment Curves
    plt.figure()
    plotted_enrichment = False
    for name, data in curve_data.items():
        enrich_curves = data.get("enrichment", [])
        if not enrich_curves: continue
        frac_grid = np.linspace(0, 1, 101)
        recalls = [np.interp(frac_grid, frac, rec) for frac, rec in enrich_curves]
        mean_recall, std_recall = np.mean(recalls, axis=0), np.std(recalls, axis=0)
        
        top10_vals = details_df.loc[details_df["Model"] == name, "Top10_Recall"].dropna()
        if not top10_vals.empty:
            plt.plot(frac_grid, mean_recall, label=f"{name} (Top10 recall={top10_vals.mean():.2f} ± {top10_vals.std():.2f})")
            plt.fill_between(frac_grid, np.maximum(mean_recall - std_recall, 0), np.minimum(mean_recall + std_recall, 1), alpha=0.05)
            plotted_enrichment = True

    if plotted_enrichment:
        plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
        plt.xlabel("Fraction screened")
        plt.ylabel("Recall")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_plots, f"{out_prefix}_enrichment_kfold_mean.png"), dpi=300, transparent=True)
    plt.close()


def scaffold_kfold_cv(final_df, k, seed, out_prefix, out_dir_cv, out_dir_plots, leakage_thresh=0.8):
    """Perform strict scaffold-aware K-Fold cross validation with Tanimoto constraint."""
    print(f"\n--- Starting Strict Scaffold {k}-Fold CV (Tanimoto < {leakage_thresh}) ---")
    
    feature_cols = [c for c in final_df.columns if c not in {"SMILES", "label", "split", "scaffold"}]
    X = final_df[feature_cols].to_numpy()
    y = final_df["label"].to_numpy()
    scaffolds = final_df["scaffold"].tolist()
    smiles_list = final_df["SMILES"].tolist()

    # Precompute fingerprints for Tanimoto constraints
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fps.append(fp_gen.GetFingerprint(mol))
        else:
            fps.append(None)

    # Group by scaffold
    scaffold_to_indices = defaultdict(list)
    for i, scaf in enumerate(scaffolds):
        scaffold_to_indices[scaf].append(i)

    items = [(scaf, idxs, len(idxs), int(np.sum(y[idxs]))) for scaf, idxs in scaffold_to_indices.items()]
    random.Random(seed).shuffle(items)
    items.sort(key=lambda x: x[2], reverse=True) # Sort by size

    fold_indices, fold_sizes, fold_actives = [[] for _ in range(k)], [0]*k, [0]*k
    for _, idxs, n_total, n_active in items:
        best = min(range(k), key=lambda i: (fold_sizes[i], fold_actives[i]))
        fold_indices[best].extend(idxs)
        fold_sizes[best] += n_total
        fold_actives[best] += n_active

    details = []
    curve_data = {}
    
    for fold_id, test_idx in enumerate(fold_indices, start=1):
        print(f"\nProcessing Fold {fold_id}/{k}...")
        
        # Initial CV Split
        base_train_idx = np.setdiff1d(np.arange(len(y)), np.array(test_idx)).tolist()
        
        # Apply Strict Tanimoto Constraint (Move leaks to train)
        train_fps = [fps[i] for i in base_train_idx if fps[i] is not None]
        strict_train_idx = list(base_train_idx)
        strict_test_idx = []
        
        for idx in test_idx:
            if fps[idx] is None:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fps[idx], train_fps)
            if sims and max(sims) >= leakage_thresh:
                strict_train_idx.append(idx)
            else:
                strict_test_idx.append(idx)
                
        leaked_count = len(test_idx) - len(strict_test_idx)
        print(f"Fold {fold_id}: Moved {leaked_count} leaked test compounds to train set.")
        print(f"Fold {fold_id} Final Sizes -> Train: {len(strict_train_idx)}, Test: {len(strict_test_idx)}")
        
        if len(strict_test_idx) == 0:
            print(f"WARNING: Fold {fold_id} has no test compounds left after strict filtering! Skipping.")
            continue

        X_train, y_train = X[strict_train_idx], y[strict_train_idx]
        X_test, y_test = X[strict_test_idx], y[strict_test_idx]
        
        models = build_models(y_train, random_state=seed)

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            # Store curve data
            curve_data.setdefault(name, {}).setdefault("roc", [])
            curve_data.setdefault(name, {}).setdefault("pr", [])
            
            if len(np.unique(y_test)) >= 2:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                curve_data[name]["roc"].append((fpr, tpr))
            if np.sum(y_test) > 0:
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                curve_data[name]["pr"].append((recall, precision))
                
            order = np.argsort(y_prob)[::-1]
            y_sorted = y_test[order]
            n_total = len(y_sorted)
            n_actives = int(y_sorted.sum())
            if n_actives > 0 and n_total > 0:
                cum_actives = np.cumsum(y_sorted)
                frac_screened = np.arange(1, n_total + 1) / n_total
                recall_curve = cum_actives / n_actives
                curve_data[name].setdefault("enrichment", []).append((frac_screened, recall_curve))

            # Store metrics
            metrics = safe_metrics(y_test, y_prob, y_pred, k=10)
            metrics20 = safe_metrics(y_test, y_prob, y_pred, k=20)
            
            metrics.update({
                "Fold": fold_id, "Model": name,
                "Top20_Recall": metrics20["Top20_Recall"], "Top20_EF": metrics20["Top20_EF"]
            })
            details.append(metrics)

    details_df = pd.DataFrame(details)
    metric_cols = ["ROC_AUC", "PR_AUC", "MCC", "Balanced_Accuracy", "Top10_Recall", "Top10_EF", "Top20_Recall", "Top20_EF"]
    summary = details_df.groupby("Model")[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["Model"] + [f"{m}_{s}" for m in metric_cols for s in ["mean", "sd"]]

    details_df.to_csv(os.path.join(out_dir_cv, f"{out_prefix}_cv_details.csv"), index=False)
    summary.to_csv(os.path.join(out_dir_cv, f"{out_prefix}_cv_summary.csv"), index=False)
    
    # Plot the curves using the strictly filtered data
    plot_kfold_mean_curves(curve_data, details_df, out_prefix, out_dir_plots)
    
    return details_df, summary


def main():
    parser = argparse.ArgumentParser(description="Run complete scaffold-aware ML pipeline.")
    parser.add_argument("--input", default="sortaseA_compound.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--leakage-thresh", type=float, default=0.8, help="Max Tanimoto threshold for strict split")
    parser.add_argument("--ecfp-bits", type=int, default=2048)
    parser.add_argument("--ecfp-radius", type=int, default=2)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--out-prefix", default="sortaseA")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Error: Input file '{args.input}' not found.")

    dirs = {k: os.path.join(args.out_dir, k) for k in ["statistics", "split", "features", "models", "plots", "cv"]}
    for p in dirs.values(): ensure_dir(p)

    # 1. Load Data & Generate Scaffolds
    df = pd.read_csv(args.input)
    print("Computing generic scaffolds (stereochem stripped)...")
    df["scaffold"] = df["SMILES"].apply(get_scaffold)
    df = df.dropna(subset=["scaffold"]).reset_index(drop=True)
    stats_df = scaffold_statistics(df, os.path.join(dirs["statistics"], "scaffold_statistics.csv"))

    # 2. Split Data
    df_split = scaffold_split(df, seed=args.seed, train_frac=args.train_frac)
    
    # 3. Apply Strict Similarity Leakage Fix
    sim_csv_path = os.path.join(dirs["statistics"], f"{args.out_prefix}_train_test_max_tanimoto.csv")
    sim_plot_path = os.path.join(dirs["plots"], f"{args.out_prefix}_train_test_max_tanimoto.png")
    df_strict_split, sim_df = enforce_similarity_threshold(
        df_split, out_csv=sim_csv_path, out_plot=sim_plot_path, threshold=args.leakage_thresh
    )
    df_strict_split.to_csv(os.path.join(dirs["split"], f"{args.out_prefix}_scaffold_split_strict.csv"), index=False)

    # 4. Generate and Clean Features
    features_path = os.path.join(dirs["features"], f"{args.out_prefix}_features_ecfp4_rdkit.csv")
    final_df = build_features(df_strict_split, output_csv=features_path, ecfp_bits=args.ecfp_bits, ecfp_radius=args.ecfp_radius)

    # 5. Model Training & Evaluation on Holdout Set
    split_mask = final_df["split"]
    feature_cols = [c for c in final_df.columns if c not in {"SMILES", "label", "split", "scaffold"}]
    
    holdout_summary_df = evaluate_models(
        X_train=final_df.loc[split_mask == "train", feature_cols],
        y_train=final_df.loc[split_mask == "train", "label"],
        X_test=final_df.loc[split_mask == "test", feature_cols],
        y_test=final_df.loc[split_mask == "test", "label"],
        out_prefix=args.out_prefix,
        out_dir_models=dirs["models"]
    )

    # 6. Strict Scaffold K-Fold Cross Validation & Plotting
    cv_details_df, cv_summary_df = scaffold_kfold_cv(
        final_df, k=args.k_folds, seed=args.seed, 
        out_prefix=args.out_prefix, out_dir_cv=dirs["cv"], 
        out_dir_plots=dirs["plots"], leakage_thresh=args.leakage_thresh
    )

    # 7. Consolidate results into a single Excel file
    excel_path = os.path.join(args.out_dir, f"{args.out_prefix}_consolidated_results.xlsx")
    print(f"\n--- Saving All Important Results to Excel: {excel_path} ---")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        stats_df.to_excel(writer, sheet_name='Scaffold_Stats', index=False)
        sim_df.to_excel(writer, sheet_name='Split_Similarity', index=False)
        holdout_summary_df.to_excel(writer, sheet_name='Holdout_Performance', index=False)
        cv_summary_df.to_excel(writer, sheet_name='CV_Summary', index=False)
        cv_details_df.to_excel(writer, sheet_name='CV_Fold_Details', index=False)
    print("Pipeline execution complete.")

if __name__ == "__main__":
    main()