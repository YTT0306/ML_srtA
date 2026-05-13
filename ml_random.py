import argparse
import os
import sys
import random

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, AllChem
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
from sklearn.model_selection import train_test_split, StratifiedKFold


def ensure_dir(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def random_split(df, seed=42, train_frac=0.8):
    """Standard stratified random split."""
    print("\n--- Performing Stratified Random Split ---")
    train_idx, test_idx = train_test_split(
        df.index, 
        train_size=train_frac, 
        random_state=seed, 
        stratify=df["label"]
    )
    
    df = df.copy()
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    print(f"Train split: {len(train_idx)} samples, {df.loc[train_idx, 'label'].sum()} actives")
    print(f"Test split: {len(test_idx)} samples, {df.loc[test_idx, 'label'].sum()} actives")
    return df


def evaluate_split_similarity(df, out_csv, out_plot):
    """
    Calculates Tanimoto similarity between Train and Test sets.
    Unlike the scaffold script, this ONLY reports the leakage and does NOT fix it,
    serving as a baseline to prove the reviewer's point about random split leakage.
    """
    print("\n--- Calculating Tanimoto Similarity (Random Split Baseline) ---")
    
    train_smiles = df[df["split"] == "train"]["SMILES"].tolist()
    test_smiles = df[df["split"] == "test"]["SMILES"].tolist()
    
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) for s in train_smiles if Chem.MolFromSmiles(s)]
    
    max_similarities = []
    valid_test_smiles = []
    
    for smi in test_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        max_similarities.append(max(sims))
        valid_test_smiles.append(smi)

    avg_max_sim = np.mean(max_similarities)
    overall_max_sim = max(max_similarities)
    leaked_count = sum(1 for sim in max_similarities if sim >= 0.8)
    
    print(f"Test Set Size: {len(valid_test_smiles)}")
    print(f"Compounds with Tanimoto >= 0.8 (LEAKAGE): {leaked_count} ({(leaked_count/len(valid_test_smiles)):.1%})")
    print(f"Average Maximum Tanimoto Similarity (Test -> Train): {avg_max_sim:.4f}")
    print(f"Overall Maximum Tanimoto Similarity (Test -> Train): {overall_max_sim:.4f}")

    sim_df = pd.DataFrame({
        "Test_SMILES": valid_test_smiles,
        "Max_Tanimoto_to_Train": max_similarities
    })
    sim_df.to_csv(out_csv, index=False)
    
    # Plot distribution
    plt.figure(figsize=(8, 6))
    plt.hist(max_similarities, bins=20, color='grey', alpha=0.7)
    plt.axvline(avg_max_sim, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {avg_max_sim:.2f}')
    plt.axvline(0.8, color='red', linestyle='dashed', linewidth=2, label='Leakage Threshold')
    plt.xlabel("Max Tanimoto coefficient to Training Set", fontsize=20)
    plt.ylabel("Number of Test Compounds", fontsize=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(frameon=False,fontsize=16)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close()
    
    return sim_df


def build_features(df, output_csv, ecfp_bits=2048, ecfp_radius=2):
    """Generate ECFP and RDKit descriptors."""
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
            rdkit_desc[~np.isfinite(rdkit_desc)] = np.nan 

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

    nan_threshold = 0.5
    feature_df.dropna(axis=1, thresh=int((1 - nan_threshold) * len(feature_df)), inplace=True)
    
    n_unique = feature_df.nunique(dropna=True)
    feature_df.drop(columns=n_unique[n_unique <= 1].index, inplace=True)

    print("Calculating correlation matrix for feature pruning...")
    corr_matrix = feature_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.90)]
    feature_df.drop(columns=to_drop_corr, inplace=True)

    final_df = pd.concat([df_valid[["SMILES", "label", "split"]], feature_df], axis=1)
    final_df.to_csv(output_csv, index=False)
    
    return final_df


def safe_metrics(y_true, y_prob, y_pred, k=10):
    metrics = {}
    metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) >= 2 else np.nan
    metrics["PR_AUC"] = average_precision_score(y_true, y_prob) if np.sum(y_true) > 0 else np.nan
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)
    metrics["Balanced_Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    
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
    summary_path = os.path.join(out_dir_models, f"{out_prefix}_random_ranking_performance.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nRanking & Classification metrics (Random Holdout Set):")
    print(summary_df)
    return summary_df


def plot_kfold_mean_curves(curve_data, details_df, out_prefix, out_dir_plots):
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
        
        auc_vals = details_df.loc[details_df["Model"] == name, "ROC_AUC"]
        plt.plot(fpr_grid, mean_tpr, label=f"{name} (AUC={auc_vals.mean():.2f} ± {auc_vals.std():.2f})")
        plt.fill_between(fpr_grid, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), alpha=0.05)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_plots, f"{out_prefix}_random_ROC_curves_kfold.png"), dpi=300, transparent=True)
    plt.close()

    # 2. PR Curves
    plt.figure()
    for name, data in curve_data.items():
        pr_curves = data.get("pr", [])
        if not pr_curves: continue
        precisions = [np.interp(recall_grid, recall[::-1], precision[::-1]) for recall, precision in pr_curves]
        mean_precision, std_precision = np.mean(precisions, axis=0), np.std(precisions, axis=0)
        
        ap_vals = details_df.loc[details_df["Model"] == name, "PR_AUC"]
        plt.plot(recall_grid, mean_precision, label=f"{name} (AP={ap_vals.mean():.2f} ± {ap_vals.std():.2f})")
        plt.fill_between(recall_grid, np.maximum(mean_precision - std_precision, 0), np.minimum(mean_precision + std_precision, 1), alpha=0.05)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_plots, f"{out_prefix}_random_PR_curves_kfold.png"), dpi=300, transparent=True)
    plt.close()


def random_kfold_cv(final_df, k, seed, out_prefix, out_dir_cv, out_dir_plots):
    """Perform Standard Stratified K-Fold cross validation."""
    print(f"\n--- Starting Standard Stratified {k}-Fold CV ---")
    
    feature_cols = [c for c in final_df.columns if c not in {"SMILES", "label", "split", "scaffold"}]
    X = final_df[feature_cols].to_numpy()
    y = final_df["label"].to_numpy()

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    details = []
    curve_data = {}
    
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Processing Fold {fold_id}/{k}...")

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        models = build_models(y_train, random_state=seed)

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            curve_data.setdefault(name, {}).setdefault("roc", [])
            curve_data.setdefault(name, {}).setdefault("pr", [])
            
            if len(np.unique(y_test)) >= 2:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                curve_data[name]["roc"].append((fpr, tpr))
            if np.sum(y_test) > 0:
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                curve_data[name]["pr"].append((recall, precision))

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

    details_df.to_csv(os.path.join(out_dir_cv, f"{out_prefix}_random_cv_details.csv"), index=False)
    summary.to_csv(os.path.join(out_dir_cv, f"{out_prefix}_random_cv_summary.csv"), index=False)
    
    plot_kfold_mean_curves(curve_data, details_df, out_prefix, out_dir_plots)
    
    return details_df, summary


def main():
    parser = argparse.ArgumentParser(description="Run Random Split ML pipeline baseline.")
    parser.add_argument("--input", default="sortaseA_compound.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--ecfp-bits", type=int, default=2048)
    parser.add_argument("--ecfp-radius", type=int, default=2)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--out-prefix", default="sortaseA_RANDOM")
    parser.add_argument("--out-dir", default="results_random")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Error: Input file '{args.input}' not found.")

    dirs = {k: os.path.join(args.out_dir, k) for k in ["split", "features", "models", "plots", "cv"]}
    for p in dirs.values(): ensure_dir(p)

    df = pd.read_csv(args.input)
    assert {"SMILES", "label"}.issubset(df.columns), "Missing required columns: SMILES or label"

    # 1. Random Split Data
    df_split = random_split(df, seed=args.seed, train_frac=args.train_frac)
    df_split.to_csv(os.path.join(dirs["split"], f"{args.out_prefix}_random_split.csv"), index=False)
    
    # 2. Evaluate Similarity Leakage (Proof for reviewer)
    sim_csv_path = os.path.join(dirs["split"], f"{args.out_prefix}_train_test_max_tanimoto.csv")
    sim_plot_path = os.path.join(dirs["plots"], f"{args.out_prefix}_train_test_max_tanimoto.png")
    sim_df = evaluate_split_similarity(df_split, out_csv=sim_csv_path, out_plot=sim_plot_path)

    # 3. Generate and Clean Features
    features_path = os.path.join(dirs["features"], f"{args.out_prefix}_features_ecfp4_rdkit.csv")
    final_df = build_features(df_split, output_csv=features_path, ecfp_bits=args.ecfp_bits, ecfp_radius=args.ecfp_radius)

    # 4. Model Training & Evaluation on Holdout Set
    split_mask = final_df["split"]
    feature_cols = [c for c in final_df.columns if c not in {"SMILES", "label", "split"}]
    
    holdout_summary_df = evaluate_models(
        X_train=final_df.loc[split_mask == "train", feature_cols],
        y_train=final_df.loc[split_mask == "train", "label"],
        X_test=final_df.loc[split_mask == "test", feature_cols],
        y_test=final_df.loc[split_mask == "test", "label"],
        out_prefix=args.out_prefix,
        out_dir_models=dirs["models"]
    )

    # 5. Standard Stratified K-Fold Cross Validation
    cv_details_df, cv_summary_df = random_kfold_cv(
        final_df, k=args.k_folds, seed=args.seed, 
        out_prefix=args.out_prefix, out_dir_cv=dirs["cv"], out_dir_plots=dirs["plots"]
    )

    # 6. Consolidate results into Excel
    excel_path = os.path.join(args.out_dir, f"{args.out_prefix}_consolidated_results.xlsx")
    print(f"\n--- Saving Baseline Results to Excel: {excel_path} ---")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        sim_df.to_excel(writer, sheet_name='Split_Similarity', index=False)
        holdout_summary_df.to_excel(writer, sheet_name='Holdout_Performance', index=False)
        cv_summary_df.to_excel(writer, sheet_name='CV_Summary', index=False)
        cv_details_df.to_excel(writer, sheet_name='CV_Fold_Details', index=False)
    print("Baseline pipeline execution complete.")

if __name__ == "__main__":
    main()