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
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return None


def scaffold_statistics(df, out_csv):
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
    print(f"Scaffold statistics saved to {out_csv}")


def scaffold_split(df, seed=42, train_frac=0.8):
    rng = random.Random(seed)

    scaffold_groups = {}
    for idx, row in df.iterrows():
        scaffold_groups.setdefault(row["scaffold"], []).append(idx)

    active_scaffolds = []
    inactive_scaffolds = []
    for scaffold, indices in scaffold_groups.items():
        n_active = int(df.loc[indices, "label"].sum())
        if n_active > 0:
            active_scaffolds.append((scaffold, indices, n_active))
        else:
            inactive_scaffolds.append((scaffold, indices, 0))

    rng.shuffle(active_scaffolds)
    rng.shuffle(inactive_scaffolds)

    n_total = len(df)
    target_train = int(train_frac * n_total)

    train_idx, test_idx = [], []

    total_actives = int(df["label"].sum())
    target_train_actives = int(train_frac * total_actives)
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

    def check(name):
        sub = df[df["split"] == name]
        return len(sub), int(sub["label"].sum())

    print("Train:", check("train"))
    print("Test:", check("test"))

    if df[df["split"] == "test"]["label"].sum() == 0:
        print("WARNING: Test set has no active compounds! Consider changing the seed.")
    
    # Check disjoint
    train_scaffolds = set(df[df["split"] == "train"]["scaffold"])
    test_scaffolds = set(df[df["split"] == "test"]["scaffold"])
    intersection = train_scaffolds.intersection(test_scaffolds)
    assert len(intersection) == 0, f"Scaffold overlap detected: {intersection}"

    print("Scaffold split sanity checks passed.")
    return df


def build_features(df, output_csv, ecfp_bits=2048, ecfp_radius=2):
    print("\n--- Generating Features ---")
    descriptor_names = [d[0] for d in Descriptors._descList]
    print(f"RDKit descriptor count: {len(descriptor_names)}")
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=ecfp_radius, fpSize=ecfp_bits
    )

    def featurize(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            ecfp = fp_gen.GetFingerprint(mol)
            ecfp_array = np.zeros((ecfp_bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(ecfp, ecfp_array)

            rdkit_desc = np.array(calculator.CalcDescriptors(mol), dtype=float)
            
            # Replace Infinity with NaN so Imputer can handle it later
            if not np.isfinite(rdkit_desc).all():
                rdkit_desc[~np.isfinite(rdkit_desc)] = np.nan

            vec = np.concatenate([ecfp_array, rdkit_desc])
            return vec
        except Exception:
            return None

    features = []
    valid_idx = []
    for i, smi in enumerate(df["SMILES"]):
        vec = featurize(smi)
        if vec is not None:
            features.append(vec)
            valid_idx.append(i)
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(df)} molecules...", end="\r")
    print("")

    X = np.array(features)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)

    initial_cols = [f"ECFP4_{i}" for i in range(ecfp_bits)] + descriptor_names
    feature_df = pd.DataFrame(X, columns=initial_cols)
    print(f"Initial feature matrix shape: {feature_df.shape}")

    # --- FEATURE CLEANING & SELECTION ---

    # 1. Drop columns with > 50% NaNs
    nan_threshold = 0.5
    nan_ratios = feature_df.isna().mean()
    cols_to_drop_nan = nan_ratios[nan_ratios > nan_threshold].index
    if len(cols_to_drop_nan) > 0:
        print(f"Dropped {len(cols_to_drop_nan)} columns with > {nan_threshold*100}% NaNs.")
        feature_df.drop(columns=cols_to_drop_nan, inplace=True)

    # 2. Drop columns with Zero Variance (Constant values)
    # Using nunique <= 1 checks for constants (ignoring NaNs)
    n_unique = feature_df.nunique(dropna=True)
    cols_to_drop_const = n_unique[n_unique <= 1].index
    if len(cols_to_drop_const) > 0:
        print(f"Dropped {len(cols_to_drop_const)} constant columns (zero variance).")
        feature_df.drop(columns=cols_to_drop_const, inplace=True)

    # 3. Drop Highly Correlated Features (Correlation > 0.90)
    # This step is computationally expensive but very useful for RDKit descriptors
    print("Calculating correlation matrix (this might take a moment)...")
    corr_matrix = feature_df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than 0.90
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.90)]
    
    if len(to_drop_corr) > 0:
        print(f"Dropped {len(to_drop_corr)} columns due to high correlation (> 0.90).")
        feature_df.drop(columns=to_drop_corr, inplace=True)

    # --- END CLEANING ---

    feature_names = feature_df.columns.tolist()
    print(f"Final feature matrix shape: {feature_df.shape}")

    final_df = pd.concat(
        [df_valid[["SMILES", "label", "split", "scaffold"]], feature_df],
        axis=1,
    )

    final_df.drop(columns=["scaffold"]).to_csv(output_csv, index=False)
    print(f"Feature table saved to: {output_csv}")

    return final_df, feature_names


def topk_metrics(y_true, y_prob, k):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_true.size == 0:
        return np.nan, np.nan

    n_actives = y_true.sum()
    if n_actives == 0:
        return np.nan, np.nan

    if k > len(y_true):
        k = len(y_true)

    order = np.argsort(y_prob)[::-1]
    y_topk = y_true[order][:k]

    recall_k = y_topk.sum() / n_actives
    ef = recall_k / (k / len(y_true))
    return recall_k, ef


def safe_roc_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def safe_pr_auc(y_true, y_prob):
    if np.sum(y_true) == 0:
        return np.nan
    return average_precision_score(y_true, y_prob)


def build_models(y_train, random_state=42):
    pos = float(np.sum(y_train))
    neg = float(len(y_train) - pos)
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    return {
        "RF": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance")),
            ]
        ),
        "XGB": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="logloss",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    out_prefix,
    out_dir_models,
    out_dir_plots,
    save_models=True,
    make_plots=True,
):
    summary = []
    models = build_models(y_train)

    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]

            roc_auc = safe_roc_auc(y_test, y_prob)
            pr_auc = safe_pr_auc(y_test, y_prob)
            recall_10, ef_10 = topk_metrics(y_test, y_prob, k=10)
            recall_20, ef_20 = topk_metrics(y_test, y_prob, k=20)

            summary.append(
                {
                    "Model": name,
                    "ROC_AUC": roc_auc,
                    "PR_AUC": pr_auc,
                    "Top10_Recall": recall_10,
                    "Top10_EF": ef_10,
                    "Top20_Recall": recall_20,
                    "Top20_EF": ef_20,
                }
            )

            if save_models:
                model_path = os.path.join(out_dir_models, f"{name}_{out_prefix}_model.pkl")
                joblib.dump(model, model_path)
                
        except Exception as e:
            print(f"Error training {name}: {e}")

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(out_dir_models, f"{out_prefix}_ranking_performance_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nRanking metrics (Holdout Set):")
    print(summary_df)

    return summary_df


def make_scaffold_folds(scaffolds, y, k, seed=42):
    scaffold_to_indices = defaultdict(list)
    for i, scaf in enumerate(scaffolds):
        scaffold_to_indices[scaf].append(i)

    items = []
    for scaf, idxs in scaffold_to_indices.items():
        n_total = len(idxs)
        n_active = int(np.sum(y[idxs]))
        items.append((scaf, idxs, n_total, n_active))

    rng = random.Random(seed)
    rng.shuffle(items)
    items.sort(key=lambda x: x[2], reverse=True)

    fold_indices = [[] for _ in range(k)]
    fold_sizes = [0] * k
    fold_actives = [0] * k

    for _, idxs, n_total, n_active in items:
        best = min(range(k), key=lambda i: (fold_sizes[i], fold_actives[i]))
        fold_indices[best].extend(idxs)
        fold_sizes[best] += n_total
        fold_actives[best] += n_active

    return fold_indices


def plot_kfold_mean_curves(curve_data, details_df, out_prefix, out_dir_plots):
    if not curve_data:
        return

    base_font = plt.rcParams.get("font.size", 10)
    plt.rcParams.update(
        {
            "font.size": base_font + 3,
            "axes.labelsize": base_font + 8,
            "xtick.labelsize": base_font + 8,
            "ytick.labelsize": base_font + 8,
            "legend.fontsize": base_font + 3,
        }
    )

    fpr_grid = np.linspace(0, 1, 101)
    recall_grid = np.linspace(0, 1, 101)

    # 1. ROC Curves
    plt.figure()
    for name, data in curve_data.items():
        roc_curves = data.get("roc", [])
        if not roc_curves:
            continue
        tprs = [np.interp(fpr_grid, fpr, tpr) for fpr, tpr in roc_curves]
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        auc_vals = details_df.loc[details_df["Model"] == name, "ROC_AUC"]
        mean_auc = auc_vals.mean()
        std_auc = auc_vals.std()
        plt.plot(fpr_grid, mean_tpr, label=f"{name} (AUC={mean_auc:.2f} ± {std_auc:.2f})")
        plt.fill_between(
            fpr_grid,
            np.maximum(mean_tpr - std_tpr, 0),
            np.minimum(mean_tpr + std_tpr, 1),
            alpha=0.05,
        )

    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir_plots, f"{out_prefix}_ROC_curves_kfold_mean.pdf"),
        transparent=True,
    )
    plt.savefig(
        os.path.join(out_dir_plots, f"{out_prefix}_ROC_curves_kfold_mean.png"),
        dpi=300,
        transparent=True,
    )
    plt.close()

    # 2. PR Curves
    plt.figure()
    for name, data in curve_data.items():
        pr_curves = data.get("pr", [])
        if not pr_curves:
            continue
        precisions = [
            np.interp(recall_grid, recall[::-1], precision[::-1]) for recall, precision in pr_curves
        ]
        mean_precision = np.mean(precisions, axis=0)
        std_precision = np.std(precisions, axis=0)
        ap_vals = details_df.loc[details_df["Model"] == name, "PR_AUC"]
        mean_ap = ap_vals.mean()
        std_ap = ap_vals.std()
        plt.plot(recall_grid, mean_precision, label=f"{name} (AP={mean_ap:.2f} ± {std_ap:.2f})")
        plt.fill_between(
            recall_grid,
            np.maximum(mean_precision - std_precision, 0),
            np.minimum(mean_precision + std_precision, 1),
            alpha=0.05,
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir_plots, f"{out_prefix}_PR_curves_kfold_mean.pdf"),
        transparent=True,
    )
    plt.savefig(
        os.path.join(out_dir_plots, f"{out_prefix}_PR_curves_kfold_mean.png"),
        dpi=300,
        transparent=True,
    )
    plt.close()

    # 3. Enrichment Curves
    plt.figure()
    plotted_enrichment = False
    for name, data in curve_data.items():
        enrich_curves = data.get("enrichment", [])
        if not enrich_curves:
            continue
        frac_grid = np.linspace(0, 1, 101)
        recalls = [np.interp(frac_grid, frac, rec) for frac, rec in enrich_curves]
        mean_recall = np.mean(recalls, axis=0)
        std_recall = np.std(recalls, axis=0)
        top10_vals = details_df.loc[details_df["Model"] == name, "Top10_Recall"]
        top10_mean = top10_vals.mean()
        top10_sd = top10_vals.std()
        plt.plot(
            frac_grid,
            mean_recall,
            label=f"{name} (Top10 recall={top10_mean:.2f} ± {top10_sd:.2f})",
        )
        plt.fill_between(
            frac_grid,
            np.maximum(mean_recall - std_recall, 0),
            np.minimum(mean_recall + std_recall, 1),
            alpha=0.05,
        )
        plotted_enrichment = True

    if plotted_enrichment:
        plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
        plt.xlabel("Fraction screened")
        plt.ylabel("Recall")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir_plots, f"{out_prefix}_enrichment_kfold_mean.pdf"),
            transparent=True,
        )
        plt.savefig(
            os.path.join(out_dir_plots, f"{out_prefix}_enrichment_kfold_mean.png"),
            dpi=300,
            transparent=True,
        )
    plt.close()


def scaffold_kfold_cv(final_df, k, seed, out_prefix, out_dir_cv, out_dir_plots):
    print("\n--- Starting Scaffold k-Fold CV ---")
    feature_cols = [
        c for c in final_df.columns if c not in {"SMILES", "label", "split", "scaffold"}
    ]
    X = final_df[feature_cols].to_numpy()
    y = final_df["label"].to_numpy()
    scaffolds = final_df["scaffold"].tolist()

    fold_indices = make_scaffold_folds(scaffolds, y, k, seed)

    details = []
    curve_data = {}
    
    for fold_id, test_idx in enumerate(fold_indices, start=1):
        print(f"Processing Fold {fold_id}/{k}...")
        train_idx = np.setdiff1d(np.arange(len(y)), np.array(test_idx))

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        models = build_models(y_train, random_state=seed)

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]

            curve_data.setdefault(name, {}).setdefault("roc", [])
            curve_data.setdefault(name, {}).setdefault("pr", [])

            if len(np.unique(y_test)) >= 2:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                curve_data[name]["roc"].append((fpr, tpr))
            if np.sum(y_test) > 0:
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                curve_data[name]["pr"].append((recall, precision))

            details.append(
                {
                    "Fold": fold_id,
                    "Model": name,
                    "ROC_AUC": safe_roc_auc(y_test, y_prob),
                    "PR_AUC": safe_pr_auc(y_test, y_prob),
                    "Top10_Recall": topk_metrics(y_test, y_prob, k=10)[0],
                    "Top10_EF": topk_metrics(y_test, y_prob, k=10)[1],
                    "Top20_Recall": topk_metrics(y_test, y_prob, k=20)[0],
                    "Top20_EF": topk_metrics(y_test, y_prob, k=20)[1],
                }
            )

            order = np.argsort(y_prob)[::-1]
            y_sorted = y_test[order]
            n_total = len(y_sorted)
            n_actives = int(y_sorted.sum())

            if n_actives > 0 and n_total > 0:
                cum_actives = np.cumsum(y_sorted)
                frac_screened = np.arange(1, n_total + 1) / n_total
                recall_curve = cum_actives / n_actives
                curve_data[name].setdefault("enrichment", []).append(
                    (frac_screened, recall_curve)
                )

    details_df = pd.DataFrame(details)
    details_path = os.path.join(out_dir_cv, f"{out_prefix}_scaffold_kfold_cv_details.csv")
    details_df.to_csv(details_path, index=False)
    print(f"Scaffold-aware k-fold details saved to {details_path}")

    metric_cols = ["ROC_AUC", "PR_AUC", "Top10_Recall", "Top10_EF", "Top20_Recall", "Top20_EF"]
    summary = details_df.groupby("Model")[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "Model"
    ] + [f"{metric}_{stat}" for metric in metric_cols for stat in ["mean", "sd"]]

    summary_path = os.path.join(out_dir_cv, f"{out_prefix}_scaffold_kfold_cv_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Scaffold-aware k-fold summary saved to {summary_path}")
    print(summary)

    plot_kfold_mean_curves(curve_data, details_df, out_prefix, out_dir_plots)

    return details_df, summary


def main():
    parser = argparse.ArgumentParser(
        description="Run scaffold statistics, scaffold split, features, models, and scaffold-aware CV."
    )
    parser.add_argument("--input", default="sortaseA_compound.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--ecfp-bits", type=int, default=2048)
    parser.add_argument("--ecfp-radius", type=int, default=2)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--out-prefix", default="sortaseA")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    # Input validation
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    out_dir_base = args.out_dir
    out_dir_stats = os.path.join(out_dir_base, "statistics")
    out_dir_split = os.path.join(out_dir_base, "split")
    out_dir_features = os.path.join(out_dir_base, "features")
    out_dir_models = os.path.join(out_dir_base, "models")
    out_dir_plots = os.path.join(out_dir_base, "plots")
    out_dir_cv = os.path.join(out_dir_base, "cv")

    for path in [
        out_dir_stats,
        out_dir_split,
        out_dir_features,
        out_dir_models,
        out_dir_plots,
        out_dir_cv,
    ]:
        ensure_dir(path)

    df = pd.read_csv(args.input)
    assert {"SMILES", "label"}.issubset(df.columns), "Missing required columns: SMILES or label"

    print("Computing Scaffolds...")
    df["scaffold"] = df["SMILES"].apply(get_scaffold)
    df = df.dropna(subset=["scaffold"]).reset_index(drop=True)

    scaffold_statistics(df, os.path.join(out_dir_stats, "scaffold_statistics.csv"))

    df_split = scaffold_split(df, seed=args.seed, train_frac=args.train_frac)
    split_path = os.path.join(out_dir_split, f"{args.out_prefix}_scaffold_split.csv")
    df_split.to_csv(split_path, index=False)
    print(f"Scaffold split saved to {split_path}")

    features_path = os.path.join(out_dir_features, f"{args.out_prefix}_features_ecfp4_rdkit.csv")
    final_df, _ = build_features(
        df_split,
        output_csv=features_path,
        ecfp_bits=args.ecfp_bits,
        ecfp_radius=args.ecfp_radius,
    )

    feature_cols = [
        c for c in final_df.columns if c not in {"SMILES", "label", "split", "scaffold"}
    ]
    X = final_df[feature_cols]
    y = final_df["label"]
    split = final_df["split"]

    X_train = X[split == "train"]
    y_train = y[split == "train"]
    X_test = X[split == "test"]
    y_test = y[split == "test"]

    print("\n--- Evaluating Models on Holdout Split ---")
    evaluate_models(
        X_train,
        y_train,
        X_test,
        y_test,
        out_prefix=args.out_prefix,
        out_dir_models=out_dir_models,
        out_dir_plots=out_dir_plots,
        save_models=True,
        make_plots=False,
    )

    scaffold_kfold_cv(
        final_df,
        k=args.k_folds,
        seed=args.seed,
        out_prefix=args.out_prefix,
        out_dir_cv=out_dir_cv,
        out_dir_plots=out_dir_plots,
    )


if __name__ == "__main__":
    main()