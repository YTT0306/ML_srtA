import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def ensure_dir(path):
    """Utility to create the output directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def style_axes(ax, width=6, height=9, tick_pad=1):
    """
    Standardizes plot aesthetics for publication-quality figures.
    Adjusts the figure size, label padding, and tick spacing.
    """
    ax.figure.set_size_inches(width, height)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    ax.tick_params(axis="both", which="major", pad=tick_pad)


def adjust_left_margin(left=0.35, right=0.98, top=0.98, bottom=0.12):
    """
    Fine-tunes subplots to prevent long feature names (like RDKit descriptors) 
    from being cut off on the left side of the chart.
    """
    plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)


def load_data(features_csv):
    """
    Loads the processed feature CSV. 
    Separates the features (X), the target labels (y), and the train/test split indicator.
    """
    df = pd.read_csv(features_csv)
    # Drop non-feature metadata columns
    X = df.drop(columns=["SMILES", "label", "split"])
    y = df["label"]
    split = df["split"]
    return X, y, split


def train_rf(X_train, y_train, seed):
    """
    Initializes and trains a Random Forest within a Pipeline.
    
    Includes:
    1. SimpleImputer: Fills missing values with the median (crucial for RDKit descriptors).
    2. RandomForestClassifier: Using 'balanced' class weights to handle the 
       typical scarcity of 'active' compounds in chemical datasets.
    """
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced", # Handles imbalanced active/inactive ratios
                    random_state=seed,
                    n_jobs=-1, # Use all available CPU cores
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def sample_explain_set(X_train, sample_size, seed):
    """
    SHAP calculations can be computationally expensive (especially for large ensembles).
    This function takes a representative sample of the training data to speed up 
    the explanation process without losing significant insights.
    """
    if sample_size <= 0 or sample_size >= len(X_train):
        return X_train.copy()
    return X_train.sample(n=sample_size, random_state=seed)


def shap_plots(model, X_explain, out_dir, out_prefix, max_display):
    """
    Generates and saves three types of SHAP visualizations:
    1. Summary Plot (Beeswarm): Shows how feature values impact model output.
    2. Bar Plot: Global feature importance based on mean absolute SHAP values.
    3. Dependence Plot: Explores the relationship between a single feature and the prediction.
    """
    ensure_dir(out_dir)

    # Global plotting parameters for high readability
    base_font = plt.rcParams.get("font.size", 10)
    plt.rcParams.update({
        "font.size": base_font + 11,
        "axes.labelsize": base_font + 11,
        "xtick.labelsize": base_font + 11,
        "ytick.labelsize": base_font + 11,
        "legend.fontsize": base_font + 11,
        "axes.edgecolor": "#000000",
        "axes.linewidth": 1.0,
    })

    # Prepare data: SHAP requires the data in the state it enters the model (imputed)
    X_explain_imp = model.named_steps["imputer"].transform(X_explain)
    X_explain_imp = pd.DataFrame(X_explain_imp, columns=X_explain.columns)

    # Initialize the TreeExplainer (optimized for Random Forest/XGBoost)
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    shap_values = explainer.shap_values(X_explain_imp)

    # Logic to handle different SHAP output formats (binary classification)
    # We focus on the 'Class 1' (Active) predictions
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        if shap_values.shape[0] == 2:
            shap_values = shap_values[1]
        elif shap_values.shape[-1] == 2:
            shap_values = shap_values[:, :, 1]

    # --- 1. SHAP Summary Plot (Beeswarm) ---
    # High feature value = Red; Low feature value = Blue.
    # Tells you if a feature increases or decreases the probability of activity.
    plt.figure()
    shap.summary_plot(shap_values, X_explain_imp, show=False, max_display=10)
    plt.tight_layout()
    ax = plt.gca()
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
    plt.xlabel("SHAP value (Impact on model output)")
    style_axes(ax, width=6, height=6)
    adjust_left_margin(left=0.42)
    plt.savefig(os.path.join(out_dir, f"{out_prefix}_shap_summary.png"), dpi=300)
    plt.close()

    # --- 2. SHAP Bar Plot ---
    # Pure feature importance ranking (magnitude only).
    plt.figure()
    shap.summary_plot(shap_values, X_explain_imp, plot_type="bar", show=False, max_display=max_display)
    plt.tight_layout()
    ax = plt.gca()
    plt.xlabel("mean(|SHAP value|)")
    style_axes(ax, width=6, height=6)
    adjust_left_margin(left=0.42)
    plt.savefig(os.path.join(out_dir, f"{out_prefix}_shap_bar.png"), dpi=300)
    plt.close()

    # --- 3. SHAP Dependence Plot ---
    # Shows the 'shape' of the influence for the most important feature.
    top_features = np.argsort(np.abs(shap_values).mean(axis=0))[::-1]
    if len(top_features) > 0:
        feature_idx = int(top_features[0])
        feature_name = X_explain_imp.columns[feature_idx]
        shap.dependence_plot(feature_name, shap_values, X_explain_imp, show=False)
        plt.tight_layout()
        ax = plt.gca()
        style_axes(ax, width=6, height=8)
        plt.savefig(os.path.join(out_dir, f"{out_prefix}_shap_dependence.png"), dpi=300)
        plt.close()


def main():
    """
    Main execution flow:
    1. Parse CLI arguments.
    2. Load feature data.
    3. Train a Random Forest model on the training split.
    4. Sub-sample the data for explanation.
    5. Run SHAP and save visual results.
    """
    parser = argparse.ArgumentParser(description="Explainable AI for Molecular Activity")
    parser.add_argument("--features", default="results/features/sortaseA_features_ecfp4_rdkit.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=500, help="Number of molecules to explain")
    parser.add_argument("--out-dir", default="results/shap")
    parser.add_argument("--out-prefix", default="sortaseA")
    parser.add_argument("--max-display", type=int, default=20, help="Max features to show in bar plot")
    args = parser.parse_args()

    X, y, split = load_data(args.features)
    X_train = X[split == "train"]
    y_train = y[split == "train"]

    print(f"Training model on {len(X_train)} compounds...")
    model = train_rf(X_train, y_train, seed=args.seed)
    
    print(f"Generating SHAP explanations for {args.sample_size} samples...")
    X_explain = sample_explain_set(X_train, args.sample_size, seed=args.seed)

    shap_plots(model, X_explain, args.out_dir, args.out_prefix, args.max_display)
    print(f"Analysis complete. Results saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
