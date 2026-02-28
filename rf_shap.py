#Model Interpretation: Applies SHAP values to identify key structural drivers of activity.

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
    os.makedirs(path, exist_ok=True)


def style_axes(ax, width=6, height=9, tick_pad=1):
    ax.figure.set_size_inches(width, height)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    ax.tick_params(axis="both", which="major", pad=tick_pad)


def adjust_left_margin(left=0.35, right=0.98, top=0.98, bottom=0.12):
    plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)


def load_data(features_csv):
    df = pd.read_csv(features_csv)
    X = df.drop(columns=["SMILES", "label", "split"])
    y = df["label"]
    split = df["split"]
    return X, y, split


def train_rf(X_train, y_train, seed):
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def sample_explain_set(X_train, sample_size, seed):
    if sample_size <= 0 or sample_size >= len(X_train):
        return X_train.copy()
    return X_train.sample(n=sample_size, random_state=seed)


def shap_plots(model, X_explain, out_dir, out_prefix, max_display):
    ensure_dir(out_dir)

    base_font = plt.rcParams.get("font.size", 10)
    plt.rcParams.update(
        {
            "font.size": base_font + 11,
            "axes.labelsize": base_font + 11,
            "xtick.labelsize": base_font + 11,
            "ytick.labelsize": base_font + 11,
            "legend.fontsize": base_font + 11,
        }
    )
    plt.rcParams.update(
        {
            "axes.edgecolor": "#000000",
            "axes.linewidth": 1.0,
        }
    )

    X_explain_imp = model.named_steps["imputer"].transform(X_explain)
    X_explain_imp = pd.DataFrame(X_explain_imp, columns=X_explain.columns)

    explainer = shap.TreeExplainer(model.named_steps["clf"])
    shap_values = explainer.shap_values(X_explain_imp)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        if shap_values.shape[0] == 2:
            shap_values = shap_values[1]
        elif shap_values.shape[-1] == 2:
            shap_values = shap_values[:, :, 1]

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_explain_imp,
        show=False,
        max_display=10,
    )
    plt.tight_layout()
    ax = plt.gca()
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#000000")
    plt.xlabel("SHAP value")
    style_axes(ax, width=6, height=6, tick_pad=1)
    adjust_left_margin(left=0.42)
    plt.savefig(os.path.join(out_dir, f"{out_prefix}_shap_summary.pdf"), transparent=True)
    plt.savefig(
        os.path.join(out_dir, f"{out_prefix}_shap_summary.png"),
        dpi=300,
        transparent=True,
    )
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_explain_imp,
        plot_type="bar",
        show=False,
        max_display=max_display,
    )
    plt.tight_layout()
    ax = plt.gca()
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#000000")
    plt.xlabel("SHAP value")
    style_axes(ax, width=6, height=6, tick_pad=1)
    adjust_left_margin(left=0.42)
    plt.savefig(os.path.join(out_dir, f"{out_prefix}_shap_bar.pdf"), transparent=True)
    plt.savefig(
        os.path.join(out_dir, f"{out_prefix}_shap_bar.png"),
        dpi=300,
        transparent=True,
    )
    plt.close()

    top_features = np.argsort(np.abs(shap_values).mean(axis=0))[::-1]
    if len(top_features) > 0:
        feature_idx = int(top_features[0])
        feature_name = X_explain_imp.columns[feature_idx]
        shap.dependence_plot(
            feature_name,
            shap_values,
            X_explain_imp,
            show=False,
        )
        plt.tight_layout()
        ax = plt.gca()
        for side in ["top", "right", "bottom", "left"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("#000000")
        style_axes(ax, width=6, height=8, tick_pad=1)
        plt.savefig(os.path.join(out_dir, f"{out_prefix}_shap_dependence.pdf"), transparent=True)
        plt.savefig(
            os.path.join(out_dir, f"{out_prefix}_shap_dependence.png"),
            dpi=300,
            transparent=True,
        )
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train RF and run SHAP analysis on a sampled training set."
    )
    parser.add_argument(
        "--features",
        default="results/features/sortaseA_features_ecfp4_rdkit.csv",
        help="Feature CSV with SMILES/label/split columns",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--out-dir", default="results/shap")
    parser.add_argument("--out-prefix", default="sortaseA")
    parser.add_argument("--max-display", type=int, default=20)
    args = parser.parse_args()

    X, y, split = load_data(args.features)
    X_train = X[split == "train"]
    y_train = y[split == "train"]

    model = train_rf(X_train, y_train, seed=args.seed)
    X_explain = sample_explain_set(X_train, args.sample_size, seed=args.seed)

    shap_plots(
        model,
        X_explain,
        out_dir=args.out_dir,
        out_prefix=args.out_prefix,
        max_display=args.max_display,
    )

    print(f"SHAP plots saved under {args.out_dir}")


if __name__ == "__main__":
    main()
