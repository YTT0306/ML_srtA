import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    scores_csv = "RF_scores.csv"
    out_dir = os.path.join("results", "plots")
    out_path = os.path.join(out_dir, "selenoline_rank_probactive2.png")

    ensure_dir(out_dir)

    base_fontsize = plt.rcParams.get("font.size", 10)
    plt.rcParams.update(
        {
            "font.size": base_fontsize + 3,
            "axes.labelsize": base_fontsize + 8,
            "xtick.labelsize": base_fontsize + 8,
            "ytick.labelsize": base_fontsize + 8,
            "legend.fontsize": base_fontsize + 3,
        }
    )

    df = pd.read_csv(scores_csv)
    if not {"ID", "Prob_Active", "class"}.issubset(df.columns):
        raise ValueError("scores CSV must contain ID, Prob_Active, and class columns.")

    df = df.sort_values("Prob_Active", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    df["ID"] = df["ID"].astype(str)
    df["class"] = df["class"].astype(str)

    colors = {
        "1": "#43A942",
        "2": "#2E80B8",
        "3": "#ED7D31",
        "4": "#9E9E9E",  
    }

    plt.figure(figsize=(7, 6))

    #class 1 & 2
    plot_order = ["2", "1"]
    df_plot = df[~df["class"].isin(["3", "4"])]  
    for cls in plot_order:
        sub = df_plot[df_plot["class"] == cls]
        plt.scatter(
            sub["rank"],
            sub["Prob_Active"],
            s=5,
            marker="|",
            c=colors.get(cls, "#B0B0B0"),
            alpha=0.3,
            label="thioline" if cls == "2" else "selenoline",
            zorder=1,
            linewidths=0.1,
        )

    # class 3: hits
    hits = df[df["class"] == "3"]
    if not hits.empty:
        plt.scatter(
            hits["rank"],
            hits["Prob_Active"],
            s=50,
            c=colors["3"],
            linewidths=0.5,
            marker="*",
            alpha=1,
            label="hits",
            zorder=2,
        )

    #  class 4：miss
    cls4 = df[df["class"] == "4"]
    if not cls4.empty:
        plt.scatter(
            cls4["rank"],
            cls4["Prob_Active"],
            s=50,
            c=colors["4"],
            linewidths=0.5,
            marker="*",
            alpha=1,
            label="class 4",
            zorder=3,
        )

    plt.xlabel("Ranked compounds")
    plt.ylabel("Prediction score")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors.get("2", "#2E80B8"),
            marker="o",
            linestyle="None",
            markersize=6,
            label="1,2-benzothiazol-3-one derivatives",
        ),
        Line2D(
            [0],
            [0],
            color=colors.get("1", "#43A942"),
            marker="o",
            linestyle="None",
            markersize=6,
            label="1,2-benzoselenazol-3-one derivatives",
        ),
        Line2D(
            [0],
            [0],
            color=colors.get("3", "#ED7D31"),
            marker="*",
            linestyle="None",
            markersize=8,
            label="Hit",
        ),
        Line2D(
            [0],
            [0],
            color=colors.get("4", "#9E9E9E"),
            marker="*",
            linestyle="None",
            markersize=8,
            label="Miss",
        ),
    ]

    plt.legend(handles=legend_handles, frameon=False, handlelength=1, handletextpad=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, transparent=True)
    plt.close()

    print(f"Rank plot saved to {out_path}")


if __name__ == "__main__":
    main()