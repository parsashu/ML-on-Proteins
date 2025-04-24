import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def histogram(file_path):
    """Plot histograms for specified columns in the protein dataset."""
    df = pd.read_csv(file_path, sep="\t", low_memory=False)

    def plot_distribution(df, column_name, ax):
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")

        null_count = df[column_name].isnull().sum()
        non_null_count = df[column_name].notnull().sum()

        sns.histplot(data=df, x=column_name, bins=30, kde=False, ax=ax)

        formatted_name = (
            column_name.replace("_", " ").replace("(", "(").replace(")", ")")
        )
        ax.set_title(f"Distribution of {formatted_name}", fontsize=10)

        units = ""
        if "C" in column_name:
            units = "(°C)"
        elif "kcal" in column_name:
            units = "(kcal/mol)"
        elif "M" in column_name and not "mol" in column_name:
            units = "(M)"

        ax.set_xlabel(f"{formatted_name} {units}", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)

        if non_null_count > 0:
            median_value = df[column_name].median()
            mode_value = (
                df[column_name].mode()[0]
                if not df[column_name].mode().empty
                else float("nan")
            )
            mean_value = df[column_name].mean()

            info_text = (
                f"Null: {null_count}\n"
                f"Non-null: {non_null_count}\n"
                f"Mean: {mean_value:.2f}\n"
                f"Median: {median_value:.2f}\n"
                f"Mode: {mode_value:.2f}"
            )
        else:
            info_text = (
                f"Null: {null_count}\n"
                f"Non-null: {non_null_count}\n"
                f"No stats (all null)"
            )

        ax.text(
            0.95,
            0.95,
            info_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    columns_to_plot = [
        "ASA",
        "pH",
        "T_(C)",
        "Tm_(C)",
        "m_(kcal/mol/M)",
        "Cm_(M)",
        "∆G_H2O_(kcal/mol)",
        "REVERSIBILITY",
    ]

    num_cols = 3
    num_rows = (len(columns_to_plot) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))
    axes = axes.flatten()

    for i, column in enumerate(columns_to_plot):
        plot_distribution(df, column, axes[i])

    for i in range(len(columns_to_plot), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()
