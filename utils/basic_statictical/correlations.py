import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def pearson_correlation_heatmap(file_path):
    """Plot Pearson correlation heatmap for specified columns in the protein dataset."""
    df = pd.read_csv(file_path, sep="\t", low_memory=False)

    columns_to_plot = [
        "ASA",
        "pH",
        "T_(C)",
        "Tm_(C)",
        "m_(kcal/mol/M)",
        "Cm_(M)",
        "∆G_H2O_(kcal/mol)",
    ]

    df_clean = df[columns_to_plot].apply(pd.to_numeric, errors="coerce")

    corr_matrix = df_clean.corr(method="pearson")
    sample_sizes = pd.DataFrame(index=columns_to_plot, columns=columns_to_plot)
    for i, col1 in enumerate(columns_to_plot):
        for j, col2 in enumerate(columns_to_plot):
            valid_samples = df_clean[[col1, col2]].dropna().shape[0]
            sample_sizes.loc[col1, col2] = valid_samples
            if valid_samples < 100:
                corr_matrix.iloc[i, j] = np.nan

    plt.figure(figsize=(6, 6))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0,
        cbar_kws={"shrink": 0.8},
        fmt=".2f",
    )

    plt.title("Pearson Correlation Heatmap of Protein Features", fontsize=16)
    plt.tight_layout()
    plt.show()


def spearman_correlation_heatmap(file_path):
    """Plot Spearman correlation heatmap for specified columns in the protein dataset."""
    df = pd.read_csv(file_path, sep="\t", low_memory=False)
    columns_to_plot = [
        "ASA",
        "pH",
        "T_(C)",
        "Tm_(C)",
        "m_(kcal/mol/M)",
        "Cm_(M)",
        "∆G_H2O_(kcal/mol)",
    ]
    df_clean = df[columns_to_plot].apply(pd.to_numeric, errors="coerce")
    corr_matrix = df_clean.corr(method="spearman")
    sample_sizes = pd.DataFrame(index=columns_to_plot, columns=columns_to_plot)
    for i, col1 in enumerate(columns_to_plot):
        for j, col2 in enumerate(columns_to_plot):
            valid_samples = df_clean[[col1, col2]].dropna().shape[0]
            sample_sizes.loc[col1, col2] = valid_samples
            if valid_samples < 100:
                corr_matrix.iloc[i, j] = np.nan
    plt.figure(figsize=(6, 6))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0,
        cbar_kws={"shrink": 0.8},
        fmt=".2f",
    )
    plt.title("Spearman Correlation Heatmap of Protein Features", fontsize=16)
    plt.tight_layout()
    plt.show()
