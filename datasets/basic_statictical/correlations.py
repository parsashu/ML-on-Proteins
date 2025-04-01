import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr


def pearson_correlation_heatmap(file_path):
    """Plot Pearson correlation heatmap for specified columns in the protein dataset."""
    df = pd.read_csv(file_path, sep="\t")

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
    print("Number of samples used in correlation calculation:", len(df_clean))

    corr_matrix = df_clean.corr(method="pearson")

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        annot=True,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        fmt=".2f",
    )

    plt.title("Pearson Correlation Heatmap of Protein Features", fontsize=16)
    plt.tight_layout()
    plt.show()
    return corr_matrix


def correlation_with_p_values(file_path):
    """Calculate correlation coefficients with p-values for statistical significance"""
    df = pd.read_csv(file_path, sep="\t")

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

    pearson_corr = pd.DataFrame(index=columns_to_plot, columns=columns_to_plot)
    pearson_p = pd.DataFrame(index=columns_to_plot, columns=columns_to_plot)

    for i, col1 in enumerate(columns_to_plot):
        for j, col2 in enumerate(columns_to_plot):
            if not df_clean[col1].empty and not df_clean[col2].empty:
                corr, p_val = pearsonr(df_clean[col1], df_clean[col2])
                pearson_corr.loc[col1, col2] = corr
                pearson_p.loc[col1, col2] = p_val

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        pearson_corr.astype(float),
        annot=True,
        cmap=sns.diverging_palette(230, 20, as_cmap=True),
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        fmt=".2f",
    )
    plt.title("Pearson Correlation Coefficients", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12, 10))
    
    pearson_p_filled = pearson_p.fillna(1.0)
    log_p_values = -np.log10(pearson_p_filled)
    log_p_values = log_p_values.astype(float)
    formatted_p_values = pearson_p.map(lambda x: f"{x:.3f}" if not pd.isna(x) else "NaN")
    
    sns.heatmap(
        log_p_values,
        annot=formatted_p_values,
        cmap="YlOrRd",
        square=True,
        linewidths=0.5,
        fmt="",
    )
    plt.title(
        "Statistical Significance (-log10 p-value)\nHigher values = more significant",
        fontsize=16,
    )
    plt.tight_layout()
    plt.show()
    return pearson_corr, pearson_p


def feature_correlation_analysis(file_path):
    """Conduct comprehensive correlation analysis on protein features"""
    pearson_matrix = pearson_correlation_heatmap(file_path)

    if pearson_matrix is None:
        return None

    corr_values, p_values = correlation_with_p_values(file_path)

    if corr_values is None or p_values is None:
        return None

    return {
        "pearson_matrix": pearson_matrix,
        "correlation_values": corr_values,
        "p_values": p_values,
    }


file_path = "datasets/protein_dataset.tsv"
results = feature_correlation_analysis(file_path)
