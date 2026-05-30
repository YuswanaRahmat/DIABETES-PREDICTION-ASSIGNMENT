import matplotlib.pyplot as plt
import seaborn as sns


def plot_histograms(df, bins=30, figsize=(15, 12)):
    numeric_columns = df.select_dtypes(include=["number"]).columns
    df[numeric_columns].hist(bins=bins, figsize=figsize, edgecolor="white")
    plt.suptitle("Dataset Histograms")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    correlation = df.select_dtypes(include=["number"]).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, cmap="RdYlGn", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()


def plot_status_distribution(df, status_column="Diabetes_Status"):
    counts = df[status_column].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140)
    plt.title("Diabetes Status Distribution")
    plt.axis("equal")
    plt.show()
