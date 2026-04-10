import pandas as pd
import matplotlib.pyplot as plt


def plot_results(csv_path="results.csv"):
    df = pd.read_csv(csv_path)

    plt.figure()

    for feature in df["feature"].unique():
        subset = df[df["feature"] == feature]
        plt.plot(subset["val_acc"].values, label=feature)

    plt.legend()
    plt.title("Feature Comparison")
    plt.xlabel("Experiment")
    plt.ylabel("Accuracy")
    plt.show()