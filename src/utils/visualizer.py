import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()