import csv
import os


def save_result(feature, model, train_acc, val_acc, file_path="results.csv"):
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        # 헤더
        if not file_exists:
            writer.writerow(["feature", "model", "train_acc", "val_acc"])

        writer.writerow([feature, model, train_acc, val_acc])