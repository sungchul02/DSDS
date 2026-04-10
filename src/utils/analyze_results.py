import pandas as pd


def show_top_results(csv_path="results.csv"):
    df = pd.read_csv(csv_path)

    df = df.sort_values(by="val_acc", ascending=False)

    print("\n🔥 TOP 5 RESULTS")
    print(df.head(5))