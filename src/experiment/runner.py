import os
import random
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.feature.feature_factory import get_feature
from src.model.model_factory import get_model
from src.dataset.dataset import AudioDataset
from src.train.trainer import train
from src.experiment.tracker import save_result

from src.evaluation.evaluator import evaluate
from src.utils.visualizer import plot_confusion_matrix


BASE_DIR = "data/audio"

CLASS_MAP = {
    "yes_drone": 1,
    "unknown": 0
}


# =========================
# 데이터 로드
# =========================
def load_data(feature_name, max_per_class=30):
    features = []
    labels = []

    feature_fn = get_feature(feature_name)

    for class_name, label in CLASS_MAP.items():
        folder_path = os.path.join(BASE_DIR, class_name)

        files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        files = random.sample(files, min(len(files), max_per_class))

        print(f"{class_name}: {len(files)}개 사용")

        for file in files:
            path = os.path.join(folder_path, file)

            try:
                y, sr = librosa.load(path, sr=22050)

                target_len = 22050 * 2
                if len(y) < target_len:
                    y = np.pad(y, (0, target_len - len(y)))
                else:
                    y = y[:target_len]

                if random.random() < 0.5:
                    y = y + np.random.randn(len(y)) * 0.005

                feat = feature_fn(y, sr)

                target = 128
                H, W = feat.shape[1], feat.shape[2]

                if H < target or W < target:
                    pad_h = max(0, target - H)
                    pad_w = max(0, target - W)
                    feat = np.pad(feat, ((0, 0), (0, pad_h), (0, pad_w)))

                feat = feat[:, :target, :target]

                features.append(feat)
                labels.append(label)

            except Exception as e:
                print(f"에러 발생: {path} → {e}")

    print(f"\n총 데이터 수: {len(features)}")

    return features, labels


# =========================
# 🔥 실험 함수 (이게 없어서 에러난 것)
# =========================
def run_experiment():
    features_list = ["mfcc", "logmel", "chroma"]
    models = ["cnn", "lstm"]

    lrs = [0.0001, 0.001]
    batch_sizes = [4]
    epochs_list = [5]

    for feature_name in features_list:
        features, labels = load_data(feature_name)

        X_train, X_val, y_train, y_val = train_test_split(
            features,
            labels,
            test_size=0.2,
            stratify=labels,
            random_state=42
        )

        for model_name in models:
            for lr in lrs:
                for batch in batch_sizes:
                    for epochs in epochs_list:

                        print(f"\n🔥 {feature_name} | {model_name} | lr={lr}")

                        train_loader = DataLoader(AudioDataset(X_train, y_train), batch_size=batch, shuffle=True)
                        val_loader = DataLoader(AudioDataset(X_val, y_val), batch_size=batch)

                        model = get_model(model_name)

                        train_acc, val_acc = train(
                            model,
                            train_loader,
                            val_loader,
                            epochs=epochs,
                            lr=lr
                        )

                        save_result(
                            feature_name,
                            f"{model_name}_lr{lr}",
                            train_acc,
                            val_acc
                        )

                        cm = evaluate(model, val_loader)
                        plot_confusion_matrix(cm)