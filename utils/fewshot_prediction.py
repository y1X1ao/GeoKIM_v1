import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def predict_all_samples(Z_all, y, region_id, shot_options, classifier_type, num_trials,
                        save_csv=True, output_dir="outputs/predictions", include_all=True,
                        x_coords=None, y_coords=None, pretrained=True, classifier_name="MLP"):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    valid_idx = region_id >= 0
    pos_idx = np.where((y == 1) & valid_idx)[0]
    neg_idx = np.where((y == -1) & valid_idx)[0]
    unlabel_idx = np.where(y == 0)[0]

    for shot in shot_options:
        all_probs = np.zeros((len(y), num_trials))
        print(f"\n🔎 Predicting all samples using {shot}-shot classifier...")

        for trial in range(num_trials):
            np.random.seed(trial)
            try:
                pos_sample = np.random.choice(pos_idx, shot, replace=False)
                neg_sample = np.random.choice(neg_idx, shot, replace=False)
            except ValueError:
                print(f"⚠️ Not enough samples for {shot}-shot.")
                continue

            train_idx = np.concatenate([pos_sample, neg_sample])
            y_train = y[train_idx]
            Z_train = Z_all[train_idx]

            if classifier_type == "LogReg":
                clf = LogisticRegression(max_iter=3000)
            elif classifier_type == "MLP":
                clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=3000)
            else:
                raise ValueError("Unsupported classifier")

            clf.fit(Z_train, y_train)

            if include_all:
                Z_pred = Z_all
            else:
                Z_pred = Z_all[unlabel_idx]

            probs = clf.predict_proba(Z_pred)[:, 1]

            if include_all:
                all_probs[:, trial] = probs
            else:
                all_probs[unlabel_idx, trial] = probs

        prob_mean = np.mean(all_probs, axis=1)

        df = pd.DataFrame({
            "Index": np.arange(len(y)),
            "Label": y,
            "Pred_Prob": prob_mean,
            "Shot": shot,
        })

        if x_coords is not None and y_coords is not None:
            df["X"] = x_coords
            df["Y"] = y_coords

        results.append(df)

        if save_csv:
            pretrain_tag = "pretrained" if pretrained else "random"
            fname = f"{output_dir}/pred_{pretrain_tag}_{classifier_name}_{shot}shot.csv"
            df.to_csv(fname, index=False)
            print(f"✅ Saved predictions to {fname}")

    return results
