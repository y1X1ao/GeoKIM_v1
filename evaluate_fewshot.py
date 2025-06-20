# evaluate_fewshot.py
import os
import yaml
import torch
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from models.transformer_encoder import TabularTransformerEncoder
from utils.fewshot_prediction import predict_all_samples
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
import pandas as pd
os.makedirs("outputs/figures/curves/", exist_ok=True)

USE_PRETRAINED = True
CLASSIFIER_TYPE = "MLP"  # "LogReg", "MLP", "ProtoNet"
split_mode = "quadrant"    # "quadrant" or "kmeans"
num_trials = 20
shot_options = [1, 5 ,10 ,20]


with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
model_cfg = cfg["model"]
SEED = cfg["training"].get("seed", 42)


def set_seed(seed=42):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
set_seed(SEED)


with open("preprocessed_geochem.pkl", "rb") as f:
    data = pickle.load(f)

X_scaled = data["X_scaled"]
refined_path = "outputs/negatives/refined_labels.npy"
if os.path.exists(refined_path):
    y = np.load(refined_path)
    print("✅ Using refined_labels.npy for evaluation.")
else:
    y = data['labels']
    print("⚠️ Refined labels not found, fallback to original labels.")
# y = data['labels']
feature_names = data["feature_names"]
x_coords = data.get("X")
y_coords = data.get("Y")
n_features = X_scaled.shape[1]


if x_coords is None or y_coords is None:
    print("⚠️ No spatial information found.")
    USE_SPATIAL_SPLIT = False
else:
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    coords = np.stack([x_coords, y_coords], axis=1)

    if split_mode == "quadrant":
        x_mid, y_mid = np.median(x_coords), np.median(y_coords)
        region_id = (x_coords > x_mid).astype(int) + (y_coords > y_mid).astype(int) * 2
    elif split_mode == "kmeans":
        label_mask = np.isin(y, [-1, 1])
        coords_labeled = np.stack([x_coords[label_mask], y_coords[label_mask]], axis=1)
        region_labeled = KMeans(n_clusters=4, random_state=SEED).fit_predict(coords_labeled)

  
        region_id = np.full_like(y, fill_value=-1)
        region_id[label_mask] = region_labeled

    
        os.makedirs("outputs/figures/", exist_ok=True)
        plt.figure(figsize=(6, 6))
        plt.scatter(x_coords[label_mask], y_coords[label_mask], c=region_labeled, cmap="tab10", s=10)
        plt.title("KMeans Spatial Cluster (±1 samples only)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar(label="Cluster ID")
        plt.tight_layout()
        plt.savefig("outputs/figures/spatial_clusters_kmeans_labeled.png")
        plt.close()
        print("🗺️  KMeans cluster plot saved (labeled samples only).")


# ==========  Encoder ==========
encoder = TabularTransformerEncoder(
    num_tokens=n_features,
    embed_dim=model_cfg["hidden_dim"],
    depth=model_cfg["num_layers"],
    heads=model_cfg["num_heads"],
    pooling=model_cfg["pooling"]
)

if model_cfg["pooling"] == "flatten":
    z_dim = model_cfg["hidden_dim"] * n_features
elif model_cfg["pooling"] == "mean+max":
    z_dim = model_cfg["hidden_dim"] * 2
else:
    z_dim = model_cfg["hidden_dim"]

if USE_PRETRAINED:
    encoder.load_state_dict(torch.load("outputs/checkpoints/encoder.pt", map_location="cpu"))
    print("✅ Using pretrained Transformer encoder")
else:
    print("⚠️ Using randomly initialized encoder")

encoder.eval()
with torch.no_grad():
    Z_all = encoder(torch.tensor(X_scaled, dtype=torch.float32)).numpy()

# ========== t-SNE  ==========
label_mask = np.isin(y, [-1, 1])
Z_subset = Z_all[label_mask]
y_subset = y[label_mask]
Z_std = StandardScaler().fit_transform(Z_subset)

tsne = TSNE(n_components=2, random_state=SEED)
Z_tsne = tsne.fit_transform(Z_std)
plt.figure(figsize=(6, 5))
plt.scatter(Z_tsne[y_subset == -1, 0], Z_tsne[y_subset == -1, 1], label="Negative", alpha=0.6)
plt.scatter(Z_tsne[y_subset == 1, 0], Z_tsne[y_subset == 1, 1], label="Positive", alpha=0.6)
plt.title(f"t-SNE Embedding ({'Pretrained' if USE_PRETRAINED else 'Random Init'})")
plt.legend()
plt.tight_layout()
plt.savefig(f"outputs/figures/tsne_{'pretrained' if USE_PRETRAINED else 'baseline'}_{split_mode}.png")
plt.close()

print(f"📈 Silhouette: {silhouette_score(Z_std, y_subset):.4f}")
print(f"📈 DBI: {davies_bouldin_score(Z_std, y_subset):.4f}")

# ========== Few-shot==========
valid_idx = region_id >= 0  
pos_idx = np.where((y == 1) & valid_idx)[0]
neg_idx = np.where((y == -1) & valid_idx)[0]

print(f"\nFew-shot Evaluation ({num_trials} trials)")
print(f"Encoder: {'Pretrained' if USE_PRETRAINED else 'Random'} | Classifier: {CLASSIFIER_TYPE}")
print(f"Spatial Split Mode: {split_mode}\n")
curve_data = {}
for fold in range(4):
    train_regions = [r for r in range(4) if r != fold]
    pos_train = pos_idx[np.isin(region_id[pos_idx], train_regions)]
    neg_train = neg_idx[np.isin(region_id[neg_idx], train_regions)]
    pos_test = pos_idx[region_id[pos_idx] == fold]
    neg_test = neg_idx[region_id[neg_idx] == fold]

    print(f"\n🧪 Fold {fold}: Train Regions = {train_regions}, Test Region = {fold}")
    for num_shots in shot_options:
        accs, aucs, f1s = [], [], []
        all_y_true, all_y_prob = [], []

        for trial in range(num_trials):
            np.random.seed(trial)
            try:
                pos_sample = np.random.choice(pos_train, num_shots, replace=False)
                neg_sample = np.random.choice(neg_train, num_shots, replace=False)
            except ValueError:
                print(f"⚠️ Not enough samples for {num_shots}-shot in Fold {fold}")
                continue

            train_idx = np.concatenate([pos_sample, neg_sample])
            test_idx = np.concatenate([pos_test, neg_test])

            Z_train, y_train = Z_all[train_idx], y[train_idx]
            Z_test, y_test = Z_all[test_idx], y[test_idx]

            if CLASSIFIER_TYPE == "LogReg":
                clf = LogisticRegression(max_iter=3000)
            elif CLASSIFIER_TYPE == "MLP":
                clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=3000)
            elif CLASSIFIER_TYPE == "ProtoNet":
                z_pos = Z_train[y_train == 1].mean(axis=0)
                z_neg = Z_train[y_train == -1].mean(axis=0)
                d_pos = np.linalg.norm(Z_test - z_pos, axis=1)
                d_neg = np.linalg.norm(Z_test - z_neg, axis=1)
                y_prob = 1 / (1 + np.exp(d_pos - d_neg))
                y_pred = (y_prob >= 0.5).astype(int) * 2 - 1
                accs.append(accuracy_score(y_test, y_pred))
                aucs.append(roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan)
                f1s.append(f1_score(y_test, y_pred))
                continue
            else:
                raise ValueError("Unsupported classifier")

            clf.fit(Z_train, y_train)
            y_pred = clf.predict(Z_test)
            y_prob = clf.predict_proba(Z_test)[:, 1]

            accs.append(accuracy_score(y_test, y_pred))
            aucs.append(roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan)
            f1s.append(f1_score(y_test, y_pred))

            
            if trial == num_trials - 1:
                all_y_true.extend(y_test)
                all_y_prob.extend(y_prob)

        auc_str = f"{np.nanmean(aucs):.4f} ± {np.nanstd(aucs):.4f}" if not np.isnan(aucs).all() else "N/A"
        print(f"{num_shots}-shot\tAcc: {np.mean(accs):.4f} ± {np.std(accs):.4f}\t"
              f"AUC: {auc_str}\t"
              f"F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

      
        key = f"{num_shots}shot_fold{fold}"
        curve_data[key] = {
            "y_true": np.array(all_y_true),
            "y_prob": np.array(all_y_prob),
        }

# ========== ROC  ==========
for key, data in curve_data.items():
    y_true = (data["y_true"] == 1).astype(int)  # 转为 0-1
    y_prob = data["y_prob"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.3f}")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({key})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/figures/curves/roc_{key}.png")
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve ({key})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/figures/curves/pr_{key}.png")
    plt.close()

# ========== Exporting predictions ==========
print("\n🧩 Exporting full-area predictions over folds & trials...")

predict_all_samples(
    Z_all=Z_all,
    y=y,
    region_id=region_id,
    shot_options=shot_options,
    classifier_type=CLASSIFIER_TYPE,
    num_trials=num_trials,
    include_all=True,
    x_coords=x_coords,
    y_coords=y_coords,
    pretrained=USE_PRETRAINED,
    classifier_name=CLASSIFIER_TYPE
)



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

baseline_models = {
    "LogReg": LogisticRegression(max_iter=3000),
    "MLP": MLPClassifier(hidden_layer_sizes=(64,), max_iter=3000),
    "SVM": SVC(kernel="rbf", probability=True),
    "RF": RandomForestClassifier(n_estimators=100),
    "GBM": GradientBoostingClassifier(n_estimators=100),
}

print("\n📊 Baseline comparison using raw features:")
for name, clf in baseline_models.items():
    print(f"\n🔍 Classifier = {name}")
    for fold in range(4):
        train_regions = [r for r in range(4) if r != fold]
        pos_train = pos_idx[np.isin(region_id[pos_idx], train_regions)]
        neg_train = neg_idx[np.isin(region_id[neg_idx], train_regions)]
        pos_test = pos_idx[region_id[pos_idx] == fold]
        neg_test = neg_idx[region_id[neg_idx] == fold]

        for num_shots in shot_options:
            accs, aucs, f1s = [], [], []
            for trial in range(num_trials):
                np.random.seed(trial)
                try:
                    pos_sample = np.random.choice(pos_train, num_shots, replace=False)
                    neg_sample = np.random.choice(neg_train, num_shots, replace=False)
                except ValueError:
                    print(f"⚠️ Not enough samples for {num_shots}-shot in Fold {fold}")
                    continue

                train_idx = np.concatenate([pos_sample, neg_sample])
                test_idx = np.concatenate([pos_test, neg_test])

                X_train, y_train_baseline = X_scaled[train_idx], y[train_idx]
                X_test, y_test_baseline = X_scaled[test_idx], y[test_idx]

                clf.fit(X_train, y_train_baseline)
                y_pred = clf.predict(X_test)
                if hasattr(clf, "predict_proba"):
                    y_prob = clf.predict_proba(X_test)[:, 1]
                else:
                    y_prob = clf.decision_function(X_test)
                    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)

                accs.append(accuracy_score(y_test_baseline, y_pred))
                aucs.append(roc_auc_score(y_test_baseline, y_prob) if len(np.unique(y_test_baseline)) > 1 else np.nan)
                f1s.append(f1_score(y_test_baseline, y_pred))

            auc_str = f"{np.nanmean(aucs):.4f} ± {np.nanstd(aucs):.4f}" if not np.isnan(aucs).all() else "N/A"
            print(f"{num_shots}-shot Fold{fold}  Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}  "
                  f"AUC: {auc_str}  F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
            
from collections import defaultdict

print("\n📤 Exporting full-area average predictions over folds & trials for baseline models...")

for name, clf in baseline_models.items():
    for num_shots in shot_options:
        y_prob_accum = np.zeros(len(X_scaled))
        count_valid = 0

        for fold in range(4):
            train_regions = [r for r in range(4) if r != fold]
            pos_train = pos_idx[np.isin(region_id[pos_idx], train_regions)]
            neg_train = neg_idx[np.isin(region_id[neg_idx], train_regions)]

            for trial in range(num_trials):
                np.random.seed(trial)
                try:
                    pos_sample = np.random.choice(pos_train, num_shots, replace=False)
                    neg_sample = np.random.choice(neg_train, num_shots, replace=False)
                except ValueError:
                    continue

                train_idx = np.concatenate([pos_sample, neg_sample])
                X_train = X_scaled[train_idx]
                y_train_ = y[train_idx]

               
                clf_trial = pickle.loads(pickle.dumps(clf))
                clf_trial.fit(X_train, y_train_)

                
                if hasattr(clf_trial, "predict_proba"):
                    y_prob = clf_trial.predict_proba(X_scaled)[:, 1]
                else:
                    decision = clf_trial.decision_function(X_scaled)
                    y_prob = (decision - decision.min()) / (decision.max() - decision.min() + 1e-8)

                y_prob_accum += y_prob
                count_valid += 1

        if count_valid > 0:
            y_prob_avg = y_prob_accum / count_valid
            output_dir = f"outputs/predictions/baselines_avg/{name}"
            os.makedirs(output_dir, exist_ok=True)

            df_avg = pd.DataFrame({
                "X": x_coords,
                "Y": y_coords,
                "probability": y_prob_avg
            })
            filename = f"{name}_shot{num_shots}_avg.csv"
            df_avg.to_csv(os.path.join(output_dir, filename), index=False)
            print(f"✅ Saved: {filename}")
        else:
            print(f"⚠️ No valid trials for {name} shot={num_shots}")
