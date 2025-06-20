import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns


os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)


df = pd.read_csv("data/geochem_data.csv")


coord_cols = ['X', 'Y']
label_col = 'label'

feature_cols = [col for col in df.columns if col not in coord_cols + [label_col]]
X_raw = df[feature_cols].values
y = df[label_col].values


X_log = np.log10(X_raw + 1)


mask = np.isin(y, [-1, 1])
X_labeled = X_log[mask]
y_labeled = y[mask]

corr_scores = []
for i in range(X_labeled.shape[1]):
    score, _ = pointbiserialr(X_labeled[:, i], y_labeled)
    corr_scores.append(abs(score))

corr_scores = np.nan_to_num(corr_scores, nan=0.0)
corr_probs = np.array(corr_scores)
corr_probs = corr_probs / corr_probs.sum()


output = {
    "X_scaled": X_log, 
    "labels": y,
    "feature_names": feature_cols,
    "correlation_probs": corr_probs,
    "X": df["X"].values,
    "Y": df["Y"].values
}
with open("preprocessed_geochem.pkl", "wb") as f:
    pickle.dump(output, f)

print("✅  preprocessed_geochem.pkl")

correlation_df = pd.DataFrame({
    'feature': feature_cols,
    'correlation_with_label': corr_scores,
    'sampling_prob': corr_probs
})
correlation_df.to_csv("outputs/feature_label_correlation.csv", index=False)
print("📄  outputs/feature_label_correlation.csv")
correlation_df_sorted = correlation_df.sort_values("correlation_with_label", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x="feature", y="correlation_with_label", data=correlation_df_sorted, palette="viridis")
plt.xticks(rotation=45)
plt.ylabel("Correlation with Label")
plt.title("Feature-wise Correlation with Mineralization Label (±1)")
plt.tight_layout()
plt.savefig("outputs/figures/feature_correlation.png")
plt.close()
print("📊outputs/figures/feature_correlation.png")
