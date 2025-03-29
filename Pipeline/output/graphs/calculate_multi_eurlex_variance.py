import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# Load predictions and labels from file
with open("predicted_5.json", "r") as f:
    data = json.load(f)

predicted_labels = data["predicted"]
true_labels = data["true"]

# Ensure no None in predictions
predicted_labels = [lbl if lbl is not None else [] for lbl in predicted_labels]

# Build label space from both true and predicted
all_labels = set(label for sublist in predicted_labels + true_labels for label in sublist)
label_options = sorted(all_labels)

# Binarize labels
mlb = MultiLabelBinarizer(classes=label_options)
binary_true = mlb.fit_transform(true_labels)
binary_pred = mlb.transform(predicted_labels)

# --- Per-sample F1 Score ---
per_sample_f1 = []
for t, p in zip(binary_true, binary_pred):
    tp = np.sum(np.logical_and(t, p))
    fp = np.sum(np.logical_and(np.logical_not(t), p))
    fn = np.sum(np.logical_and(t, np.logical_not(p)))
    denom = 2 * tp + fp + fn
    score = (2 * tp) / denom if denom != 0 else 0.0
    per_sample_f1.append(score)

f1_variance = np.var(per_sample_f1)

# --- Per-sample R-Precision ---
r_precisions = []
for true, pred in zip(true_labels, predicted_labels):
    if not true:
        continue  # Skip if no true labels
    k = len(true)
    top_k_pred = pred[:k]
    correct = len(set(top_k_pred) & set(true))
    r_precision = correct / k
    r_precisions.append(r_precision)

r_precision_variance = np.var(r_precisions) if r_precisions else 0.0

# --- Output ---
print(f"F1 Variance: {f1_variance:.4f}")
print(f"R-Precision Variance: {r_precision_variance:.4f}")
