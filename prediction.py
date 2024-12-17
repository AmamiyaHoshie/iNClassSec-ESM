import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import os
from sklearn.metrics import (matthews_corrcoef, confusion_matrix,
                             recall_score, precision_score, accuracy_score,
                             f1_score, roc_auc_score, average_precision_score,
                             precision_recall_curve)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================
# Part 1: Load Models
# ============================

xgb_pipeline = joblib.load("xgb_pipeline.pkl")
meta_model = joblib.load("meta_model.pkl")

class DeepFeedForwardClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.5):
        super(DeepFeedForwardClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.relu3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.layer2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.layer3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.dropout3(out)

        out = self.fc(out)
        return out.squeeze(1)


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# ============================
# Part 2: Load Independent Test Data
# ============================
feature_folders = ['AAC', 'CKSAAP', 'CTDC', 'CTDD', 'CTDT', 'CTriad',
                   'DPC', 'GAAC', 'GDPC', 'Moran', 'pse-pssm']

positive_test_data_xgb = pd.concat(
    [pd.read_csv(f'features/{folder}/PeNGaRoo_independent_test_P.txt', header=None)
     for folder in feature_folders],
    axis=1
)

negative_test_data_xgb = pd.concat(
    [pd.read_csv(f'features/{folder}/PeNGaRoo_independent_test_N.txt', header=None)
     for folder in feature_folders],
    axis=1
)

test_data_xgb = pd.concat([positive_test_data_xgb, negative_test_data_xgb], axis=0).reset_index(drop=True)
test_labels_xgb = np.concatenate([np.ones(positive_test_data_xgb.shape[0]),
                                  np.zeros(negative_test_data_xgb.shape[0])])

test_data_xgb = test_data_xgb.apply(pd.to_numeric, errors='coerce')
test_data_xgb = test_data_xgb.fillna(test_data_xgb.mean())

test_pos_path = "features/esm-3/PeNGaRoo_independent_test_P.npy"
test_neg_path = "features/esm-3/PeNGaRoo_independent_test_N.npy"

if not (os.path.exists(test_pos_path) and os.path.exists(test_neg_path)):
    raise FileNotFoundError("Independent test embeddings do not exist.")

test_positive_embeddings = np.load(test_pos_path, allow_pickle=True)
test_negative_embeddings = np.load(test_neg_path, allow_pickle=True)

test_positive_labels = [1] * len(test_positive_embeddings)
test_negative_labels = [0] * len(test_negative_embeddings)

test_embeddings = np.vstack((test_positive_embeddings, test_negative_embeddings))
test_labels_nn = np.array(test_positive_labels + test_negative_labels, dtype=np.float32)

# ============================
# Part 3: Predict with Loaded Models
# ============================
test_preds_xgb = xgb_pipeline.predict_proba(test_data_xgb)[:, 1]

input_dim = test_embeddings.shape[1]
dnn_model = DeepFeedForwardClassifier(input_dim=input_dim, hidden_dim=128, dropout=0.5).to(device)
dnn_model.load_state_dict(torch.load("best_model.pth", map_location=device))
dnn_model.eval()

test_dataset_nn = EmbeddingDataset(test_embeddings, np.zeros(len(test_embeddings)))
test_loader_nn = DataLoader(test_dataset_nn, batch_size=64, shuffle=False)

test_preds_nn = []
with torch.no_grad():
    for embeddings, _ in tqdm(test_loader_nn, desc="Predicting DNN"):
        embeddings = embeddings.to(device)
        outputs = dnn_model(embeddings)
        test_preds_nn.extend(torch.sigmoid(outputs).cpu().numpy())
test_preds_nn = np.array(test_preds_nn)

X_stack_test = np.vstack((test_preds_xgb, test_preds_nn)).T
meta_test_probs = meta_model.predict_proba(X_stack_test)[:, 1]

# ============================
# Part 4: Evaluate & Save Results
# ============================
best_threshold = 0.15
meta_test_preds_best = (meta_test_probs >= best_threshold).astype(int)

accuracy_best = accuracy_score(test_labels_xgb, meta_test_preds_best)
precision_best = precision_score(test_labels_xgb, meta_test_preds_best, zero_division=0)
recall_best = recall_score(test_labels_xgb, meta_test_preds_best, zero_division=0)
f1_best = f1_score(test_labels_xgb, meta_test_preds_best, zero_division=0)
roc_auc_best = roc_auc_score(test_labels_xgb, meta_test_probs)
average_precision_best = average_precision_score(test_labels_xgb, meta_test_probs)
mcc_best = matthews_corrcoef(test_labels_xgb, meta_test_preds_best)
try:
    tn, fp, fn, tp = confusion_matrix(test_labels_xgb, meta_test_preds_best).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
except ValueError:
    specificity = 0

print("Independent Test Set Performance:")
print(f"Accuracy: {accuracy_best:.4f}")
print(f"Precision: {precision_best:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Recall: {recall_best:.4f}")
print(f"F1-score: {f1_best:.4f}")
print(f"AUROC: {roc_auc_best:.4f}")
print(f"AUPRC: {average_precision_best:.4f}")
print(f"MCC: {mcc_best:.4f}")

cm = confusion_matrix(test_labels_xgb, meta_test_preds_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Independent Test)')
plt.savefig('independent_confusion_matrix.png')
plt.show()

pd.DataFrame({
    "Pred_Probs": meta_test_probs,
    "Pred_Labels": meta_test_preds_best,
    "True_Labels": test_labels_xgb
}).to_csv("independent_test_predictions.csv", index=False)
