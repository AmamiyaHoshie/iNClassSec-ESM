import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (matthews_corrcoef, confusion_matrix,
                             recall_score, precision_score, accuracy_score,
                             f1_score, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

feature_folders = ['AAC', 'CKSAAP', 'CTDC', 'CTDD', 'CTDT', 'CTriad',
                   'DPC', 'GAAC', 'GDPC', 'Moran', 'pse-pssm']

print("Loading training features for XGBoost...")
positive_data_xgb = pd.concat(
    [pd.read_csv(f'features/{folder}/PeNGaRoo_train_P.txt', header=None)
     for folder in feature_folders],
    axis=1
)
negative_data_xgb = pd.concat(
    [pd.read_csv(f'features/{folder}/PeNGaRoo_train_N.txt', header=None)
     for folder in feature_folders],
    axis=1
)
train_data_xgb = pd.concat([positive_data_xgb, negative_data_xgb], axis=0).reset_index(drop=True)
train_labels_xgb = np.concatenate([np.ones(positive_data_xgb.shape[0]),
                                   np.zeros(negative_data_xgb.shape[0])])
train_data_xgb = train_data_xgb.apply(pd.to_numeric, errors='coerce')
train_data_xgb = train_data_xgb.fillna(train_data_xgb.mean())

print("Loading test features for XGBoost...")
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

print("Splitting data for XGBoost...")
X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
    train_data_xgb, train_labels_xgb, test_size=0.2, random_state=SEED, stratify=train_labels_xgb)

print("Defining XGBoost pipeline...")
pipeline_xgb = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('featureSelection', SelectKBest(chi2, k=1200)),
    ('sampling', SMOTE(random_state=SEED)),
    ('classifier', XGBClassifier(
        alpha=10,
        base_score=0.5,
        booster='gbtree',
        colsample_bylevel=1,
        colsample_bynode=1,
        colsample_bytree=0.8,
        gamma=0,
        learning_rate=0.1,
        max_delta_step=0,
        max_depth=7,
        min_child_weight=1,
        n_estimators=100,
        n_jobs=1,
        objective='binary:logistic',
        random_state=0,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        subsample=0.6,
        verbosity=1
    ))
])

print("Setting up 5-fold CV for XGBoost...")
cv_xgb = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_preds_xgb_train = np.zeros(len(X_train_xgb))
oof_preds_xgb_val = np.zeros(len(X_val_xgb))
test_preds_xgb = np.zeros(len(test_data_xgb))

print("Performing XGBoost CV...")
for fold, (train_idx, val_idx) in enumerate(cv_xgb.split(X_train_xgb, y_train_xgb)):
    X_train_fold, X_val_fold = X_train_xgb.iloc[train_idx], X_train_xgb.iloc[val_idx]
    y_train_fold, y_val_fold = y_train_xgb[train_idx], y_train_xgb[val_idx]

    pipeline_xgb.fit(X_train_fold, y_train_fold)
    oof_preds_xgb_train[val_idx] = pipeline_xgb.predict_proba(X_val_fold)[:, 1]

oof_preds_xgb_val = pipeline_xgb.predict_proba(X_val_xgb)[:, 1]
test_preds_xgb = pipeline_xgb.predict_proba(test_data_xgb)[:, 1]

print("Loading embeddings for DNN...")
train_pos_path = "features/esm-3/PeNGaRoo_train_P.npy"
train_neg_path = "features/esm-3/PeNGaRoo_train_N.npy"
test_pos_path = "features/esm-3/PeNGaRoo_independent_test_P.npy"
test_neg_path = "features/esm-3/PeNGaRoo_independent_test_N.npy"

for path in [train_pos_path, train_neg_path, test_pos_path, test_neg_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found.")

positive_embeddings = np.load(train_pos_path, allow_pickle=True)
negative_embeddings = np.load(train_neg_path, allow_pickle=True)

if len(positive_embeddings) == 0 or len(negative_embeddings) == 0:
    raise ValueError("Positive or negative embeddings are empty.")

positive_labels = [1] * len(positive_embeddings)
negative_labels = [0] * len(negative_embeddings)

all_embeddings = np.vstack((positive_embeddings, negative_embeddings))
all_labels = np.array(positive_labels + negative_labels, dtype=np.float32)

test_positive_embeddings = np.load(test_pos_path, allow_pickle=True)
test_negative_embeddings = np.load(test_neg_path, allow_pickle=True)

test_positive_labels = [1] * len(test_positive_embeddings)
test_negative_labels = [0] * len(test_negative_embeddings)

test_embeddings = np.vstack((test_positive_embeddings, test_negative_embeddings))
test_labels_nn = np.array(test_positive_labels + test_negative_labels, dtype=np.float32)

print("Splitting data for DNN...")
X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
    all_embeddings, all_labels, test_size=0.2, random_state=SEED, stratify=all_labels)


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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.where(targets == 1, torch.sigmoid(inputs), 1 - torch.sigmoid(inputs))
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def get_criterion(class_weights, device, use_focal=False):
    if use_focal:
        return FocalLoss(alpha=0.25, gamma=2.0, reduction='mean').to(device)
    else:
        if len(class_weights) > 1:
            pos_weight = class_weights[1]
        else:
            pos_weight = 1.0
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)
        return criterion


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for embeddings, targets in tqdm(dataloader, desc="Training", leave=False):
        embeddings, targets = embeddings.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(embeddings)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * embeddings.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for embeddings, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            embeddings, targets = embeddings.to(device), targets.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * embeddings.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, np.array(all_targets), np.array(all_probs)


print("Performing DNN CV...")
NUM_FOLDS = 5
skf_nn = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

oof_preds_nn_train = np.zeros(len(X_train_nn))
oof_preds_nn_val = np.zeros(len(X_val_nn))
test_preds_nn = np.zeros(len(test_embeddings))

for fold, (train_idx, val_idx) in enumerate(skf_nn.split(X_train_nn, y_train_nn)):
    print(f"DNN Fold {fold + 1}...")
    X_fold_train, X_fold_val = X_train_nn[train_idx], X_train_nn[val_idx]
    y_fold_train, y_fold_val = y_train_nn[train_idx], y_train_nn[val_idx]

    print("Applying SMOTE...")
    smote = SMOTE(random_state=SEED)
    X_fold_train_res, y_fold_train_res = smote.fit_resample(X_fold_train, y_fold_train)

    class_weights_values = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_fold_train_res),
        y=y_fold_train_res
    )
    class_weights = torch.tensor(class_weights_values, dtype=torch.float32).to(device)

    train_dataset = EmbeddingDataset(X_fold_train_res, y_fold_train_res)
    val_dataset = EmbeddingDataset(X_fold_val, y_fold_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    embed_dim = X_fold_train_res.shape[1]
    model = DeepFeedForwardClassifier(input_dim=embed_dim, hidden_dim=128, dropout=0.5).to(device)

    criterion = get_criterion(class_weights, device, use_focal=False)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6)

    scaler = torch.cuda.amp.GradScaler()

    best_mcc = -1
    best_threshold = 0.5
    patience = 10
    trigger_times = 0

    for epoch in range(1, 51):
        print(f"Epoch {epoch}/50")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_targets, val_probs = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        thresholds = np.linspace(0.0, 1.0, num=100)
        mcc_scores = []
        for thresh in thresholds:
            preds_thresh = (val_probs >= thresh).astype(int)
            mcc = matthews_corrcoef(val_targets, preds_thresh)
            mcc_scores.append(mcc)
        mcc_scores = np.array(mcc_scores)
        best_thresh_idx = np.argmax(mcc_scores)
        current_best_threshold = thresholds[best_thresh_idx]
        current_best_mcc = mcc_scores[best_thresh_idx]
        print(f"Current Best Threshold: {current_best_threshold:.4f} (MCC={current_best_mcc:.4f})")

        if current_best_mcc > best_mcc:
            best_mcc = current_best_mcc
            best_threshold = current_best_threshold
            torch.save(model.state_dict(), f"best_model_fold_{fold + 1}.pth")
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered")
                break

    print(f"Fold {fold + 1} Best Threshold: {best_threshold:.4f} (MCC={best_mcc:.4f})")
    oof_preds_nn_train[val_idx] = val_probs

val_dataset_nn = EmbeddingDataset(X_val_nn, y_val_nn)
val_loader_nn = DataLoader(val_dataset_nn, batch_size=64, shuffle=False)
model.eval()
with torch.no_grad():
    val_probs = []
    for embeddings, targets in val_loader_nn:
        embeddings = embeddings.to(device)
        outputs = model(embeddings)
        val_probs.extend(torch.sigmoid(outputs).cpu().numpy())
oof_preds_nn_val = np.array(val_probs)

test_dataset_nn = EmbeddingDataset(test_embeddings, np.zeros(len(test_embeddings)))
test_loader_nn = DataLoader(test_dataset_nn, batch_size=64, shuffle=False)
model.eval()
with torch.no_grad():
    test_probs = []
    for embeddings, _ in tqdm(test_loader_nn, desc="Testing"):
        embeddings = embeddings.to(device)
        outputs = model(embeddings)
        test_probs.extend(torch.sigmoid(outputs).cpu().numpy())
test_preds_nn = np.array(test_probs)

print("Stacking...")
if len(oof_preds_xgb_train) != len(oof_preds_nn_train):
    raise ValueError("Mismatch in OOF predictions between XGBoost and DNN.")

if len(oof_preds_xgb_val) != len(oof_preds_nn_val):
    raise ValueError("Mismatch in validation predictions between XGBoost and DNN.")

X_stack = np.vstack((oof_preds_xgb_train, oof_preds_nn_train)).T
y_stack = y_train_xgb

X_stack_test = np.vstack((test_preds_xgb, test_preds_nn)).T

meta_model = LogisticRegression(random_state=SEED, max_iter=1000)

cv_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

accuracy_list = []
precision_list = []
recall_list = []
specificity_list = []
f1_list = []
roc_auc_list = []
average_precision_list = []
mcc_list = []

plt.figure(figsize=(8, 6))
print("Performing stacking CV...")
for fold, (train_idx, val_idx) in enumerate(cv_meta.split(X_stack, y_stack)):
    X_fold_train, X_fold_val = X_stack[train_idx], X_stack[val_idx]
    y_fold_train, y_fold_val = y_stack[train_idx], y_stack[val_idx]

    meta_model.fit(X_fold_train, y_fold_train)
    y_fold_val_probs = meta_model.predict_proba(X_fold_val)[:, 1]
    y_fold_val_preds = meta_model.predict(X_fold_val)

    fpr, tpr, _ = roc_curve(y_fold_val, y_fold_val_probs)
    roc_auc = roc_auc_score(y_fold_val, y_fold_val_probs)
    aucs.append(roc_auc)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {fold +1} (AUC = {roc_auc:.3f})')

    accuracy = accuracy_score(y_fold_val, y_fold_val_preds)
    precision = precision_score(y_fold_val, y_fold_val_preds, zero_division=0)
    recall = recall_score(y_fold_val, y_fold_val_preds, zero_division=0)
    try:
        tn, fp, fn, tp = confusion_matrix(y_fold_val, y_fold_val_preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    except ValueError:
        specificity = 0
    f1 = f1_score(y_fold_val, y_fold_val_preds, zero_division=0)
    average_precision = average_precision_score(y_fold_val, y_fold_val_probs)
    mcc = matthews_corrcoef(y_fold_val, y_fold_val_preds)

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    specificity_list.append(specificity)
    f1_list.append(f1)
    roc_auc_list.append(roc_auc)
    average_precision_list.append(average_precision)
    mcc_list.append(mcc)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = roc_auc_score(y_stack, meta_model.predict_proba(X_stack)[:, 1])
std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = {0:.3f} ± {1:.3f})'.format(mean_auc, std_auc),
         lw=2, alpha=0.8)

std_tpr = np.std(tprs, axis=0)
tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.2,
                 label='± 1 std. dev.')

plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Chance')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of the final model')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('stacked_model_roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print("Stacked Model CV Results (Mean ± Std):")
print(f"Accuracy: {np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}")
print(f"Precision: {np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}")
print(f"Recall: {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}")
print(f"Specificity: {np.mean(specificity_list):.4f} ± {np.std(specificity_list):.4f}")
print(f"F1-score: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
print(f"AUROC: {np.mean(roc_auc_list):.4f} ± {np.std(roc_auc_list):.4f}")
print(f"AUPRC: {np.mean(average_precision_list):.4f} ± {np.std(average_precision_list):.4f}")
print(f"MCC: {np.mean(mcc_list):.4f} ± {np.std(mcc_list):.4f}")

meta_test_preds = meta_model.predict(X_stack_test)
meta_test_probs = meta_model.predict_proba(X_stack_test)[:, 1]

accuracy = accuracy_score(test_labels_xgb, meta_test_preds)
precision = precision_score(test_labels_xgb, meta_test_preds, zero_division=0)
recall = recall_score(test_labels_xgb, meta_test_preds, zero_division=0)
try:
    tn, fp, fn, tp = confusion_matrix(test_labels_xgb, meta_test_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
except ValueError:
    specificity = 0
f1 = f1_score(test_labels_xgb, meta_test_preds, zero_division=0)
roc_auc = roc_auc_score(test_labels_xgb, meta_test_probs)
average_precision = average_precision_score(test_labels_xgb, meta_test_probs)
mcc = matthews_corrcoef(test_labels_xgb, meta_test_preds)

print("Stacked Model Test Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUROC: {roc_auc:.4f}")
print(f"AUPRC: {average_precision:.4f}")
print(f"MCC: {mcc:.4f}")
