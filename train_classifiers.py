import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# Paths
# -----------------------------
DATA_FEATURES_DIR = "/Users/athtiwar/Desktop/DAY1_PYTHON/resume_match_project_korean/data/features/"
ARTIFACTS_DIR = "/Users/athtiwar/Desktop/DAY1_PYTHON/resume_match_project_korean/artifacts/"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# -----------------------------
# Load processed data with features
# -----------------------------
df = pd.read_csv(os.path.join(DATA_FEATURES_DIR, "processed_data_with_features.csv"))

# -----------------------------
# Features & target
# -----------------------------
feature_cols = ['cosine_sim_tfidf', 'job_title_match', 'keyword_overlap']
X = df[feature_cols].values.astype(float)
y = df['label'].values.astype(float)

# -----------------------------
# Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Feature scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open(os.path.join(ARTIFACTS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# -----------------------------
# Logistic Regression
# -----------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1-score:", f1_score(y_test, y_pred_lr))

with open(os.path.join(ARTIFACTS_DIR, "model_lr.pkl"), "wb") as f:
    pickle.dump(lr_model, f)

# -----------------------------
# Support Vector Machine (SVM)
# -----------------------------
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

print("\nSVM:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm))
print("Recall:", recall_score(y_test, y_pred_svm))
print("F1-score:", f1_score(y_test, y_pred_svm))

with open(os.path.join(ARTIFACTS_DIR, "model_svm.pkl"), "wb") as f:
    pickle.dump(svm_model, f)

# -----------------------------
# PyTorch ANN (MPS on Mac)
# -----------------------------
# Use MPS if available (Apple GPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.reshape(-1,1), dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class ANNModel(nn.Module):
    def __init__(self, input_dim):
        super(ANNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

ann_model = ANNModel(X_train_scaled.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(ann_model.parameters(), lr=0.001)

# Training loop
epochs = 15
for epoch in range(epochs):
    ann_model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = ann_model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluation
ann_model.eval()
with torch.no_grad():
    y_pred_ann = ann_model(X_test_tensor)
    y_pred_label = (y_pred_ann.cpu().numpy() > 0.5).astype(int)

print("\nPyTorch ANN (MPS):")
print("Accuracy:", accuracy_score(y_test, y_pred_label))
print("Precision:", precision_score(y_test, y_pred_label))
print("Recall:", recall_score(y_test, y_pred_label))
print("F1-score:", f1_score(y_test, y_pred_label))

# Save PyTorch ANN model
torch.save(ann_model.state_dict(), os.path.join(ARTIFACTS_DIR, "model_ann.pt"))

print("\nTraining complete. All models and scaler saved in:", ARTIFACTS_DIR)
