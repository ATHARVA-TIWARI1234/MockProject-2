# src/models/train_transformer2.py
# -------------------------------------------------------
# Train Resume-Job Matching Model using BGE-base + Transformer
# Optimized for Apple GPU (MPS) with Matching Scores + Labels
# -------------------------------------------------------

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
DATA_PATH = "/Users/athtiwar/Desktop/DAY1_PYTHON/resume_match_project_korean/data/features/processed_data.csv"
MODEL_DIR = "/Users/athtiwar/Desktop/DAY1_PYTHON/resume_match_project_korean/artifacts/"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------
# Reproducibility
# -------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

# -------------------------------------------------------
# Load Processed Data
# -------------------------------------------------------
df = pd.read_csv(DATA_PATH, engine="python")

# -------------------------------------------------------
# Extra Features
# -------------------------------------------------------
def job_title_match(resume, jd):
    return 1 if len(set(resume.lower().split()) & set(jd.lower().split())) > 0 else 0

def keyword_overlap(resume, jd):
    return len(set(resume.split()) & set(jd.split())) / (len(set(jd.split())) + 1e-6)

# Add extra features if missing
if "job_title_match" not in df.columns:
    df["job_title_match"] = df.apply(lambda x: job_title_match(x["resume_norm"], x["jd_norm"]), axis=1)
if "keyword_overlap" not in df.columns:
    df["keyword_overlap"] = df.apply(lambda x: keyword_overlap(x["resume_norm"], x["jd_norm"]), axis=1)

# -------------------------------------------------------
# Load BGE-base Model
# -------------------------------------------------------
print("⚡ Generating BGE-base embeddings...")
bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")  # safer on CPU

# Convert to lists
resume_texts = df["resume_norm"].astype(str).tolist()
jd_texts = df["jd_norm"].astype(str).tolist()

# Use small batch size to avoid memory issues
resume_embeddings = bge_model.encode(resume_texts, convert_to_numpy=True, batch_size=8, show_progress_bar=True)
jd_embeddings = bge_model.encode(jd_texts, convert_to_numpy=True, batch_size=8, show_progress_bar=True)

# -------------------------------------------------------
# Combine Embeddings + Extra Features
# -------------------------------------------------------
extra_features = df[["job_title_match", "keyword_overlap"]].values.astype(np.float32)
assert resume_embeddings.shape[0] == jd_embeddings.shape[0] == extra_features.shape[0] == df.shape[0]

X = np.hstack([resume_embeddings, jd_embeddings, extra_features]).astype(np.float32)
y = df["label"].values.astype(np.float32).reshape(-1, 1)

# -------------------------------------------------------
# Train/Test Split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------------
# Feature Scaling
# -------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open(os.path.join(MODEL_DIR, "scaler_bge.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# -------------------------------------------------------
# Enable Apple GPU (MPS) if available
# -------------------------------------------------------
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple GPU via MPS")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available. Using CPU instead.")

# -------------------------------------------------------
# Convert Data to Tensors
# -------------------------------------------------------
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# -------------------------------------------------------
# Lightweight Transformer Model
# -------------------------------------------------------
class LightweightTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=256, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.2):
        super(LightweightTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_hidden = nn.Sequential(
            nn.Linear(model_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.matching_score_layer = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        hidden = self.fc_hidden(x)
        matching_score = self.sigmoid(self.matching_score_layer(hidden))  # Continuous score
        predicted_label = (matching_score > 0.5).float()                 # Binary classification
        return matching_score, predicted_label

# Init model
transformer_model = LightweightTransformer(X_train_scaled.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.0005)

# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------
epochs = 5
print("\n🚀 Training started...")
for epoch in range(1, epochs + 1):
    transformer_model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        matching_scores, preds = transformer_model(xb)
        loss = criterion(matching_scores, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.6f}")

# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------
transformer_model.eval()
with torch.no_grad():
    matching_scores, preds = transformer_model(X_test_tensor)
    matching_scores_cpu = matching_scores.cpu().numpy()
    preds_cpu = preds.cpu().numpy()
    y_test_cpu = y_test_tensor.cpu().numpy()

print("\n✅ BGE-base + Transformer + Extra Features (MPS/CPU):")
print("Accuracy:", accuracy_score(y_test_cpu, preds_cpu))
print("Precision:", precision_score(y_test_cpu, preds_cpu))
print("Recall:", recall_score(y_test_cpu, preds_cpu))
print("F1-score:", f1_score(y_test_cpu, preds_cpu))

# -------------------------------------------------------
# Sample Predictions
# -------------------------------------------------------
print("\n🔹 Sample Predictions:")
for i in range(5):
    print(f"Matching Score: {matching_scores_cpu[i][0]:.4f} | Predicted Label: {int(preds_cpu[i][0])} | True Label: {int(y_test_cpu[i][0])}")

# -------------------------------------------------------
# Save Model & Scaler
# -------------------------------------------------------
torch.save(transformer_model.state_dict(), os.path.join(MODEL_DIR, "model_bge_transformer_mps.pt"))
print("\n✅ Model and scaler saved in", MODEL_DIR)
