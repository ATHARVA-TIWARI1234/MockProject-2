# -----------------------------
# Install sentence-transformers (if needed)
# -----------------------------
# !pip install -q sentence-transformers  # run once in notebook if needed

# -----------------------------
# Imports
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
import pickle
import os

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "/Users/athtiwar/Desktop/DAY1_PYTHON/resume_match_project_korean/data/features/processed_data.csv"
MODEL_DIR = "/Users/athtiwar/Desktop/DAY1_PYTHON/resume_match_project_korean/artifacts/"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Set random seeds for reproducibility
# -----------------------------
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------
# Load processed data
# -----------------------------
df = pd.read_csv(DATA_PATH, engine='python')

# -----------------------------
# Extra features
# -----------------------------
def job_title_match(resume, jd):
    return 1 if len(set(resume.lower().split()) & set(jd.lower().split())) > 0 else 0

def keyword_overlap(resume, jd):
    return len(set(resume.split()) & set(jd.split())) / (len(set(jd.split())) + 1e-6)

df['job_title_match'] = df.apply(lambda x: job_title_match(x['resume_norm'], x['jd_norm']), axis=1)
df['keyword_overlap'] = df.apply(lambda x: keyword_overlap(x['resume_norm'], x['jd_norm']), axis=1)

# -----------------------------
# SBERT embeddings
# -----------------------------
print("⚡ Generating SBERT embeddings...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

resume_embeddings = sbert_model.encode(
    df['resume_norm'].tolist(), convert_to_numpy=True, batch_size=16, show_progress_bar=True
)
jd_embeddings = sbert_model.encode(
    df['jd_norm'].tolist(), convert_to_numpy=True, batch_size=16, show_progress_bar=True
)

# -----------------------------
# Concatenate embeddings + extra features
# -----------------------------
extra_features = df[['job_title_match', 'keyword_overlap']].values.astype(np.float32)
X = np.hstack([resume_embeddings, jd_embeddings, extra_features]).astype(np.float32)
y = df['label'].values.astype(np.float32).reshape(-1, 1)

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

with open(os.path.join(MODEL_DIR, "scaler_sbert_extra.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# -----------------------------
# Enable Apple GPU (MPS)
# -----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple GPU via MPS")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available. Using CPU instead.")

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # slightly larger batch size for GPU

# -----------------------------
# Lightweight Transformer Model
# -----------------------------
class LightweightTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=128, nhead=4, num_layers=1, dim_feedforward=128, dropout=0.2):
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
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)  # Add seq_len dimension
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        return self.classifier(x)

# Initialize model
transformer_model = LightweightTransformer(X_train_scaled.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)

# -----------------------------
# Training Loop (MPS Optimized)
# -----------------------------
epochs = 5
print("\n🚀 Training started...")
for epoch in range(epochs):
    transformer_model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = transformer_model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")

# -----------------------------
# Evaluation
# -----------------------------
transformer_model.eval()
with torch.no_grad():
    y_pred = transformer_model(X_test_tensor)
    y_pred_label = (y_pred.cpu().numpy() > 0.5).astype(int)  # ✅ Fixed MPS .numpy() issue

print("\n✅ SBERT + Lightweight Transformer + Extra Features (MPS):")
print("Accuracy:", accuracy_score(y_test, y_pred_label))
print("Precision:", precision_score(y_test, y_pred_label))
print("Recall:", recall_score(y_test, y_pred_label))
print("F1-score:", f1_score(y_test, y_pred_label))

# -----------------------------
# Save Model
# -----------------------------
torch.save(transformer_model.state_dict(), os.path.join(MODEL_DIR, "model_sbert_transformer_extra_mps.pt"))
print("\n✅ Model and scaler saved in", MODEL_DIR)
