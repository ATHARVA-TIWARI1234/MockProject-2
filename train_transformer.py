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
# Reproducibility
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

# create features if not present
if 'job_title_match' not in df.columns:
    df['job_title_match'] = df.apply(lambda x: job_title_match(x['resume_norm'], x['jd_norm']), axis=1)
if 'keyword_overlap' not in df.columns:
    df['keyword_overlap'] = df.apply(lambda x: keyword_overlap(x['resume_norm'], x['jd_norm']), axis=1)

# -----------------------------
# SBERT embeddings (CPU)
# -----------------------------
print("⚡ Generating SBERT embeddings on CPU (safer for MPS setups)...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

resume_texts = df['resume_norm'].astype(str).tolist()
jd_texts = df['jd_norm'].astype(str).tolist()

resume_embeddings = sbert_model.encode(resume_texts, convert_to_numpy=True, batch_size=16, show_progress_bar=True)
jd_embeddings = sbert_model.encode(jd_texts, convert_to_numpy=True, batch_size=16, show_progress_bar=True)

# -----------------------------
# Build dataset: embeddings + extra features
# -----------------------------
extra_features = df[['job_title_match', 'keyword_overlap']].values.astype(np.float32)
assert resume_embeddings.shape[0] == jd_embeddings.shape[0] == extra_features.shape[0] == df.shape[0], "Mismatch lengths"

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
# Enable Apple GPU (MPS) if available
# -----------------------------
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple GPU via MPS")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available. Using CPU instead.")

# -----------------------------
# Convert data to PyTorch tensors on the selected device
# -----------------------------
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# -----------------------------
# Lightweight Transformer Model (Modified)
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
            nn.Linear(64, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_logits=False):
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)

        logits = self.classifier(x)
        probs = self.sigmoid(logits)

        if return_logits:
            return logits, probs
        return probs

# Init model
transformer_model = LightweightTransformer(X_train_scaled.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)

# -----------------------------
# Training Loop
# -----------------------------
epochs = 5
print("\n🚀 Training started...")
for epoch in range(1, epochs + 1):
    transformer_model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        _, probs = transformer_model(xb, return_logits=True)
        loss = criterion(probs, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.6f}")

# -----------------------------
# Evaluation
# -----------------------------
transformer_model.eval()
with torch.no_grad():
    logits, probs = transformer_model(X_test_tensor, return_logits=True)
    preds = (probs > 0.5).float()

# Move to CPU for metrics
logits_cpu = logits.cpu().numpy()
probs_cpu = probs.cpu().numpy()
y_test_cpu = y_test_tensor.cpu().numpy()
y_pred_label = preds.cpu().numpy()

print("\n✅ SBERT + Lightweight Transformer + Extra Features (MPS/CPU):")
print("Accuracy:", accuracy_score(y_test_cpu, y_pred_label))
print("Precision:", precision_score(y_test_cpu, y_pred_label))
print("Recall:", recall_score(y_test_cpu, y_pred_label))
print("F1-score:", f1_score(y_test_cpu, y_pred_label))

# -----------------------------
# Save matching scores + results
# -----------------------------
results_df = pd.DataFrame({
    "resume_text": df.iloc[y_test_tensor.cpu().numpy().reshape(-1)].get("resume_norm", ""),
    "jd_text": df.iloc[y_test_tensor.cpu().numpy().reshape(-1)].get("jd_norm", ""),
    "matching_score": logits_cpu.reshape(-1),
    "matching_probability": probs_cpu.reshape(-1),
    "final_prediction": y_pred_label.reshape(-1),
    "true_label": y_test_cpu.reshape(-1)
})

results_path = os.path.join(MODEL_DIR, "sbert_transformer_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\n📄 Results saved to: {results_path}")

# -----------------------------
# Save model & scaler
# -----------------------------
torch.save(transformer_model.state_dict(), os.path.join(MODEL_DIR, "model_sbert_transformer_extra_mps.pt"))
print("\n✅ Model and scaler saved in", MODEL_DIR)
