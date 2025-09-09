import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

# -----------------------------
# Paths
# -----------------------------
DATA_FEATURES_DIR = "/Users/athtiwar/Desktop/DAY1_PYTHON/resume_match_project_korean/data/features/"
ARTIFACTS_FEATURES_DIR = "/Users/athtiwar/Desktop/DAY1_PYTHON/resume_match_project_korean/artifacts/features/"

# -----------------------------
# Load processed data
# -----------------------------
df = pd.read_csv(os.path.join(DATA_FEATURES_DIR, "processed_data.csv"))

# -----------------------------
# Load TF-IDF matrices
# -----------------------------
with open(os.path.join(ARTIFACTS_FEATURES_DIR, "tfidf_resume.pkl"), "rb") as f:
    tfidf_resume = pickle.load(f)

with open(os.path.join(ARTIFACTS_FEATURES_DIR, "tfidf_jd.pkl"), "rb") as f:
    tfidf_jd = pickle.load(f)

# -----------------------------
# Cosine similarity (TF-IDF)
# -----------------------------
cosine_sim_scores = []
for i in range(tfidf_resume.shape[0]):
    score = cosine_similarity(tfidf_resume[i], tfidf_jd[i])[0][0]
    cosine_sim_scores.append(score)

df['cosine_sim_tfidf'] = cosine_sim_scores

# -----------------------------
# Job title match (simple check if job title appears in resume)
# -----------------------------
def job_title_match(resume, jd):
    resume_words = set(resume.lower().split())
    jd_words = set(jd.lower().split())
    common = resume_words & jd_words
    return 1 if len(common) > 0 else 0

df['job_title_match'] = df.apply(lambda x: job_title_match(x['resume_norm'], x['jd_norm']), axis=1)

# -----------------------------
# Keyword overlap
# -----------------------------
def keyword_overlap(resume, jd):
    resume_set = set(resume.split())
    jd_set = set(jd.split())
    return len(resume_set & jd_set) / (len(jd_set) + 1e-6)

df['keyword_overlap'] = df.apply(lambda x: keyword_overlap(x['resume_norm'], x['jd_norm']), axis=1)

# -----------------------------
# Save updated dataframe
# -----------------------------
df.to_csv(os.path.join(DATA_FEATURES_DIR, "processed_data_with_features.csv"), index=False)

print("Feature engineering complete!")
print("New features added: 'cosine_sim_tfidf', 'job_title_match', 'keyword_overlap'")
print("Saved updated CSV at:", os.path.join(DATA_FEATURES_DIR, "processed_data_with_features.csv"))
