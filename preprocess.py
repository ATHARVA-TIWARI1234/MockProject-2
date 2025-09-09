import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument  # COMMENTED OUT

# -----------------------------
# NLTK Setup
# -----------------------------
def _ensure_nltk():
    for pkg in ["stopwords", "punkt", "wordnet"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
_ensure_nltk()

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
PUNCT_TABLE = str.maketrans("", "", string.punctuation.replace("+", "").replace("#", ""))

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "i'm": "i am",
    "'re": " are",
    "'s": " is",
    "'ll": " will",
    "'d": " would",
    "'ve": " have"
}

COMMON_SKILLS = [
    "machine learning", "deep learning", "data science", "natural language processing",
    "computer vision", "react native", "front end", "back end", "object oriented programming"
]

# -----------------------------
# Text Preprocessing
# -----------------------------
def expand_contractions(text: str) -> str:
    for contraction, expanded in CONTRACTIONS.items():
        text = re.sub(rf"\b{contraction}\b", expanded, text)
    return text

def merge_ngrams(text: str, ngrams=COMMON_SKILLS) -> str:
    for ng in ngrams:
        text = re.sub(rf"\b{ng}\b", ng.replace(" ", "_"), text)
    return text

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = expand_contractions(text)
    text = merge_ngrams(text)
    text = re.sub(r"http\S+|www\S+|[\w\.-]+@[\w\.-]+|\+?\d[\d\s-]{8,}\d", " ", text)
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    return text.split()

def lemmatize_tokens(tokens):
    return [LEMMATIZER.lemmatize(tok) for tok in tokens]

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def normalize(text: str) -> str:
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = lemmatize_tokens(tokens)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)

# -----------------------------
# Load data
# -----------------------------
DATA_PATH = "/Users/athtiwar/Desktop/DAY1_PYTHON/resume_match_project_korean/data/raw/translated_output_file.csv"
df = pd.read_csv(DATA_PATH)

# Normalize resume and JD columns
df['resume_norm'] = df['resume_en'].apply(normalize)
df['jd_norm'] = df['jd_en'].apply(normalize)

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_resume = vectorizer.fit_transform(df['resume_norm'])
tfidf_jd = vectorizer.transform(df['jd_norm'])

# -----------------------------
# Doc2Vec Embeddings (COMMENTED OUT)
# -----------------------------
# def train_doc2vec(text_series, vector_size=100, epochs=40):
#     tagged_docs = [TaggedDocument(words=doc.split(), tags=[str(i)]) 
#                    for i, doc in enumerate(text_series)]
#     model = Doc2Vec(vector_size=vector_size, min_count=2, workers=4, epochs=epochs)
#     model.build_vocab(tagged_docs)
#     model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
#     # Generate vectors
#     vectors = [model.infer_vector(doc.words) for doc in tagged_docs]
#     return model, vectors

# doc2vec_resume_model, doc2vec_resume_vectors = train_doc2vec(df['resume_norm'])
# doc2vec_jd_model, doc2vec_jd_vectors = train_doc2vec(df['jd_norm'])

# -----------------------------
# Save processed data & features
# -----------------------------
DATA_FEATURES_DIR = "/Users/athtiwar/Desktop/DAY1_PYTHON/resume_match_project_korean/data/features/"
ARTIFACTS_FEATURES_DIR = "/Users/athtiwar/Desktop/DAY1_PYTHON/resume_match_project_korean/artifacts/features/"

os.makedirs(DATA_FEATURES_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_FEATURES_DIR, exist_ok=True)

# Save processed dataframe
df.to_csv(os.path.join(DATA_FEATURES_DIR, "processed_data.csv"), index=False)

# Save TF-IDF matrices + vectorizer
with open(os.path.join(ARTIFACTS_FEATURES_DIR, "tfidf_resume.pkl"), "wb") as f:
    pickle.dump(tfidf_resume, f)

with open(os.path.join(ARTIFACTS_FEATURES_DIR, "tfidf_jd.pkl"), "wb") as f:
    pickle.dump(tfidf_jd, f)

with open(os.path.join(ARTIFACTS_FEATURES_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

# -----------------------------
# Doc2Vec saving (COMMENTED OUT)
# -----------------------------
# doc2vec_resume_model.save(os.path.join(ARTIFACTS_FEATURES_DIR, "doc2vec_resume.model"))
# doc2vec_jd_model.save(os.path.join(ARTIFACTS_FEATURES_DIR, "doc2vec_jd.model"))
# with open(os.path.join(ARTIFACTS_FEATURES_DIR, "doc2vec_resume_vectors.pkl"), "wb") as f:
#     pickle.dump(doc2vec_resume_vectors, f)
# with open(os.path.join(ARTIFACTS_FEATURES_DIR, "doc2vec_jd_vectors.pkl"), "wb") as f:
#     pickle.dump(doc2vec_jd_vectors, f)

print("Preprocessing complete!")
print("Processed CSV saved in:", DATA_FEATURES_DIR)
print("TF-IDF features saved in:", ARTIFACTS_FEATURES_DIR)
