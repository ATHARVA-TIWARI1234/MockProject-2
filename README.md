# Resume–Job Description Matching Project

## 📌 Project Overview

This project aims to build an **AI-powered resume-job matching system** that evaluates how well a candidate's resume matches a given job description (JD). The system leverages **TF-IDF**, **transformer-based embeddings**, **feature engineering**, and multiple ML/DL models to achieve high accuracy.

---

## 📂 Dataset & Preprocessing

### **1. Preprocessing (preprocess.py)**

* Lowercasing, punctuation removal, and normalization.
* Tokenization and stopword removal.
* Lemmatization for consistent word forms.
* Handles missing values and inconsistent formats.
* Outputs a **clean CSV**: `processed_data.csv`.

**Output Columns:**

* `resume_norm` → Preprocessed resume text.
* `jd_norm` → Preprocessed job description text.
* `label` → Ground truth (1 = Match, 0 = No match).

---

## 🧩 Feature Engineering (feature\_engineering.py)

Feature engineering enhances the dataset using **semantic similarity** and **keyword-based matching**.

### **1. TF-IDF Vectorization**

* Uses `tfidf_resume.pkl` and `tfidf_jd.pkl`.
* Generates TF-IDF vectors for resumes & job descriptions.
* Computes **cosine similarity** between resume and JD vectors.

```python
score = cosine_similarity(tfidf_resume[i], tfidf_jd[i])[0][0]
df['cosine_sim_tfidf'] = cosine_sim_scores
```

### **2. Job Title Match**

Checks if any job title keywords from JD appear in the resume.

```python
def job_title_match(resume, jd):
    resume_words = set(resume.lower().split())
    jd_words = set(jd.lower().split())
    return 1 if len(resume_words & jd_words) > 0 else 0

df['job_title_match'] = df.apply(lambda x: job_title_match(x['resume_norm'], x['jd_norm']), axis=1)
```

### **3. Keyword Overlap**

Measures the ratio of overlapping words between JD and resume.

```python
def keyword_overlap(resume, jd):
    resume_set = set(resume.split())
    jd_set = set(jd.split())
    return len(resume_set & jd_set) / (len(jd_set) + 1e-6)

df['keyword_overlap'] = df.apply(lambda x: keyword_overlap(x['resume_norm'], x['jd_norm']), axis=1)
```

### **Final Output**

* Saves `processed_data_with_features.csv`.
* New Features:

  * `cosine_sim_tfidf`
  * `job_title_match`
  * `keyword_overlap`

---

## 🤖 Models & Performance (Rank-wise)

We experimented with multiple ML/DL models.

| Rank | Model                                        | Accuracy   | Precision | Recall | F1-score   |
| ---- | -------------------------------------------- | ---------- | --------- | ------ | ---------- |
| 🥇 1 | **SBERT + ANN + Extra Features (5 epochs)**  | **95.11%** | 91.69%    | 99.20% | **95.30%** |
| 🥈 2 | **BGE-small + Transformer + Extra Features** | 94.88%     | 91.31%    | 99.20% | 95.09%     |
| 🥉 3 | **SBERT + ANN + Extra Features (2 epochs)**  | 93.75%     | 89.79%    | 98.72% | 94.04%     |
| 4    | **BGE-base + Transformer + Extra Features**  | 93.59%     | 89.78%    | 98.38% | 93.88%     |
| 5    | **SBERT + Transformer + Extra Features**     | 92.57%     | 88.29%    | 98.17% | 92.97%     |
| 6    | **PyTorch ANN**                              | 90.30%     | 89.37%    | 91.49% | 90.42%     |
| 7    | **SVM**                                      | 90.20%     | 90.97%    | 89.25% | 90.11%     |
| 8    | **Logistic Regression**                      | 89.97%     | 91.52%    | 88.11% | 89.78%     |

**Best Model → SBERT + ANN + Extra Features (5 epochs)** ✅

---

## 📊 TF-IDF Usage

The following pickle files are used:

* `tfidf_resume.pkl` → TF-IDF vectors for resumes.
* `tfidf_jd.pkl` → TF-IDF vectors for job descriptions.
* `tfidf_vectorizer.pkl` → Fitted vectorizer for inference.

These are essential for calculating `cosine_sim_tfidf`, a key feature for semantic similarity.

---

## 🚀 How to Run the Project

### **1. Clone the Repository**

```bash
git clone <repo_url>
cd resume_match_project
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Preprocess Data**

```bash
python src/data/preprocess.py
```

### **4. Feature Engineering**

```bash
python src/features/feature_engineering.py
```

### **5. Train Models**

```bash
python src/models/train_model.py
```

### **6. Evaluate Models**

```bash
python src/models/evaluate.py
```

---

## 📌 Key Insights

* Adding **extra engineered features** (TF-IDF + job title + keyword overlap) significantly boosts performance.
* **SBERT + ANN + Extra Features** achieves the highest accuracy.
* BGE-small + Transformer is the second-best alternative with lower inference time.

---

## 📎 Next Steps

* Integrate a **FastAPI** endpoint for real-time resume matching.
* Deploy the model using **Docker**.
* Add support for multilingual resumes.

---

## 🏆 Final Recommendation

Use **SBERT + ANN + Extra Features (5 epochs)** as the production model.
