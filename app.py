

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re, string, io, requests, threading, webbrowser
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────────
models_ready  = False
models        = {}        # {name: trained model}
vectorizer    = None
model_metrics = {}
prediction_history = []

# ── Dataset URLs ──────────────────────────────────────────────────
# SMS Spam Collection — well-known public dataset
DATASET_URLS = [
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv",
]

# ── Text Preprocessing ────────────────────────────────────────────
def preprocess(text: str) -> str:
    """
    Full text cleaning pipeline:
      1. Lowercase
      2. Remove URLs
      3. Remove email addresses
      4. Remove phone numbers
      5. Remove punctuation
      6. Remove digits
      7. Collapse whitespace
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+",     " ",  text)   # URLs
    text = re.sub(r"\S+@\S+",              " ",  text)   # emails
    text = re.sub(r"\b\d{10,}\b",          " ",  text)   # phone numbers
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)  # punctuation
    text = re.sub(r"\d+",                  " ",  text)   # digits
    text = re.sub(r"\s+",                  " ",  text).strip()
    return text

# ── Load Dataset ──────────────────────────────────────────────────
def load_dataset():
    # Try online sources
    for url in DATASET_URLS:
        try:
            print(f"[..] Trying: {url}")
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()

            # TSV format (label \t text)
            if url.endswith(".tsv"):
                df = pd.read_csv(io.StringIO(r.text), sep="\t",
                                 header=None, names=["label", "text"])
            else:
                df = pd.read_csv(io.StringIO(r.text), encoding="latin-1")
                # Keep only first two columns
                df = df.iloc[:, :2]
                df.columns = ["label", "text"]

            df["label"] = df["label"].str.strip().str.lower()
            df = df[df["label"].isin(["ham", "spam"])].dropna()
            df["label_int"] = (df["label"] == "spam").astype(int)
            print(f"[OK] Loaded {len(df):,} emails "
                  f"({df['label_int'].sum()} spam, {(df['label_int']==0).sum()} ham)")
            return df
        except Exception as e:
            print(f"[!!] Failed: {e}")
            continue

    # Synthetic fallback
    print("[..] Using synthetic demo data")
    return make_synthetic()

def make_synthetic():
    """Realistic synthetic spam/ham dataset as fallback."""
    np.random.seed(42)
    spam_templates = [
        "WINNER!! You have been selected for a cash prize of {n} pounds call now to claim",
        "FREE entry to win {n} tickets text WIN to 87121 now limited time offer",
        "Urgent your mobile number has won a {n} prize claim now call 09061 free",
        "Congratulations you won a prize worth {n} call this number to claim your reward",
        "You have been awarded {n} cash bonus click here to redeem now expires soon",
        "SIX chances to win CASH Txt GOLD to 87066 cost 150p per msg receive {n}",
        "PRIVATE! Your 2004 account statement for {n} shows a prize claim immediately",
        "Had your mobile {n} months update to latest camera phone for free call now",
        "URGENT! We are trying to contact you last weekends draw shows you won {n} prize",
        "Txt STOP if u want 2 stop receivin these msgs from us {n} special offer ends today",
    ]
    ham_templates = [
        "Hey can we meet tomorrow for lunch I was thinking around noon",
        "The meeting has been rescheduled to {n}pm please confirm if you can attend",
        "Thanks for letting me know I will be there on time see you soon",
        "Can you please send me the report by end of day today thanks",
        "Happy birthday hope you have a wonderful day full of joy and happiness",
        "Just checking in how are you doing hope everything is going well",
        "I will call you back in {n} minutes just finishing up at work",
        "The project deadline has been moved to next Friday please update your schedule",
        "Good morning hope you slept well see you at the office later today",
        "Do you want to grab coffee after class today around {n} oclock",
    ]
    rows = []
    for _ in range(747):
        t = np.random.choice(spam_templates).replace("{n}", str(np.random.randint(100,9999)))
        rows.append({"label": "spam", "text": t, "label_int": 1})
    for _ in range(4825):
        t = np.random.choice(ham_templates).replace("{n}", str(np.random.randint(1,12)))
        rows.append({"label": "ham", "text": t, "label_int": 0})
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ── Feature Extraction ─────────────────────────────────────────────
def extract_features(text: str) -> dict:
    """Extra hand-crafted features for display purposes."""
    return {
        "char_count":    len(text),
        "word_count":    len(text.split()),
        "upper_ratio":   round(sum(1 for c in text if c.isupper()) / max(len(text),1), 3),
        "exclaim_count": text.count("!"),
        "digit_count":   sum(1 for c in text if c.isdigit()),
        "currency":      any(c in text for c in ["£","$","€","₹"]),
        "has_url":       bool(re.search(r"http|www", text, re.I)),
        "has_phone":     bool(re.search(r"\b\d{7,}\b", text)),
    }

# ── Train Models ──────────────────────────────────────────────────
def train_all():
    global models, vectorizer, model_metrics, models_ready

    print("\n" + "="*52)
    print("  Task 1: Email Spam Classification")
    print("  ARCH Technologies — ML Internship")
    print("="*52)

    df = load_dataset()

    # Preprocess
    print("[..] Preprocessing text...")
    df["clean"] = df["text"].apply(preprocess)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label_int"],
        test_size=0.20, random_state=42, stratify=df["label_int"]
    )

    # TF-IDF vectorization
    print("[..] Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=2
    )
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)
    print(f"[OK] TF-IDF: {X_tr.shape[1]:,} features | "
          f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Define models to train
    model_defs = {
        "Naive Bayes":          MultinomialNB(alpha=0.1),
        "Logistic Regression":  LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
        "Linear SVM":           LinearSVC(C=1.0, max_iter=2000),
    }

    # Train and evaluate each
    for name, clf in model_defs.items():
        print(f"[..] Training {name}...")
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        models[name] = clf
        model_metrics[name] = {
            "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 1),
            "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 1),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100, 1),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0) * 100, 1),
            "cm":        confusion_matrix(y_test, y_pred).tolist(),
        }
        m = model_metrics[name]
        print(f"[OK] {name:22s} Acc:{m['accuracy']}%  F1:{m['f1']}%")

    # Dataset stats
    model_metrics["_dataset"] = {
        "total":      len(df),
        "spam":       int(df["label_int"].sum()),
        "ham":        int((df["label_int"] == 0).sum()),
        "vocab_size": len(vectorizer.vocabulary_),
        "train_size": len(X_train),
        "test_size":  len(X_test),
    }

    models_ready = True
    print("="*52)
    print("  All models ready!")
    print("  http://127.0.0.1:5000")
    print("="*52 + "\n")

# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/metrics")
def metrics_page():
    return render_template("metrics.html")

@app.route("/history")
def history_page():
    return render_template("history.html",
                           history=list(reversed(prediction_history)))

@app.route("/api/status")
def api_status():
    return jsonify({
        "ready":   models_ready,
        "metrics": model_metrics,
        "models":  list(models.keys()),
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not models_ready:
        return jsonify({"error": "Models not ready"}), 503

    data    = request.get_json()
    text    = data.get("text", "").strip()
    model_n = data.get("model", "Naive Bayes")

    if len(text) < 3:
        return jsonify({"error": "Text too short"}), 400
    if model_n not in models:
        return jsonify({"error": "Unknown model"}), 400

    clf     = models[model_n]
    cleaned = preprocess(text)
    vec     = vectorizer.transform([cleaned])

    pred    = int(clf.predict(vec)[0])
    label   = "spam" if pred == 1 else "ham"

    # Probability (SVM doesn't have predict_proba)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(vec)[0].tolist()
        confidence = round(max(proba) * 100, 1)
    else:
        # Use decision function for SVM
        score = clf.decision_function(vec)[0]
        prob_spam = round(1 / (1 + np.exp(-score)) * 100, 1)
        proba = [round(100 - prob_spam, 1) / 100, round(prob_spam, 1) / 100]
        confidence = round(abs(score / (abs(score) + 1)) * 100, 1)

    features = extract_features(text)

    # Save to history
    entry = {
        "id":         len(prediction_history) + 1,
        "text":       text[:200],
        "label":      label,
        "confidence": confidence,
        "model":      model_n,
        "features":   features,
        "timestamp":  datetime.now().strftime("%b %d, %Y · %H:%M:%S"),
        "word_count": len(text.split()),
    }
    prediction_history.append(entry)

    return jsonify({
        "label":      label,
        "confidence": confidence,
        "proba":      {"ham": round(proba[0] * 100, 1), "spam": round(proba[1] * 100, 1)},
        "features":   features,
        "model":      model_n,
        "cleaned":    cleaned[:300],
    })

@app.route("/api/history")
def api_history():
    return jsonify(list(reversed(prediction_history)))

@app.route("/api/history/clear", methods=["POST"])
def api_clear():
    prediction_history.clear()
    return jsonify({"ok": True})

# ── Launch ─────────────────────────────────────────────────────────
def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    # Train in main thread (before server starts)
    train_all()
    threading.Timer(1.2, open_browser).start()
    app.run(debug=False, port=5000, use_reloader=False)
