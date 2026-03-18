# SpamGuard — Email Spam Classifier

## Models
| Model | Description |
|-------|-------------|
| Naive Bayes | Fast probabilistic classifier, great baseline for text |
| Logistic Regression | Linear model with probability estimates |
| Linear SVM | Strong text classifier, often best performer |

## Project Structure
```
spam_classifier/
├── app.py               # Flask backend + ML pipeline
├── requirements.txt
├── README.md
└── templates/
    ├── base.html        # Shared layout + sidebar
    ├── index.html       # Classifier page  ( / )
    ├── metrics.html     # Model evaluation ( /metrics )
    └── history.html     # History log      ( /history )
```

## Setup & Run
```bash
pip install -r requirements.txt
python app.py
```
Browser opens automatically at `http://127.0.0.1:5000`

## ML Pipeline
```
Raw email text
   ↓
Text Cleaning (lowercase, remove URLs, punctuation, digits)
   ↓
TF-IDF Vectorization (8,000 features, unigrams + bigrams)
   ↓
3 classifiers trained: Naive Bayes, Logistic Regression, Linear SVM
   ↓
Evaluated on 20% test set → Accuracy, Precision, Recall, F1
   ↓
Real-time prediction via Flask API
```

## Dataset
Auto-downloads SMS Spam Collection dataset from GitHub.
Falls back to synthetic data if no internet connection.
