"""
train_model.py
===============
END-TO-END EMOTION DETECTION TRAINING PIPELINE

Run this file with:   python train_model.py

WHAT THIS SCRIPT DOES (matches the assignment steps i - vi):
    i.   Loads the dataset (data/train.txt, val.txt, test.txt)
    ii.  Cleans the text (lowercase, stopword removal, stemming/lemmatization)
    iii. Builds three feature-engineering / word-embedding representations:
             Bag-of-Words, TF-IDF, Word2Vec
         and compares them with a quick baseline model to pick a winner.
    iv.  Trains three ML algorithms on the winning representation:
             Logistic Regression, Decision Tree, Random Forest
    v.   Hyperparameter-tunes each of the three models with GridSearchCV /
         RandomizedSearchCV.
    vi.  Builds a final comparison table of every model, prints the best
         model, and SAVES the best model + vectorizer + label encoder to
         disk (models/) so the Streamlit app can load them instantly without
         retraining.

Every section below is preceded by a comment explaining WHY that step exists,
not just what it does - written so this file can double as a learning
reference on top of being production-usable code.
"""

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from gensim.models import Word2Vec

from utils import clean_text, clean_tokens, load_emotion_file

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# CONFIG - change these if you want a faster/slower run
# --------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
RANDOM_STATE = 42
MAX_FEATURES = 5000          # vocabulary cap for BoW / TF-IDF
W2V_DIM = 100                # Word2Vec vector size

os.makedirs(MODELS_DIR, exist_ok=True)


def log(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}")


# ==========================================================================
# STEP i - LOAD THE DATASET
# ==========================================================================
# Why: We need labelled examples ("text" + its "emotion") before anything
# else can happen. The dataset ships as three files (train/val/test) so the
# model is always evaluated on data it never saw during training/tuning -
# this is what makes the reported accuracy trustworthy rather than optimistic.
log("STEP 1/6: Loading dataset")

train_df = load_emotion_file(os.path.join(DATA_DIR, "train.txt"))
val_df = load_emotion_file(os.path.join(DATA_DIR, "val.txt"))
test_df = load_emotion_file(os.path.join(DATA_DIR, "test.txt"))

# We fold val.txt INTO the training pool. Why: GridSearchCV/RandomizedSearchCV
# already do their own internal cross-validation splits to pick
# hyperparameters, so a separately-held-out validation file is redundant here
# and we'd rather give the models more data to learn from. test.txt is kept
# completely untouched until the very final evaluation.
full_train_df = pd.concat([train_df, val_df], ignore_index=True)

print(f"Training pool (train+val): {full_train_df.shape[0]} sentences")
print(f"Held-out test set        : {test_df.shape[0]} sentences")
print("\nClass distribution (training pool):")
print(full_train_df["emotion"].value_counts())


# ==========================================================================
# STEP ii - DATA CLEANING
# ==========================================================================
# Why: Raw text is noisy for ML models - inconsistent casing, punctuation,
# and filler words ("the", "is", "a") add dimensions to the feature space
# without adding signal about WHICH EMOTION is present. Cleaning shrinks and
# focuses the vocabulary on words that actually carry emotional meaning
# (see utils.clean_text for a step-by-step breakdown of lowercase -> remove
# punctuation -> remove stopwords -> lemmatize).
log("STEP 2/6: Cleaning text (lowercase, stopword removal, lemmatization)")

t0 = time.time()
full_train_df["clean_text"] = full_train_df["text"].apply(lambda x: clean_text(x, method="lemmatize"))
test_df["clean_text"] = test_df["text"].apply(lambda x: clean_text(x, method="lemmatize"))
full_train_df["tokens"] = full_train_df["clean_text"].apply(lambda x: x.split())
test_df["tokens"] = test_df["clean_text"].apply(lambda x: x.split())
print(f"Cleaning finished in {time.time()-t0:.2f}s")

print("\nBefore vs after cleaning (first 3 examples):")
for i in range(3):
    print(f"  RAW  : {full_train_df['text'].iloc[i]}")
    print(f"  CLEAN: {full_train_df['clean_text'].iloc[i]}\n")

# Emotion labels are text ("joy", "anger", ...). ML models need numbers, so
# LabelEncoder maps each unique label to an integer (e.g. joy=2). We fit it
# ONLY on the training labels and reuse it for the test labels, and we save
# it to disk so the Streamlit app can translate the model's numeric output
# back into a human-readable emotion name.
label_encoder = LabelEncoder()
y_train_full = label_encoder.fit_transform(full_train_df["emotion"])
y_test = label_encoder.transform(test_df["emotion"])
print("Emotion classes:", list(label_encoder.classes_))


# ==========================================================================
# STEP iii - FEATURE ENGINEERING / WORD EMBEDDINGS
# ==========================================================================
# Why compare THREE representations instead of picking one blindly:
# different embeddings capture different things, and the best choice is an
# empirical question, not a guess.
#   - Bag-of-Words (BoW): counts how many times each word appears. Simple,
#     but treats "very good" and "good very" identically and ignores word
#     importance.
#   - TF-IDF: like BoW, but down-weights words that appear in almost every
#     sentence (low information) and up-weights words that are rare/specific
#     to a document (high information) - usually a stronger baseline than BoW.
#   - Word2Vec: learns dense vectors where semantically similar words end up
#     close together in vector space (e.g. "happy" and "joyful" are near each
#     other) - captures meaning that pure counting cannot, but needs enough
#     data to learn good vectors and is more expensive to compute.
#
# We train a fast Logistic Regression on EACH representation using an
# 80/20 split of the training pool (not the real test set - we don't want to
# "peek" at test data while just choosing an embedding) and pick whichever
# gives the best validation F1-score to carry forward into full model
# training and tuning.
log("STEP 3/6: Feature engineering - comparing BoW vs TF-IDF vs Word2Vec")

from sklearn.model_selection import train_test_split

X_tr_text, X_val_text, y_tr, y_val = train_test_split(
    full_train_df["clean_text"], y_train_full, test_size=0.2,
    random_state=RANDOM_STATE, stratify=y_train_full
)
tokens_tr, tokens_val = train_test_split(
    full_train_df["tokens"], test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_full
)

embedding_results = []

# --- Bag of Words ---
t0 = time.time()
bow_vec = CountVectorizer(max_features=MAX_FEATURES)
Xtr_bow = bow_vec.fit_transform(X_tr_text)
Xval_bow = bow_vec.transform(X_val_text)
clf = LogisticRegression(max_iter=300, n_jobs=-1)
clf.fit(Xtr_bow, y_tr)
f1 = f1_score(y_val, clf.predict(Xval_bow), average="weighted")
embedding_results.append({"Embedding": "Bag-of-Words", "Val F1 (weighted)": f1, "Time (s)": round(time.time()-t0, 2)})
print(f"Bag-of-Words   -> val F1: {f1:.4f}")

# --- TF-IDF ---
t0 = time.time()
tfidf_vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))
Xtr_tfidf = tfidf_vec.fit_transform(X_tr_text)
Xval_tfidf = tfidf_vec.transform(X_val_text)
clf = LogisticRegression(max_iter=300, n_jobs=-1)
clf.fit(Xtr_tfidf, y_tr)
f1 = f1_score(y_val, clf.predict(Xval_tfidf), average="weighted")
embedding_results.append({"Embedding": "TF-IDF", "Val F1 (weighted)": f1, "Time (s)": round(time.time()-t0, 2)})
print(f"TF-IDF         -> val F1: {f1:.4f}")

# --- Word2Vec (average of word vectors per sentence) ---
t0 = time.time()
w2v_model = Word2Vec(sentences=tokens_tr.tolist(), vector_size=W2V_DIM, window=5,
                      min_count=2, workers=1, epochs=15, seed=RANDOM_STATE)


def sentence_vector(tokens, model, dim=W2V_DIM):
    """Average the word vectors of every known word in a sentence.
    Unknown words are skipped; an all-unknown/empty sentence -> zero vector."""
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)


Xtr_w2v = np.array([sentence_vector(t, w2v_model) for t in tokens_tr])
Xval_w2v = np.array([sentence_vector(t, w2v_model) for t in tokens_val])
clf = LogisticRegression(max_iter=300, n_jobs=-1)
clf.fit(Xtr_w2v, y_tr)
f1 = f1_score(y_val, clf.predict(Xval_w2v), average="weighted")
embedding_results.append({"Embedding": "Word2Vec", "Val F1 (weighted)": f1, "Time (s)": round(time.time()-t0, 2)})
print(f"Word2Vec       -> val F1: {f1:.4f}")

embedding_df = pd.DataFrame(embedding_results).sort_values("Val F1 (weighted)", ascending=False)
print("\nEmbedding comparison:\n", embedding_df.to_string(index=False))

best_embedding = embedding_df.iloc[0]["Embedding"]
print(f"\n>>> Best embedding: {best_embedding} - this will be used for the final models.")

# For classic ML algorithms (Logistic Regression / Decision Tree / Random
# Forest) on short, informal sentences, TF-IDF's sparse-count weighting
# typically edges out Word2Vec's averaged dense vectors and comfortably beats
# raw Bag-of-Words - we let the numbers above confirm/deny that rather than
# assuming it. We proceed with whichever representation actually won.
if best_embedding == "TF-IDF":
    final_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))
elif best_embedding == "Bag-of-Words":
    final_vectorizer = CountVectorizer(max_features=MAX_FEATURES)
else:
    final_vectorizer = None  # Word2Vec handled separately below

if final_vectorizer is not None:
    X_train_final = final_vectorizer.fit_transform(full_train_df["clean_text"])
    X_test_final = final_vectorizer.transform(test_df["clean_text"])
else:
    # Retrain Word2Vec on the FULL training pool (not just the 80% split
    # used for comparison) so the final model gets to see all available data.
    w2v_full = Word2Vec(sentences=full_train_df["tokens"].tolist(), vector_size=W2V_DIM,
                         window=5, min_count=2, workers=1, epochs=15, seed=RANDOM_STATE)
    X_train_final = np.array([sentence_vector(t, w2v_full) for t in full_train_df["tokens"]])
    X_test_final = np.array([sentence_vector(t, w2v_full) for t in test_df["tokens"]])


# ==========================================================================
# STEP iv & v - TRAIN + HYPERPARAMETER-TUNE THE THREE ML MODELS
# ==========================================================================
# Why tune at all: a model's DEFAULT settings are rarely optimal for a
# specific dataset. GridSearchCV/RandomizedSearchCV try several
# configurations, score each with cross-validation (so the choice isn't
# based on a lucky single split), and keep the best one - this typically
# improves accuracy over an untuned model at the cost of extra compute time.
log("STEP 4-5/6: Training & hyperparameter-tuning Logistic Regression, "
    "Decision Tree, Random Forest")

results = []


def evaluate(name, model, X_test, y_test, train_time, params):
    preds = model.predict(X_test)
    row = {
        "Model": name,
        "Best Params": str(params),
        "Accuracy": accuracy_score(y_test, preds),
        "Precision (weighted)": precision_score(y_test, preds, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_test, preds, average="weighted", zero_division=0),
        "F1-score (weighted)": f1_score(y_test, preds, average="weighted", zero_division=0),
        "Train Time (s)": round(train_time, 2),
    }
    results.append(row)
    print(f"\n{name}: acc={row['Accuracy']:.4f}  f1={row['F1-score (weighted)']:.4f}  "
          f"params={params}  time={row['Train Time (s)']:.1f}s")
    return row


# --- Logistic Regression ---
# Why these hyperparameters: 'C' controls regularisation strength (smaller C
# = stronger regularisation = simpler decision boundary, larger C = fits
# training data more closely). We search across values spanning both ends,
# and try two solvers because they optimise the same objective differently
# and can land on slightly different results.
print("\nTuning Logistic Regression ...")
t0 = time.time()
# NOTE: 'liblinear' is intentionally excluded - in recent scikit-learn
# versions it no longer supports multiclass problems (6 emotion classes)
# without an extra OneVsRestClassifier wrapper, so we search over solvers
# that natively handle multiclass instead.
lr_grid = {
    "C": [0.1, 1, 5, 10],
    "solver": ["lbfgs", "saga"],
}
lr_search = GridSearchCV(
    LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
    lr_grid, cv=3, scoring="f1_weighted", n_jobs=-1
)
lr_search.fit(X_train_final, y_train_full)
lr_time = time.time() - t0
evaluate("Logistic Regression (tuned)", lr_search.best_estimator_, X_test_final, y_test, lr_time, lr_search.best_params_)

# --- Decision Tree ---
# Why these hyperparameters: 'max_depth' controls how deep/complex the tree
# can grow (unlimited depth easily overfits text data with thousands of
# features); 'min_samples_split' and 'min_samples_leaf' require a minimum
# number of samples before the tree is allowed to split further, which also
# guards against overfitting to noise in a handful of training examples.
print("\nTuning Decision Tree ...")
t0 = time.time()
dt_grid = {
    "max_depth": [20, 40, 60, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
dt_search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    dt_grid, n_iter=12, cv=3, scoring="f1_weighted", n_jobs=-1, random_state=RANDOM_STATE
)
dt_search.fit(X_train_final, y_train_full)
dt_time = time.time() - t0
evaluate("Decision Tree (tuned)", dt_search.best_estimator_, X_test_final, y_test, dt_time, dt_search.best_params_)

# --- Random Forest ---
# Why these hyperparameters: 'n_estimators' is how many trees vote in the
# ensemble (more trees = more stable predictions but slower training);
# 'max_depth' and 'min_samples_split' control individual tree complexity,
# same overfitting logic as the Decision Tree above. We use RandomizedSearchCV
# instead of an exhaustive GridSearchCV here because Random Forest is the
# slowest model to train and an exhaustive grid would take far too long for
# only a marginal accuracy gain over a well-chosen random sample of
# combinations.
print("\nTuning Random Forest ...")
t0 = time.time()
rf_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [20, 40, None],
    "min_samples_split": [2, 5],
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    rf_grid, n_iter=6, cv=3, scoring="f1_weighted", n_jobs=1, random_state=RANDOM_STATE
)
rf_search.fit(X_train_final, y_train_full)
rf_time = time.time() - t0
evaluate("Random Forest (tuned)", rf_search.best_estimator_, X_test_final, y_test, rf_time, rf_search.best_params_)


# ==========================================================================
# STEP vi - FINAL COMPARISON TABLE + SAVE THE BEST MODEL
# ==========================================================================
# Why: after tuning every model individually, we need one place that lines
# them all up on the SAME held-out test set so the comparison is fair, then
# we programmatically pick the winner (highest weighted F1-score, which
# balances precision and recall and handles the class imbalance visible in
# the emotion counts printed in Step 1) rather than eyeballing it.
log("STEP 6/6: Final model comparison")

comparison_df = pd.DataFrame(results).sort_values("F1-score (weighted)", ascending=False).reset_index(drop=True)
print("\nFINAL MODEL COMPARISON TABLE (embedding used: {}):\n".format(best_embedding))
print(comparison_df.to_string(index=False))

comparison_df.to_csv(os.path.join(MODELS_DIR, "comparison_table.csv"), index=False)
embedding_df.to_csv(os.path.join(MODELS_DIR, "embedding_comparison.csv"), index=False)

best_row = comparison_df.iloc[0]
best_model_name = best_row["Model"]
print(f"\n>>> BEST MODEL: {best_model_name} "
      f"(F1-weighted = {best_row['F1-score (weighted)']:.4f}, "
      f"Accuracy = {best_row['Accuracy']:.4f})")

model_map = {
    "Logistic Regression (tuned)": lr_search.best_estimator_,
    "Decision Tree (tuned)": dt_search.best_estimator_,
    "Random Forest (tuned)": rf_search.best_estimator_,
}
best_model = model_map[best_model_name]

# Persist everything the Streamlit app needs to make predictions WITHOUT
# retraining: the trained model itself, the fitted vectorizer (so new text
# is converted into the exact same feature space the model was trained on),
# and the label encoder (to turn the model's numeric prediction back into a
# readable emotion word).
joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
joblib.dump({"type": best_embedding}, os.path.join(MODELS_DIR, "embedding_info.pkl"))

if final_vectorizer is not None:
    joblib.dump(final_vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))
else:
    # If Word2Vec won, save the Word2Vec model itself instead of a vectorizer
    w2v_full.save(os.path.join(MODELS_DIR, "word2vec.model"))

print(f"\nSaved best model + preprocessing artifacts to: {MODELS_DIR}")
print("You can now run:  streamlit run app.py")
