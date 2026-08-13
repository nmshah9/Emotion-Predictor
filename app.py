"""
app.py
======
STREAMLIT WEB APP for the Emotion Detection project.

Run with:   streamlit run app.py

WHAT THIS FILE DOES:
Loads the model artifacts that train_model.py already trained and saved to
disk (models/best_model.pkl, models/vectorizer.pkl, models/label_encoder.pkl)
and wraps them in a simple web UI: the user types a sentence, we clean it
with the SAME function used during training (from utils.py, to avoid
training/serving mismatch), convert it to features with the SAME fitted
vectorizer, and feed it to the SAME trained model to get a predicted emotion
+ confidence scores for every class.

IMPORTANT: This app does NOT train anything itself - it only loads what
train_model.py already produced. You must run `python train_model.py` once
before `streamlit run app.py` will work (see README.md).
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils import clean_text

# --------------------------------------------------------------------------
# PAGE CONFIG - sets browser tab title/icon and a wide, welcoming layout.
# Must be the first Streamlit command in the script.
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="centered",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

EMOTION_EMOJI = {
    "joy": "😀",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲",
}
EMOTION_COLOR = {
    "joy": "#FFD93D",
    "sadness": "#4D96FF",
    "anger": "#FF4D4D",
    "fear": "#9D4DFF",
    "love": "#FF6FA8",
    "surprise": "#4DFFB8",
}


# --------------------------------------------------------------------------
# LOAD ARTIFACTS (cached so this only runs ONCE per app session, not on
# every button click / rerun - loading pickle files from disk is cheap but
# there's no reason to repeat it unnecessarily).
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model...")
def load_artifacts():
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    label_encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    embedding_info_path = os.path.join(MODELS_DIR, "embedding_info.pkl")

    if not (os.path.exists(model_path) and os.path.exists(label_encoder_path)):
        return None  # signals "not trained yet" to the UI below

    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    embedding_info = joblib.load(embedding_info_path)

    vectorizer = None
    w2v_model = None
    if embedding_info["type"] == "Word2Vec":
        from gensim.models import Word2Vec
        w2v_model = Word2Vec.load(os.path.join(MODELS_DIR, "word2vec.model"))
    else:
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))

    comparison_path = os.path.join(MODELS_DIR, "comparison_table.csv")
    comparison_df = pd.read_csv(comparison_path) if os.path.exists(comparison_path) else None

    return {
        "model": model,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "w2v_model": w2v_model,
        "embedding_type": embedding_info["type"],
        "comparison_df": comparison_df,
    }


def word2vec_sentence_vector(tokens, model, dim=100):
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)


def predict_emotion(text, artifacts):
    """
    Takes a raw sentence typed by the user and returns:
      - predicted_label: the single most likely emotion (string)
      - probs: a dict of {emotion_label: probability} for every class,
               used to draw the confidence bar chart.
    """
    cleaned = clean_text(text, method="lemmatize")

    if artifacts["embedding_type"] == "Word2Vec":
        tokens = cleaned.split()
        X = word2vec_sentence_vector(tokens, artifacts["w2v_model"]).reshape(1, -1)
    else:
        X = artifacts["vectorizer"].transform([cleaned])

    model = artifacts["model"]
    pred_idx = model.predict(X)[0]
    predicted_label = artifacts["label_encoder"].inverse_transform([pred_idx])[0]

    probs = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        for idx, p in enumerate(proba):
            label = artifacts["label_encoder"].inverse_transform([idx])[0]
            probs[label] = float(p)
    else:
        # Models without predict_proba (rare here) still get a usable UI:
        # give the predicted class 100% and everything else 0%.
        for label in artifacts["label_encoder"].classes_:
            probs[label] = 1.0 if label == predicted_label else 0.0

    return predicted_label, cleaned, probs


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🎭 Emotion Detection from Text")
st.write(
    "Type a sentence below and the model will predict the underlying "
    "emotion: **joy, sadness, anger, fear, love,** or **surprise**."
)

artifacts = load_artifacts()

if artifacts is None:
    st.error(
        "No trained model was found in the `models/` folder.\n\n"
        "This app only loads an already-trained model - it doesn't train one "
        "itself. Please run the training script first:\n\n"
        "```\npython train_model.py\n```\n\n"
        "That will create `models/best_model.pkl` and the other files this "
        "app needs, and only takes a few minutes. Then re-run "
        "`streamlit run app.py`."
    )
    st.stop()

with st.sidebar:
    st.header("ℹ️ About this model")
    st.write(f"**Feature engineering used:** {artifacts['embedding_type']}")
    if artifacts["comparison_df"] is not None:
        st.write("**Model comparison (on held-out test set):**")
        st.dataframe(
            artifacts["comparison_df"][["Model", "Accuracy", "F1-score (weighted)"]]
            .round(4),
            hide_index=True,
        )
    st.caption(
        "This model was trained on the dair-ai Emotion dataset (6 classes) "
        "using classic ML algorithms (Logistic Regression / Decision Tree / "
        "Random Forest) with hyperparameter tuning. See train_model.py and "
        "the notebook for the full pipeline."
    )

user_text = st.text_area(
    "Enter a sentence:",
    placeholder="e.g. I can't believe I got the job, this is the best day ever!",
    height=120,
)

col1, col2 = st.columns([1, 4])
with col1:
    predict_clicked = st.button("Predict Emotion", type="primary")

if predict_clicked:
    if not user_text.strip():
        st.warning("Please type a sentence first.")
    else:
        predicted_label, cleaned_text, probs = predict_emotion(user_text, artifacts)
        emoji = EMOTION_EMOJI.get(predicted_label, "")
        color = EMOTION_COLOR.get(predicted_label, "#888888")

        st.markdown(
            f"""
            <div style="padding:20px;border-radius:12px;background-color:{color}22;
                        border:2px solid {color};text-align:center;">
                <h2 style="margin:0;">{emoji} {predicted_label.upper()}</h2>
                <p style="margin:4px 0 0 0;">Confidence: {probs[predicted_label]*100:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.subheader("Confidence across all emotions")
        prob_df = (
            pd.DataFrame({"Emotion": list(probs.keys()), "Confidence": list(probs.values())})
            .sort_values("Confidence", ascending=False)
            .set_index("Emotion")
        )
        st.bar_chart(prob_df)

        with st.expander("See how your text was cleaned before prediction"):
            st.write("**Original:**", user_text)
            st.write("**Cleaned (lowercased, stopwords removed, lemmatized):**", cleaned_text or "*(empty after cleaning)*")

st.divider()
st.caption(
    "Built with scikit-learn + Streamlit | Dataset: dair-ai Emotion dataset "
    "(train/val/test text;label format)"
)
