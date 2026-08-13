# 🎭 Emotion Detection from Text — End-to-End Project

Predicts one of 6 emotions (**joy, sadness, anger, fear, love, surprise**)
from a sentence of text, using classic ML algorithms (Logistic Regression,
Decision Tree, Random Forest) trained on cleaned/vectorised text, with a
Streamlit web app for interactive predictions.

## What's inside this folder

```
emotion_detection_project/
├── data/
│   ├── train.txt              # training sentences (text;label per line)
│   ├── val.txt                # validation sentences
│   └── test.txt                # held-out test sentences
├── notebooks/
│   └── Emotion_Detection_EndToEnd.ipynb   # interactive, fully-explained notebook
├── models/                     # created after you run training (see below)
│   ├── best_model.pkl
│   ├── vectorizer.pkl
│   ├── label_encoder.pkl
│   ├── embedding_info.pkl
│   ├── comparison_table.csv
│   └── embedding_comparison.csv
├── utils.py                    # shared text-cleaning functions (used by both training and the app)
├── train_model.py              # the .py end-to-end training script
├── app.py                      # the Streamlit web app
├── requirements.txt
└── README.md                   # this file
```

---

## Step-by-step setup in VS Code (do this once)

### Step 1 — Install Python 3.12

Confirm you have Python 3.12 installed:
```bash
python --version
```
If it prints something other than `3.12.x`, install Python 3.12 from
[python.org](https://www.python.org/downloads/) first. **Why:** the project
was built and tested on 3.12.3; using a very different major version can
occasionally cause package-installation issues.

### Step 2 — Open the project folder in VS Code

`File → Open Folder...` → select the `emotion_detection_project` folder.
**Why:** VS Code's integrated terminal and Jupyter extension both default
to using this folder as the working directory, which keeps all the
relative file paths in the code correct.

### Step 3 — Create a virtual environment

In the VS Code terminal (`` Ctrl+` ``):
```bash
python -m venv venv
```
**Why:** a virtual environment keeps this project's package versions
isolated from any other Python projects on your machine, avoiding version
conflicts.

Activate it:
- **Windows (CMD):** `venv\Scripts\activate.bat`
- **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
- **macOS/Linux:** `source venv/bin/activate`

You'll know it worked because your terminal prompt now starts with `(venv)`.

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```
**Why:** this installs the exact libraries the project needs — pandas/numpy
for data handling, scikit-learn for ML models, nltk for text cleaning,
gensim for Word2Vec, joblib for saving models, and streamlit for the web
app.

> **First-time-only internet requirement:** the first time you run the
> training script or notebook, NLTK will automatically download two small
> data packages it needs (a stopword list and a lemmatisation dictionary —
> a few hundred KB total). This needs an internet connection **once**; after
> that they're cached locally on your machine and no further downloads
> happen. If your network blocks `nltk.org`/`github.com`, see
> **Troubleshooting** below.

### Step 5 — In VS Code, select the right Python interpreter

`Ctrl+Shift+P` → `Python: Select Interpreter` → choose the one that shows
`('venv': venv)`. **Why:** otherwise VS Code may run your code/notebook
with a different Python installation than the one you just installed
packages into, and you'll get `ModuleNotFoundError`.

---

## Running the project

### Option A — Run the training script (fastest way to get a working app)

```bash
python train_model.py
```

This runs the full pipeline (see **What the pipeline does** below) and
prints its progress to the terminal, finishing with a model comparison
table and confirmation that the model was saved. **Expected runtime: roughly
3–10 minutes**, depending on your CPU (more CPU cores = faster, since
several steps use all available cores automatically).

You'll see 6 clearly labelled stages print to the terminal as it runs, and
at the end a `models/` folder will contain everything needed for the app.

### Option B — Run the Jupyter notebook (to see and understand each step)

Open `notebooks/Emotion_Detection_EndToEnd.ipynb` in VS Code and click
**Run All** (or run cells one at a time to inspect intermediate outputs).
Every code cell is preceded by a markdown cell explaining *why* that step
matters, not just what it does. It produces the exact same saved model
files as `train_model.py`.

### Step 6 — Launch the Streamlit app

Once either Option A or Option B has completed (so `models/best_model.pkl`
exists):
```bash
streamlit run app.py
```
Your browser should open automatically to `http://localhost:8501`. If not,
copy that URL into your browser manually. Type a sentence, click
**Predict Emotion**, and you'll see the predicted emotion, a confidence bar
chart across all 6 emotions, and (by expanding a section) how your text was
cleaned before being fed to the model.

To stop the app, go back to the terminal and press `Ctrl+C`.

---

## What the pipeline does (matches the assignment steps)

| # | Step | Where it happens |
|---|---|---|
| i | Load the dataset (`train.txt`/`val.txt`/`test.txt`) | `utils.load_emotion_file()` |
| ii | Clean text: lowercase, remove punctuation, remove stopwords, lemmatize (stemming also available) | `utils.clean_text()` |
| iii | Compare 3 feature-engineering techniques — Bag-of-Words, TF-IDF, Word2Vec — and keep the best-performing one | `train_model.py` Step 3 |
| iv | Train Logistic Regression, Decision Tree, and Random Forest | `train_model.py` Step 4 |
| v | Hyperparameter-tune each model with GridSearchCV / RandomizedSearchCV | `train_model.py` Step 5 |
| vi | Build a final comparison table (Accuracy / Precision / Recall / F1) and save the best model | `train_model.py` Step 6, saved to `models/comparison_table.csv` |

Every step is explained in detail with inline comments in `train_model.py`
and with markdown cells in the notebook — open either file to read the
*why* behind each choice, not just the code.

**On a reference run**, the pipeline reported (your exact numbers may vary
slightly run-to-run due to algorithmic randomness even with a fixed seed,
depending on your hardware/library versions):

| Model | Accuracy | F1-score (weighted) |
|---|---|---|
| Logistic Regression (tuned) | ~0.88 | ~0.88 |
| Random Forest (tuned) | ~0.88 | ~0.88 |
| Decision Tree (tuned) | ~0.87 | ~0.87 |

Logistic Regression on Bag-of-Words features was the best performer in
testing — fast to train, and a strong, well-regularised linear baseline is
often hard to beat on short, clean text classification tasks like this one.
Your run may pick a different winner depending on the exact
cross-validation folds and hardware — the script always saves whichever
model actually wins on your run, not a hardcoded choice.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'utils'`**
Make sure your terminal's current directory is the project root (not
`notebooks/`) when running `train_model.py`/`app.py`. The notebook handles
this automatically via a `sys.path` fix in its first cell.

**NLTK download fails / times out (no internet, or firewall)**
Manually download once from a machine with internet, then copy the
resulting `nltk_data` folder (usually created at `~/nltk_data` on
macOS/Linux or `%APPDATA%\nltk_data` on Windows) onto the machine running
this project — this only ever needs to happen once, before that first run.

**`streamlit run app.py` says "No trained model was found"**
You need to run `python train_model.py` (or run the notebook) at least
once first — the app only *loads* an already-trained model, it doesn't
train one itself.

**Training feels slow**
Random Forest hyperparameter tuning is the slowest stage (it trains many
trees, several times, as part of the search). This is expected and normal
— give it a few minutes. If you want a faster run for testing purposes,
you can reduce `n_iter` in the `RandomizedSearchCV` calls inside
`train_model.py`, at some cost to how thoroughly the best hyperparameters
are searched.

**Different results than expected**
Small differences in accuracy/F1 across runs or machines are normal and
expected — they can come from library version differences (scikit-learn,
gensim), floating-point differences across CPU architectures, or which
exact hyperparameter combinations `RandomizedSearchCV` happened to sample.
The `RANDOM_STATE = 42` setting minimizes this but cannot eliminate every
source of variation.

---

## Dataset details

The dataset is the "Emotion" text-classification dataset: 20,000 short,
informal English sentences (originally sourced from Twitter-style text),
each labelled with one of 6 emotions. Files are plain text, one example per
line, `<sentence>;<emotion label>` — already split into `train.txt` (16,000
lines), `val.txt` (2,000 lines), and `test.txt` (2,000 lines).

Class distribution is imbalanced (`joy` and `sadness` are far more common
than `surprise`), which is exactly why the pipeline reports **weighted
F1-score** as the primary comparison metric rather than plain accuracy —
see the notebook's Step 6 markdown cell for a full explanation of why that
matters here.
