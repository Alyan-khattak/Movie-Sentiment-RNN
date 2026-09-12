# 🎬 Movie Sentiment RNN

> End-to-end MLOps pipeline for IMDB movie review sentiment analysis using Simple RNN.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-blue)](https://mlflow.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-model%20registry-yellow)](https://huggingface.co)

---

## What This Project Does

Given any movie review text, the model predicts whether the sentiment is **Positive** or **Negative** with a confidence score.

```
Input  : "This film is an absolute masterpiece. The performances are breathtaking."
Output : Positive — 91.3% confidence
```

This is not just a notebook experiment. It is a full production-grade MLOps pipeline following the same architecture as industry-standard systems — with experiment tracking, model registry, versioned artifacts, and one-click retraining via a REST API.

---

## Results

| Metric         | Value  |
|----------------|--------|
| Train Accuracy | 91.25% |
| Test Accuracy  | 80.96% |
| Dataset        | IMDB (50,000 reviews) |
| Vocabulary     | 10,000 words |
| Token Window   | 500 tokens |

---

## Model Architecture

```
Input (500,)
      ↓
Embedding(input_dim=10000, output_dim=128, input_length=500)
      ↓  each word → 128-dimensional vector
SimpleRNN(128, activation='relu')
      ↓  compress sequence → single 128-dim vector
Dense(1, activation='sigmoid')
      ↓
Output: probability 0.0 → 1.0
        > 0.5 → Positive
        < 0.5 → Negative
```

**Training config:**
- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 10 (EarlyStopping patience=5)
- Batch size: 32
- Validation split: 20%

---

## Project Structure

```
Movie-Sentiment-RNN/
│
├── app.py                              ← FastAPI app (lifespan pulls model from HuggingFace)
├── main.py                             ← Training pipeline entry point
├── setup.py
├── requirements.txt
├── requirements-prod.txt               ← Pinned versions for Docker
├── Dockerfile
├── .dockerignore
├── .env                                ← (not committed — create your own)
│
├── .github/workflows/
│   └── main.yml                        ← GitHub Actions CI/CD
│
├── movie_rnn/
│   ├── constants/
│   │   └── training_pipeline/
│   │       └── __init__.py             ← ALL constants in one place
│   │
│   ├── entity/
│   │   ├── config_entity.py            ← 5 config dataclasses
│   │   └── artifact_entity.py          ← 5 artifact dataclasses
│   │
│   ├── components/
│   │   ├── data_ingestion.py           ← keras.datasets.imdb → .npy arrays
│   │   ├── data_validation.py          ← shape + vocab + label checks
│   │   ├── data_transformation.py      ← pad_sequences(maxlen=500)
│   │   └── model_trainer.py            ← train → MLflow → HuggingFace push
│   │
│   ├── pipeline/
│   │   └── training_pipeline.py        ← orchestrates all 4 components
│   │
│   ├── utils/
│   │   ├── main_utils/utils.py         ← save/load numpy + pickle helpers
│   │   └── dl_utils/
│   │       ├── model/estimator.py      ← IMDBSentimentModel wrapper
│   │       └── metric/classification_metric.py
│   │
│   ├── cloud/
│   │   └── hf_syncer.py               ← HuggingFace push / pull
│   │
│   ├── exception/
│   └── logging/
│
├── data_schema/
│   └── schema.yaml
│
├── templates/
│   ├── index.html                      ← Landing page
│   ├── predict.html                    ← Review input form
│   └── result.html                     ← Sentiment result display
│
└── Artifacts/                          ← Timestamped pipeline runs (gitignored)
    └── MM_DD_YYYY_HH_MM_SS/
        ├── data_ingestion/
        ├── data_validation/
        ├── data_transformation/
        └── model_trainer/
```

---

## Pipeline — 4 Stages

### Stage 1: Data Ingestion
Loads the IMDB dataset directly from `keras.datasets.imdb`. Saves raw integer-encoded sequences as `.npy` arrays and the word index dictionary as `word_index.pkl`.

```
keras.datasets.imdb.load_data(num_words=10000)
      ↓
X_train.npy  X_test.npy  (25000 samples each, variable length)
y_train.npy  y_test.npy  (0 = Negative, 1 = Positive)
word_index.pkl            (vocabulary: 88k words)
```

### Stage 2: Data Validation
Validates the loaded data before transformation. Checks sample counts, label integrity, vocabulary index range, and word index loading.

```
Checks:
  ✓ X_train.shape[0] == 25000
  ✓ X_test.shape[0]  == 25000
  ✓ Labels ∈ {0, 1}
  ✓ Max vocab index  <= 10000
  ✓ word_index loaded correctly
      ↓
validation_report.yaml
```

### Stage 3: Data Transformation
Applies `pad_sequences(maxlen=500)` to convert variable-length sequences into fixed-length arrays the RNN can process.

```
X_train (25000,) variable length  →  X_train_padded (25000, 500)
X_test  (25000,) variable length  →  X_test_padded  (25000, 500)

Short reviews → zero-padded at the front
Long reviews  → truncated to last 500 tokens
```

### Stage 4: Model Trainer
Builds and trains the SimpleRNN, evaluates metrics, logs to MLflow on DagsHub, saves locally, and pushes to HuggingFace Hub.

```
build_model()
      ↓
model.fit() + EarlyStopping(patience=5)
      ↓
get_classification_metrics() → accuracy, precision, recall, F1
      ↓
MLflow → DagsHub (params + metrics + model)
      ↓
final_model/model.keras + final_model/word_index.pkl
      ↓
HuggingFace Hub push
```

---

## Prediction Flow (at Inference Time)

```python
text = "This film is an absolute masterpiece"
      ↓
text.lower().split()
→ ["this", "film", "is", "an", "absolute", "masterpiece"]
      ↓
word_index.get(word, 2) + 3   # +3 offset for reserved tokens
→ [15, 48, 7, 23, 312, 891]
      ↓
pad_sequences([encoded], maxlen=500)
→ shape (1, 500)
      ↓
model.predict()
→ [[0.913]]
      ↓
score = 0.913
label = "Positive"
confidence = 91.3%
```

---

## Setup & Run Locally

### 1. Clone and create environment

```bash
git clone https://github.com/Alyan-khattak/Movie-Sentiment-RNN.git
cd Movie-Sentiment-RNN

conda create -n movie-rnn python=3.10 -y
conda activate movie-rnn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Create `.env` file

```bash
DAGSHUB_REPO_OWNER=your-dagshub-username
DAGSHUB_REPO_NAME=Movie-Sentiment-RNN
MLFLOW_TRACKING_URI=https://dagshub.com/your-username/Movie-Sentiment-RNN.mlflow
MLFLOW_TRACKING_USERNAME=your-dagshub-username
MLFLOW_TRACKING_PASSWORD=your-dagshub-token
HF_TOKEN=your-huggingface-token
HF_REPO_ID=your-hf-username/movie-sentiment-rnn
```

### 4. Run the training pipeline

```bash
python main.py
```

This runs all 4 stages and pushes the trained model to HuggingFace.

### 5. Start the FastAPI app

```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Visit `http://localhost:8080`

---

## Run with Docker

```bash
# Build
docker build -t movie-sentiment-rnn .

# Run
docker run -p 8080:8080 --env-file .env movie-sentiment-rnn
```

---

## API Routes

| Method | Route      | Description                        |
|--------|------------|------------------------------------|
| GET    | `/`        | Landing page                       |
| GET    | `/predict` | Review input form                  |
| POST   | `/predict` | Run inference, return result page  |
| GET    | `/train`   | Trigger full retraining pipeline   |

---

## MLflow Tracking

All experiments are tracked on DagsHub. Each training run logs:

- **Parameters:** embedding_dim, rnn_units, epochs, batch_size, patience, max_len, max_features
- **Metrics:** train/test accuracy, precision, recall, F1
- **Model:** keras model artifact

---

## Tech Stack

| Component        | Technology                    |
|------------------|-------------------------------|
| Model            | TensorFlow / Keras SimpleRNN  |
| API              | FastAPI + Uvicorn             |
| Experiment Tracking | MLflow + DagsHub           |
| Model Registry   | HuggingFace Hub               |
| Containerization | Docker                        |
| CI/CD            | GitHub Actions                |
| Serialization    | dill + numpy                  |
| Templates        | Jinja2                        |

---

## Key Engineering Decisions

**Why `.npy` instead of `.csv`?**
IMDB data is sequences of integers, not tabular data. NumPy's native format is faster and more appropriate for sequence data.

**Why save `word_index.pkl` separately?**
The model only understands integers. At prediction time, raw text must be converted to integers using the same vocabulary that was used during training. Without `word_index.pkl`, the model cannot make predictions.

**Why `+3` offset during prediction?**
IMDB reserves the first 3 indices: `0` = padding, `1` = start token, `2` = unknown word. All actual word indices are shifted by +3 to avoid collision.

**Why not GridSearchCV for hyperparameter tuning?**
The `scikeras` wrapper used for GridSearchCV with Keras models causes CPU hang on machines without GPU (all cores pegged, no output). Hardcoded best parameters are used instead: `embedding_dim=128`, `rnn_units=128`, `epochs=10`.

**Why EarlyStopping with `restore_best_weights=True`?**
Prevents saving the last epoch's weights (which may be worse than an earlier epoch). The model from the epoch with the best `val_loss` is kept.

---

## Repository

GitHub: [Alyan-khattak/Movie-Sentiment-RNN](https://github.com/Alyan-khattak/Movie-Sentiment-RNN)

---

*Built as part of an end-to-end MLOps learning journey — same production architecture as Network Security MLOps and Car Price Predictor ANN.*