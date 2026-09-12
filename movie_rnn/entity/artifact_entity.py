# ═══════════════════════════════════════════════════════════════════
# movie_rnn/entity/artifact_entity.py
# ═══════════════════════════════════════════════════════════════════
# Har pipeline component ka OUTPUT (artifact) yahan define hota hai
#
# WHY ARTIFACT ENTITY?
# Car Price + Network Security mein same pattern:
#   Component kaam karta hai → typed object return karta hai
#   Next component usi object ko input leta hai
#   No bare tuples — typed dataclasses only
#
# PATTERN:
# DataIngestion.initiate() → DataIngestionArtifact return karta hai
#                                      ↓
#                            DataValidation config isko leta hai
#
# ABHI SIRF:
#   - DataIngestionArtifact
# Baaki artifacts baad mein add honge jab unka component banega
# ═══════════════════════════════════════════════════════════════════

from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════
# DataIngestionArtifact
# ══════════════════════════════════════════════════════════════════
# DataIngestion component yeh object return karta hai
# Iske andar saare output file paths hain

@dataclass
class DataIngestionArtifact:
    x_train_path: str
    # → "Artifacts/timestamp/data_ingestion/raw_data/X_train.npy"
    # padded nahi — raw integer sequences

    x_test_path: str
    # → "Artifacts/timestamp/data_ingestion/raw_data/X_test.npy"

    y_train_path: str
    # → "Artifacts/timestamp/data_ingestion/raw_data/y_train.npy"
    # 0 or 1 — Negative / Positive

    y_test_path: str
    # → "Artifacts/timestamp/data_ingestion/raw_data/y_test.npy"

    word_index_path: str
    # → "Artifacts/timestamp/data_ingestion/raw_data/word_index.pkl"
    # IMP: yeh path ModelTrainer tak travel karega
    #      wahan se HuggingFace pe push hoga




# ══════════════════════════════════════════════════════════════════
# DataValidationArtifact
# ═════════

@dataclass
class DataValidationArtifact:
    validation_status: bool
    # True  → sab checks pass → pipeline continue karo
    # False → kuch fail → pipeline rok do

    validation_report_path: str
    # → "Artifacts/timestamp/data_validation/validation_report.yaml"

    message: str
    # pass hone pe → "All validation checks passed"
    # fail hone pe → exactly kya fail hua


# ══════════════════════════════════════════════════════════════════
# DataTransformationArtifact
# ═════════

@dataclass
class DataTransformationArtifact:
    x_train_padded_path: str
    # → "Artifacts/.../transformed/X_train_padded.npy"
    # shape: (25000, 500) — ab fixed length

    x_test_padded_path: str
    # → "Artifacts/.../transformed/X_test_padded.npy"
    # shape: (25000, 500)




# ══════════════════════════════════════════════════════════════════
# Model Trainer 
# ═════════
@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
    # → "Artifacts/.../model_trainer/trained_model/model.keras"

    train_accuracy: float
    test_accuracy: float
    # IMP: test_accuracy >= 0.80 → accepted
    #      test_accuracy <  0.80 → pipeline fail

    is_model_accepted: bool
    # True  → HuggingFace push hoga
    # False → pipeline rukegi
# ─────────────────────────────────────────────────────────────────
# DRY RUN
#
# artifact = DataIngestioavnArtifact(
#     x_train_path   = "Artifacts/.../raw_data/X_train.npy",
#     x_test_path    = "Artifacts/.../raw_data/X_test.npy",
#     y_train_path   = "Artifacts/.../raw_data/y_train.npy",
#     y_test_path    = "Artifacts/.../raw_data/y_test.npy",
#     word_index_path = "Artifacts/.../raw_data/word_index.pkl",
# )
#
# next component lega:
# DataValidationConfig(training_pipeline_config, data_ingestion_artifact)
# → artifact.x_train_path se data padhega
# ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────
# COMPARISON — Car Price vs Movie Sentiment
#
# ┌──────────────────┬─────────────────────┬──────────────────────┐
# │ Field            │ Car Price           │ Movie Sentiment       │
# ├──────────────────┼─────────────────────┼──────────────────────┤
# │ train data       │ train.csv (path)    │ X_train.npy (path)   │
# │ test data        │ test.csv (path)     │ X_test.npy (path)    │
# │ labels           │ inside csv          │ y_train/y_test .npy  │
# │ extra file       │ —                   │ word_index.pkl       │
# │ format           │ .csv tabular        │ .npy sequences       │
# └──────────────────┴─────────────────────┴──────────────────────┘
# ─────────────────────────────────────────────────────────────────