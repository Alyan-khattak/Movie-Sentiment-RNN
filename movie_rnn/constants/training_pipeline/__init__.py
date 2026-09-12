# ═══════════════════════════════════════════════════════════════════
# movie_rnn/constants/training_pipeline/__init__.py
# ═══════════════════════════════════════════════════════════════════
# Poore project mein use hone wale SAB constants yahan hain
# Koi bhi hardcoded value kisi bhi file mein nahi hogi — sab yahan se aayegi
#
# WHY CONSTANTS FOLDER?
# Car Price + Network Security projects mein same pattern tha:
#   Sab values EK JAGAH — constants/training_pipeline/__init__.py
#   Koi bhi file import karke use kare:
#   "from movie_rnn.constants import training_pipeline"
#   Path change karna ho → sirf yahan aao → poora project update
#
# IMP: __init__.py isliye — taaki yeh folder ek Python package ban jaata hai
#      "from movie_rnn.constants import training_pipeline" kaam kare
# ═══════════════════════════════════════════════════════════════════

import os

# ─────────────────────────────────────────────────────────────────
# COMMON CONSTANTS — poore pipeline mein use honge
# ─────────────────────────────────────────────────────────────────

TARGET_COLUMN = "sentiment"
# predict karna hai yeh column
# 1 = Positive, 0 = Negative

PIPELINE_NAME = "MovieSentiment"
# pipeline ka naam — artifacts folder structure mein use hoga

ARTIFACT_DIR = "Artifacts"
# IMP: root folder — sab pipeline outputs yahan save honge
# timestamped subfolders honge:
# Artifacts/
# └── 09_11_2026_14_32_00/   ← har run ka alag folder
#     ├── data_ingestion/
#     ├── data_validation/
#     ├── data_transformation/
#     └── model_trainer/
#



# ─────────────────────────────────────────────────────────────────
# DATA INGESTION CONSTANTS
# prefix: DATA_INGESTION_ → easily identify karo kahan use hoga
# ─────────────────────────────────────────────────────────────────

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
# artifact subfolder naam
# path banega: Artifacts/timestamp/data_ingestion/

DATA_INGESTION_RAW_DATA_DIR: str = "raw_data"
# raw downloaded data yahan save hoga
# path: Artifacts/timestamp/data_ingestion/raw_data/
# IMP: raw_data = backup before any processing

DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
# 80% train, 20% test
# IMP: imdb dataset already split hai — yeh ratio
#      sirf future custom datasets ke liye reserve hai
#      abhi keras ki built-in split use hogi (25k/25k)

# ─────────────────────────────────────────────────────────────────
# IMDB DATASET CONSTANTS
# ─────────────────────────────────────────────────────────────────

IMDB_MAX_FEATURES: int = 10000
# vocabulary size — sirf top 10000 frequent words load honge
# notebook se liya: max_features = 10000
# baaki sab words → unknown token (index 2)

IMDB_WORD_INDEX_FILE: str = "word_index.pkl"
# word → integer mapping save hogi yahan
# IMP: yeh file model ke saath HuggingFace pe bhi jayegi
#      prediction time pe text → integers ke liye zaroori hai

IMDB_X_TRAIN_FILE: str = "X_train.npy"
IMDB_X_TEST_FILE:  str = "X_test.npy"
IMDB_Y_TRAIN_FILE: str = "y_train.npy"
IMDB_Y_TEST_FILE:  str = "y_test.npy"
# raw (unpadded) arrays yahan save honge
# .npy format → numpy ka native format → fast load/save
# IMP: Car Price mein .csv tha kyunki tabular data tha
#      Yahan sequences hain → .npy zyada suitable hai




# ─────────────────────────────────────────────────────────────────
# DATA VALIDATION CONSTANTS
# prefix: DATA_VALIDATION_
# ─────────────────────────────────────────────────────────────────

DATA_VALIDATION_DIR_NAME: str = "data_validation"
# path: Artifacts/timestamp/data_validation/

DATA_VALIDATION_REPORT_FILE_NAME: str = "validation_report.yaml"
# validation results yahan save honge
# path: Artifacts/timestamp/data_validation/validation_report.yaml

# expected shapes — agar match nahi kiya → pipeline rok do
DATA_VALIDATION_EXPECTED_TRAIN_SAMPLES: int = 25000
DATA_VALIDATION_EXPECTED_TEST_SAMPLES:  int = 25000
DATA_VALIDATION_EXPECTED_MAX_INDEX:     int = 10000
# IMP: har sequence mein koi bhi index 10000 se zyada nahi hona chahiye
#      kyunki max_features = 10000 set kiya tha load_data mein



# ─────────────────────────────────────────────────────────────────
# DATA TRANSFORMATION CONSTANTS
# prefix: DATA_TRANSFORMATION_
# ─────────────────────────────────────────────────────────────────

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
# path: Artifacts/timestamp/data_transformation/

DATA_TRANSFORMATION_TRANSFORMED_DIR: str = "transformed"
# padded arrays yahan save honge

IMDB_MAX_LEN: int = 500
# har sequence ko 500 tokens pe pad/truncate karo
# notebook se liya: max_len = 500
# chhota review → zeros se pad
# lamba review  → 500 pe truncate





# ─────────────────────────────────────────────────────────────────
# MODEL TRAINER CONSTANTS
# prefix: MODEL_TRAINER_
# ─────────────────────────────────────────────────────────────────

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
# path: Artifacts/timestamp/model_trainer/

MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_MODEL_FILE_NAME: str = "model.keras"
# path: Artifacts/timestamp/model_trainer/trained_model/model.keras

# ── ANN ARCHITECTURE ─────────────────────────────────────────────
MODEL_TRAINER_EMBEDDING_DIM: int = 128
# Embedding layer output size — notebook se liya
# 10000 words → each word = 128 dimensional vector

MODEL_TRAINER_RNN_UNITS: int = 128
# SimpleRNN units — notebook se liya

MODEL_TRAINER_EPOCHS: int = 10
# notebook se liya

MODEL_TRAINER_BATCH_SIZE: int = 32
#  32 standard hai, faster training

MODEL_TRAINER_VALIDATION_SPLIT: float = 0.2
# 20% train data → validation ke liye

# ── EARLY STOPPING ───────────────────────────────────────────────
MODEL_TRAINER_EARLY_STOPPING_PATIENCE: int = 5
# notebook se liya — 5 epochs improvement nahi → stop

# ── EXPECTED PERFORMANCE ─────────────────────────────────────────
MODEL_TRAINER_EXPECTED_ACCURACY: float = 0.80
# IMP: agar test accuracy < 0.80 → model reject karo
#      Car Price mein expected R² = 0.80 tha — same concept