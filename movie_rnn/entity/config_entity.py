# ═══════════════════════════════════════════════════════════════════
# movie_rnn/entity/config_entity.py
# ═══════════════════════════════════════════════════════════════════
# Har pipeline component ka CONFIG yahan define hota hai
#
# WHY ENTITY/CONFIG?
# Car Price + Network Security mein same pattern:
#   Config ALAG FILE mein — entity/config_entity.py
#   Component sirf config object leta hai — paths khud nahi banata
#   Separation of concerns → clean architecture
#
# PATTERN:
# constants/ → raw values (strings, ints, floats)
#      ↓
# config_entity.py → in values se paths banao (os.path.join)
#      ↓
# component → config object lo, kaam karo
#
# IMP: TrainingPipelineConfig → timestamp generate karta hai
#      sab doosre configs TrainingPipelineConfig se path lete hain
#      taaki sab ek hi timestamp folder mein jaayein
#
# ABHI SIRF:
#   - TrainingPipelineConfig
#   - DataIngestionConfig
# Baaki configs baad mein add honge jab unka component banega
# ═══════════════════════════════════════════════════════════════════

from datetime import datetime
import os
from movie_rnn.constants import training_pipeline


# ══════════════════════════════════════════════════════════════════
# CLASS 1: TrainingPipelineConfig
# ══════════════════════════════════════════════════════════════════
# Master config — timestamp generate karta hai
# Sab doosre configs isko inject karte hain
# Taaki sab ek hi timestamped run folder mein save hon

class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        """
        Pipeline level config — har run pe naya timestamp folder banata hai

        PATH STRUCTURE:
        Artifacts/
        └── 09_11_2026_14_32_00/     ← self.artifact_dir
        """

        # datetime object → string
        # %m=month %d=day %Y=year %H=hour %M=minute %S=second
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        # DRY RUN:
        # datetime(2026,9,11,14,32,0) → "09_11_2026_14_32_00"

        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        # → "MovieSentiment"

        self.artifact_name: str = training_pipeline.ARTIFACT_DIR
        # → "Artifacts"

        self.artifact_dir: str = os.path.join(self.artifact_name, timestamp)
        # os.path.join("Artifacts", "09_11_2026_14_32_00")
        # → "Artifacts/09_11_2026_14_32_00"
        # IMP: har run pe NAYA folder — history preserve hoti hai

        self.timestamp: str = timestamp
        # → "09_11_2026_14_32_00"


# ══════════════════════════════════════════════════════════════════
# CLASS 2: DataIngestionConfig
# ══════════════════════════════════════════════════════════════════
# DataIngestion component ke liye sab paths yahan define hain
# keras se data download karke kahan save karna hai → sab yahan

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        DataIngestion ke liye sab paths banata hai
        TrainingPipelineConfig inject hota hai — timestamp milta hai

        PATH STRUCTURE:
        Artifacts/
        └── 09_11_2026_14_32_00/              ← training_pipeline_config.artifact_dir
            └── data_ingestion/                ← DATA_INGESTION_DIR_NAME
                └── raw_data/                  ← DATA_INGESTION_RAW_DATA_DIR
                    ├── X_train.npy            ← IMDB_X_TRAIN_FILE
                    ├── X_test.npy             ← IMDB_X_TEST_FILE
                    ├── y_train.npy            ← IMDB_Y_TRAIN_FILE
                    ├── y_test.npy             ← IMDB_Y_TEST_FILE
                    └── word_index.pkl         ← IMDB_WORD_INDEX_FILE
        """

        # ── BASE DIR ─────────────────────────────────────────────
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,       # "Artifacts/timestamp"
            training_pipeline.DATA_INGESTION_DIR_NAME    # "data_ingestion"
        )
        # → "Artifacts/09_11_2026_14_32_00/data_ingestion"

        # ── RAW DATA DIR ─────────────────────────────────────────
        self.raw_data_dir: str = os.path.join(
            self.data_ingestion_dir,                     # base dir
            training_pipeline.DATA_INGESTION_RAW_DATA_DIR  # "raw_data"
        )
        # → "Artifacts/09_11_2026_14_32_00/data_ingestion/raw_data"

        # ── ARRAY FILE PATHS ─────────────────────────────────────
        self.x_train_path: str = os.path.join(
            self.raw_data_dir, training_pipeline.IMDB_X_TRAIN_FILE
        )
        # → ".../raw_data/X_train.npy"

        self.x_test_path: str = os.path.join(
            self.raw_data_dir, training_pipeline.IMDB_X_TEST_FILE
        )
        # → ".../raw_data/X_test.npy"

        self.y_train_path: str = os.path.join(
            self.raw_data_dir, training_pipeline.IMDB_Y_TRAIN_FILE
        )
        # → ".../raw_data/y_train.npy"

        self.y_test_path: str = os.path.join(
            self.raw_data_dir, training_pipeline.IMDB_Y_TEST_FILE
        )
        # → ".../raw_data/y_test.npy"

        # ── WORD INDEX PATH ──────────────────────────────────────
        self.word_index_path: str = os.path.join(
            self.raw_data_dir, training_pipeline.IMDB_WORD_INDEX_FILE
        )
        # → ".../raw_data/word_index.pkl"
        # IMP: word_index yahan save hoga → baad mein HuggingFace pe bhi jayega
        #      prediction time pe text → integers convert karne ke liye zaroori

        # ── DATASET CONFIG ───────────────────────────────────────
        self.max_features: int = training_pipeline.IMDB_MAX_FEATURES
        # → 10000 — top 10k words only





class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        PATH STRUCTURE:
        Artifacts/
        └── timestamp/
            └── data_validation/
                └── validation_report.yaml
        """

        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME
        )
        # → "Artifacts/timestamp/data_validation"

        self.validation_report_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_REPORT_FILE_NAME
        )
        # → "Artifacts/timestamp/data_validation/validation_report.yaml"

        self.expected_train_samples: int = training_pipeline.DATA_VALIDATION_EXPECTED_TRAIN_SAMPLES
        # → 25000

        self.expected_test_samples: int = training_pipeline.DATA_VALIDATION_EXPECTED_TEST_SAMPLES
        # → 25000

        self.expected_max_index: int = training_pipeline.DATA_VALIDATION_EXPECTED_MAX_INDEX
        # → 10000
  





class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        PATH STRUCTURE:
        Artifacts/
        └── timestamp/
            └── data_transformation/
                └── transformed/
                    ├── X_train_padded.npy
                    └── X_test_padded.npy
        """

        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )
        # → "Artifacts/timestamp/data_transformation"

        self.transformed_dir: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DIR
        )
        # → "Artifacts/timestamp/data_transformation/transformed"

        self.x_train_padded_path: str = os.path.join(
            self.transformed_dir, "X_train_padded.npy"
        )
        # → ".../transformed/X_train_padded.npy"

        self.x_test_padded_path: str = os.path.join(
            self.transformed_dir, "X_test_padded.npy"
        )
        # → ".../transformed/X_test_padded.npy"

        self.max_len: int = training_pipeline.IMDB_MAX_LEN
        # → 
        


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        PATH STRUCTURE:
        Artifacts/
        └── timestamp/
            └── model_trainer/
                └── trained_model/
                    └── model.keras
        """

        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        # → "Artifacts/timestamp/model_trainer"

        self.trained_model_path: str = os.path.join(
            self.model_trainer_dir,
            training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
            training_pipeline.MODEL_TRAINER_MODEL_FILE_NAME
        )
        # → "Artifacts/timestamp/model_trainer/trained_model/model.keras"

        self.embedding_dim: int   = training_pipeline.MODEL_TRAINER_EMBEDDING_DIM
        self.rnn_units: int       = training_pipeline.MODEL_TRAINER_RNN_UNITS
        self.epochs: int          = training_pipeline.MODEL_TRAINER_EPOCHS
        self.batch_size: int      = training_pipeline.MODEL_TRAINER_BATCH_SIZE
        self.validation_split: float = training_pipeline.MODEL_TRAINER_VALIDATION_SPLIT
        self.patience: int        = training_pipeline.MODEL_TRAINER_EARLY_STOPPING_PATIENCE
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_ACCURACY
# ─────────────────────────────────────────────────────────────────
# DRY RUN
#
# pipeline_config = TrainingPipelineConfig()
# → artifact_dir = "Artifacts/09_11_2026_14_32_00"
#
# ingestion_config = DataIngestionConfig(pipeline_config)
# → data_ingestion_dir = "Artifacts/09_11_2026_14_32_00/data_ingestion"
# → raw_data_dir       = "Artifacts/.../data_ingestion/raw_data"
# → x_train_path       = "Artifacts/.../raw_data/X_train.npy"
# → x_test_path        = "Artifacts/.../raw_data/X_test.npy"
# → y_train_path       = "Artifacts/.../raw_data/y_train.npy"
# → y_test_path        = "Artifacts/.../raw_data/y_test.npy"
# → word_index_path    = "Artifacts/.../raw_data/word_index.pkl"
# → max_features       = 10000
# ─────────────────────────────────────────────────────────────────

