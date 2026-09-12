# ═══════════════════════════════════════════════════════════════════
# movie_rnn/components/model_trainer.py
# ═══════════════════════════════════════════════════════════════════
# numpy arrays → SimpleRNN train → metrics → MLflow → model save
#
# CAR PRICE SE FARQ:
# Car Price → GridSearchCV (KerasRegressor) → best params → Dense ANN
# YAHAN     → Fixed architecture → SimpleRNN train → EarlyStopping
#             GridSearchCV nahi — scikeras CPU hang karta tha
#             Car Price mein bhi same issue tha — isliye hardcoded
#
# FLOW:
# DataTransformationArtifact (X_train_padded.npy, X_test_padded.npy)
# DataIngestionArtifact      (y_train.npy, y_test.npy, word_index.pkl)
#       ↓ load_numpy_array() + load_object()
# X_train (25000×500), y_train (25000,)
#       ↓ build_model() → Embedding → SimpleRNN → Dense(sigmoid)
# compiled keras model
#       ↓ model.fit() + EarlyStopping
# trained model
#       ↓ get_classification_metrics() × 2 (train + test)
# train_metrics, test_metrics
#       ↓ track_mlflow()
# DagsHub experiment logged
#       ↓ final_model/ save
#       ↓ push_model_to_huggingface()
# ModelTrainerArtifact
# ═══════════════════════════════════════════════════════════════════

import os
import sys
import logging
import numpy as np
import mlflow
import dagshub
import dill
from dotenv import load_dotenv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping

from movie_rnn.entity.config_entity import ModelTrainerConfig
from movie_rnn.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from movie_rnn.utils.main_utils.utils import (
    load_numpy_array,
    load_object,
)
from movie_rnn.utils.dl_utils.metric.classification_metric import get_classification_metrics
from movie_rnn.cloud.hf_syncer import push_model_to_huggingface
from movie_rnn.exception.exception import MovieSentimentException
from movie_rnn.constants.training_pipeline import HF_MODEL_DIR

load_dotenv()
# .env se DAGSHUB creds + HF_TOKEN load hoga


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        """
        Parameters:
            model_trainer_config          : config_entity.py se
                ├── trained_model_path
                ├── expected_accuracy = 0.80
                ├── embedding_dim     = 128
                ├── rnn_units         = 128
                ├── epochs            = 10
                ├── batch_size        = 32
                ├── validation_split  = 0.2
                └── patience          = 5

            data_ingestion_artifact : DataIngestion ka output
                ├── y_train_path
                ├── y_test_path
                └── word_index_path

            data_transformation_artifact : DataTransformation ka output
                ├── x_train_padded_path → X_train_padded.npy
                └── x_test_padded_path  → X_test_padded.npy
        """
        try:
            self.config                  = model_trainer_config
            self.ingestion_artifact      = data_ingestion_artifact
            self.transformation_artifact = data_transformation_artifact
            logging.info("ModelTrainer initialized")
        except Exception as e:
            raise MovieSentimentException(e, sys)





    def track_mlflow(
        self,
        model,
        train_metrics,
        test_metrics,
    ) -> None:
        """
        MLflow mein metrics log karta hai → DagsHub pe sync hoga

        IMP: dagshub.init() ANDAR hai — module level pe nahi
             Car Price mein yahi bug tha:
             module level pe → import hote hi auth maangega
             → Docker startup crash
        """
        try:
            dagshub.init(
                repo_owner = "alyan-khattak",
                repo_name  = "Movie-Sentiment-RNN",
                mlflow     = True
            )
            # DRY RUN:
            # repo_owner = "alyan-khattak"
            # repo_name  = "movie-sentiment-rnn"

            mlflow.set_experiment("MovieSentimentRNN")

            with mlflow.start_run():

                # ── LOG PARAMS ────────────────────────────────────
                mlflow.log_params({
                    "embedding_dim"    : self.config.embedding_dim,
                    "rnn_units"        : self.config.rnn_units,
                    "epochs"           : self.config.epochs,
                    "batch_size"       : self.config.batch_size,
                    "patience"         : self.config.patience,
                    "max_len"          : 500,
                    "max_features"     : 10000,
                    "validation_split" : self.config.validation_split,
                })

                # ── LOG TRAIN METRICS ─────────────────────────────
                mlflow.log_metrics({
                    "train_accuracy"  : train_metrics.accuracy,
                    "train_precision" : train_metrics.precision,
                    "train_recall"    : train_metrics.recall,
                    "train_f1"        : train_metrics.f1_score,
                })

                # ── LOG TEST METRICS ──────────────────────────────
                mlflow.log_metrics({
                    "test_accuracy"   : test_metrics.accuracy,
                    "test_precision"  : test_metrics.precision,
                    "test_recall"     : test_metrics.recall,
                    "test_f1"         : test_metrics.f1_score,
                })

                # ── LOG MODEL ─────────────────────────────────────
                mlflow.keras.log_model(model, "model")

            logging.info(
                f"MLflow logged | "
                f"train_acc: {train_metrics.accuracy:.4f} | "
                f"test_acc: {test_metrics.accuracy:.4f}"
            )

        except Exception as e:
            raise MovieSentimentException(e, sys)





    def build_model(self) -> keras.Model:
        """
        SimpleRNN architecture banata hai — exactly notebook jaisi

        ARCHITECTURE:
        Input (500,)
              ↓
        Embedding(10000, 128, input_length=500)
              ↓  har word → 128 dim vector | output: (500, 128)
        SimpleRNN(128, activation='relu')
              ↓  sequence compress | output: (128,)
        Dense(1, activation='sigmoid')
              ↓  output: (1,) → 0.0 to 1.0 probability

        IMP: sigmoid → binary classification
             > 0.5 → Positive | < 0.5 → Negative
        """
        try:
            model = Sequential([
                Embedding(
                    input_dim    = 10000,
                    # IMP: input_dim = max_features = 10000 (vocab size)
                    output_dim   = self.config.embedding_dim,
                    # → 128
                    input_length = 500,
                    # → max_len = 500
                ),
                SimpleRNN(
                    self.config.rnn_units,
                    # → 128
                    activation = "relu"
                    # notebook se liya — relu on RNN
                ),
                Dense(1, activation="sigmoid"),
                # sigmoid → 0 to 1 probability
                # binary_crossentropy ke saath use hota hai
            ])

            model.compile(
                optimizer = "adam",
                loss      = "binary_crossentropy",
                metrics   = ["accuracy"]
            )

            model.summary()
            logging.info(f"Model Build Successfully ")
            return model

        except Exception as e:
            raise MovieSentimentException(e, sys)




    def train_model(
        self,
        X_train, y_train,
        X_test,  y_test,
        word_index: dict,
    ) -> ModelTrainerArtifact:
        """
        Full training pipeline:
        1. model build karo
        2. EarlyStopping se train karo
        3. metrics calculate karo
        4. MLflow log karo
        5. model save karo (Artifacts/ + final_model/)
        6. HuggingFace push karo

        Returns:
            ModelTrainerArtifact
        """
        try:
            # ════════════════════════════════════════════════════
            # HYPERPARAMETERS
            # → GridSearchCV nahi — CPU hang karta hai (scikeras)
            # → Car Price mein bhi same issue tha — hardcoded rakha
            # → GPU wale system pe neeche wala uncomment kar sakte hain
            # ════════════════════════════════════════════════════

            # ── GridSearch (GPU wale uncomment karein) ────────────
            # best_params    = self.get_best_params(X_train, y_train)
            # best_epochs    = best_params.get("epochs",     10)
            # IMP: scikeras prefix "model__" lagata hai params pe

            # ── HardCoded Params (CPU ke liye) ────────────────────
            best_epochs = self.config.epochs
            # → 10 (constants se — hardcode nahi)

            logging.info(
                f"Using — embedding: {self.config.embedding_dim} | "
                f"rnn_units: {self.config.rnn_units} | "
                f"epochs: {best_epochs}"
            )

            # ── STEP 1: model banao ───────────────────────────────
            model = self.build_model()

            # ── STEP 2: EarlyStopping callback ────────────────────
            early_stopping = EarlyStopping(
                monitor              = "val_loss",
                patience             = self.config.patience,
                # patience=5 → 5 epochs mein val_loss improve nahi → stop
                restore_best_weights = True,
                # True → best epoch ka model milega — not last epoch
                # IMP: True rakho — warna overfit epoch ka model milega
            )

            # ── STEP 3: Train ─────────────────────────────────────
            logging.info("Training started...")
            history = model.fit(
                X_train, y_train,
                epochs           = best_epochs,
                batch_size       = self.config.batch_size,
                validation_split = self.config.validation_split,
                # IMP: validation_split — train data ka 20% val ke liye
                #      Car Price mein validation_data=(X_test,y_test) tha
                #      Yahan split use kiya — test data unseen rakho
                callbacks        = [early_stopping],
                verbose          = 1,
            )
            logging.info(
                f"Training complete | "
                f"Epochs run: {len(history.history['loss'])}"
            )

            # ── STEP 4: Save model (Artifacts/) ──────────────────
            model_dir = os.path.dirname(self.config.trained_model_path)
            os.makedirs(model_dir, exist_ok=True)
            model.save(self.config.trained_model_path)
            # → "Artifacts/timestamp/model_trainer/trained_model/model.keras"
            logging.info(f"Model saved: {self.config.trained_model_path}")

            # ── STEP 5: Metrics calculate karo ────────────────────
            y_train_prob = model.predict(X_train, verbose=0)
            y_test_prob  = model.predict(X_test,  verbose=0)
            # y_train_prob → [[0.87], [0.23], ...] shape: (25000, 1)

            y_train_pred = (y_train_prob > 0.5).astype(int).flatten()
            y_test_pred  = (y_test_prob  > 0.5).astype(int).flatten()
            # > 0.5       → [[True], [False], ...]
            # astype(int) → [[1],    [0],     ...]
            # flatten()   → [1, 0, ...]  shape: (25000,)


            # These Methods are defined in utls/dl_utls/classification_metric
            train_metrics = get_classification_metrics(y_train, y_train_pred)
            test_metrics  = get_classification_metrics(y_test,  y_test_pred)

            logging.info(
                f"Train → acc: {train_metrics.accuracy:.4f} | "
                f"f1: {train_metrics.f1_score:.4f}"
            )
            logging.info(
                f"Test  → acc: {test_metrics.accuracy:.4f} | "
                f"f1: {test_metrics.f1_score:.4f}"
            )

            # ── STEP 6: Accuracy threshold check ──────────────────
            if test_metrics.accuracy < self.config.expected_accuracy:
                raise Exception(
                    f"Model rejected — "
                    f"test accuracy {test_metrics.accuracy:.4f} "
                    f"< expected {self.config.expected_accuracy}"
                )
            # IMP: 0.80 threshold
            # Car Price mein expected R² = 0.80 tha — same concept

            # ── STEP 7: MLflow track karo ─────────────────────────
            self.track_mlflow(model, train_metrics, test_metrics)

            # ── STEP 8: final_model/ mein save ────────────────────
            os.makedirs(HF_MODEL_DIR, exist_ok=True)
            # HF_MODEL_DIR = "final_model"

            model.save(os.path.join(HF_MODEL_DIR, "model.keras"))
            # → "final_model/model.keras"

            wi_path = os.path.join(HF_MODEL_DIR, "word_index.pkl")
            with open(wi_path, "wb") as f:
                dill.dump(word_index, f)
            # IMP: dill.dump(OBJECT, FILE) — argument order
            # → "final_model/word_index.pkl"

            logging.info("final_model/ saved")

            # ── STEP 9: HuggingFace push ──────────────────────────
            push_model_to_huggingface()
            # hf_syncer.py se — upload_folder("final_model/") → HF

            # ── STEP 10: Artifact banao ───────────────────────────
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_path = self.config.trained_model_path,
                train_accuracy     = train_metrics.accuracy,
                test_accuracy      = test_metrics.accuracy,
                is_model_accepted  = True,
            )

            logging.info(f"ModelTrainer completed: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise MovieSentimentException(e, sys)


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        ModelTrainer ka main entry point.

        Returns:
            ModelTrainerArtifact
        """
        try:
            logging.info("=" * 55)
            logging.info("ModelTrainer started")
            logging.info("=" * 55)

            # ── STEP 1: padded arrays load karo ──────────────────
            X_train = load_numpy_array(
                self.transformation_artifact.x_train_padded_path
            )
            X_test  = load_numpy_array(
                self.transformation_artifact.x_test_padded_path
            )
            logging.info(
                f"Arrays loaded — "
                f"X_train: {X_train.shape} | X_test: {X_test.shape}"
            )
            # DRY RUN:
            # X_train.shape → (25000, 500)
            # X_test.shape  → (25000, 500)

            # ── STEP 2: labels + word_index load karo ────────────
            y_train    = load_numpy_array(self.ingestion_artifact.y_train_path)
            y_test     = load_numpy_array(self.ingestion_artifact.y_test_path)
            word_index = load_object(self.ingestion_artifact.word_index_path)
            # y_train.shape → (25000,) — 0/1 values
            # y_test.shape  → (25000,) — 0/1 values
            # word_index    → dict, len=88584

            logging.info(
                f"y_train: {y_train.shape} | "
                f"y_test: {y_test.shape} | "
                f"vocab: {len(word_index)}"
            )

            # ── STEP 3: Train ─────────────────────────────────────
            return self.train_model(
                X_train, y_train,
                X_test,  y_test,
                word_index,
            )

        except Exception as e:
            raise MovieSentimentException(e, sys)


# ─────────────────────────────────────────────────────────────────
# DRY RUN — full flow
#
# trainer  = ModelTrainer(config, ingestion_artifact, transformation_artifact)
# artifact = trainer.initiate_model_trainer()
#
# Training logs:
#   Epoch 1/10 → loss: 0.6821 — accuracy: 0.5823
#   Epoch 5/10 → loss: 0.3421 — accuracy: 0.8612
#   EarlyStopping at epoch 8 (val_loss stopped improving)
#
# artifact.trained_model_path → "Artifacts/.../trained_model/model.keras"
# artifact.train_accuracy     → 0.9416
# artifact.test_accuracy      → 0.8623
# artifact.is_model_accepted  → True
# ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────
# COMPARISON — Car Price vs Movie Sentiment
#
# ┌──────────────────┬──────────────────────┬─────────────────────┐
# │ Feature          │ Car Price            │ Movie Sentiment      │
# ├──────────────────┼──────────────────────┼─────────────────────┤
# │ Architecture     │ Dense ANN            │ Embedding+SimpleRNN │
# │ Param search     │ GridSearchCV         │ Hardcoded (CPU)     │
# │                  │ (commented out CPU)  │ same reason         │
# │ Loss             │ mse (regression)     │ binary_crossentropy │
# │ Output layer     │ linear (no sigmoid)  │ sigmoid             │
# │ Metric           │ R² / MAE / RMSE      │ Accuracy / F1       │
# │ Threshold        │ R² >= 0.80           │ Accuracy >= 0.80    │
# │ val strategy     │ validation_data=test │ validation_split    │
# │ HF files         │ model.keras +        │ model.keras +       │
# │                  │ preprocessor.pkl     │ word_index.pkl      │
# │ dagshub.init()   │ inside track_mlflow  │ inside track_mlflow │
# │                  │ NOT module level     │ NOT module level    │
# └──────────────────┴──────────────────────┴─────────────────────┘
# ─────────────────────────────────────────────────────────────────