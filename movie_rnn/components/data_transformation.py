# ═══════════════════════════════════════════════════════════════════
# movie_rnn/components/data_transformation.py
# ═══════════════════════════════════════════════════════════════════
# Pipeline ka TEESRA component — sequences ko pad karta hai
#
# WHY DATA TRANSFORMATION?
# Car Price mein:
#   ColumnTransformer → StandardScaler + OHE → .npy arrays
# Movie Sentiment mein:
#   pad_sequences(maxlen=500) → fixed length arrays
#
# IMP: Yahan sirf X_train aur X_test pad hote hain
#      y_train aur y_test already integers hain → kuch nahi karna
#      word_index bhi nahi chhedna → woh as-is ModelTrainer tak jayega
#
# WHY PADDING?
# RNN ko fixed length input chahiye
# Har review ki alag length hai:
#   review 1 → [1, 14, 22]          length: 3
#   review 2 → [1, 14, 22, 16, 43]  length: 5
# pad_sequences ke baad:
#   review 1 → [0, 0, 0, 0, 1, 14, 22]          length: 500
#   review 2 → [0, 0, 0, 1, 14, 22, 16, 43]     length: 500
# zeros pehle aate hain (pre-padding — keras default)
#
# FLOW:
# DataIngestionArtifact → X_train.npy, X_test.npy load
#       ↓
# pad_sequences(maxlen=500) apply
#       ↓
# X_train_padded.npy, X_test_padded.npy save
#       ↓
# DataTransformationArtifact return
# ═══════════════════════════════════════════════════════════════════

import os
import sys
import logging
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from movie_rnn.entity.config_entity import DataTransformationConfig
from movie_rnn.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)
from movie_rnn.utils.main_utils.utils import load_numpy_array, save_numpy_array
from movie_rnn.exception.exception import MovieSentimentException


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ):
        """
        Parameters:
            data_transformation_config  : paths + max_len
            data_ingestion_artifact     : raw .npy paths
            data_validation_artifact    : validation status
                                          IMP: status=False → transform mat karo
        """
        try:
            # ── VALIDATION GATE ───────────────────────────────────
            if not data_validation_artifact.validation_status:
                raise Exception(
                    f"Data validation failed — transformation rok di | "
                    f"reason: {data_validation_artifact.message}"
                )
            # IMP: agar validation fail tha → yahan pipeline rukti hai
            #      Car Price + Network Security mein bhi same pattern tha

            self.config              = data_transformation_config
            self.ingestion_artifact  = data_ingestion_artifact
            logging.info("DataTransformation initialized")

        except Exception as e:
            raise MovieSentimentException(e, sys)


    def apply_padding(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray
    ) -> tuple:
        """
        pad_sequences apply karta hai dono arrays pe

        Parameters:
            X_train : shape (25000,) — variable length sequences
            X_test  : shape (25000,) — variable length sequences

        Returns:
            (X_train_padded, X_test_padded)
            both shape → (25000, 500)

        HOW PAD_SEQUENCES WORKS:
            maxlen=500
            chhota review (len < 500) → front mein zeros add
                [1, 14, 22] → [0, 0, ..., 0, 1, 14, 22]  (500 total)
            lamba review  (len > 500) → front se truncate
                [1,2,3,...600 items] → last 500 items rakhta hai

        IMP: fit_transform sirf train pe — test pe sirf transform
             lekin pad_sequences stateless hai — koi fitting nahi hoti
             maxlen ek fixed constant hai → dono pe same apply hota hai
             Car Price mein StandardScaler fit_transform(train) transform(test)
             Yahan woh distinction nahi hai — padding deterministic hai
        """
        try:
            logging.info(f"Applying pad_sequences | maxlen={self.config.max_len}")

            X_train_padded = pad_sequences(
                X_train,
                maxlen=self.config.max_len
            )
            # DRY RUN:
            # X_train.shape  = (25000,)   ← array of variable lists
            # maxlen         = 500
            # X_train_padded = (25000, 500) ← fixed 2D array
            # X_train_padded[0] = [0, 0, ..., 0, 1, 14, 22, 16, 43]
            #                      ^^^^^^^^^^^^ zeros = padding

            X_test_padded = pad_sequences(
                X_test,
                maxlen=self.config.max_len
            )
            # DRY RUN:
            # X_test_padded = (25000, 500)

            logging.info(
                f"Padding done | "
                f"X_train: {X_train_padded.shape} | "
                f"X_test: {X_test_padded.shape}"
            )

            return X_train_padded, X_test_padded

        except Exception as e:
            raise MovieSentimentException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Main method — training pipeline yahi call karega

        FLOW:
        X_train.npy, X_test.npy load
              ↓
        pad_sequences(maxlen=500)
              ↓
        X_train_padded.npy, X_test_padded.npy save
              ↓
        DataTransformationArtifact return
        """
        try:
            logging.info(">>> DataTransformation started <<<")

            # ── STEP 1: raw arrays load karo ─────────────────────
            X_train = load_numpy_array(self.ingestion_artifact.x_train_path)
            X_test  = load_numpy_array(self.ingestion_artifact.x_test_path)
            # X_train.shape → (25000,)  variable length sequences
            # X_test.shape  → (25000,)

            # ── STEP 2: padding apply karo ────────────────────────
            X_train_padded, X_test_padded = self.apply_padding(X_train, X_test)
            # X_train_padded.shape → (25000, 500)
            # X_test_padded.shape  → (25000, 500)

            # ── STEP 3: padded arrays save karo ──────────────────
            save_numpy_array(
                self.config.x_train_padded_path,
                X_train_padded
            )
            # → "Artifacts/.../transformed/X_train_padded.npy"

            save_numpy_array(
                self.config.x_test_padded_path,
                X_test_padded
            )
            # → "Artifacts/.../transformed/X_test_padded.npy"

            # ── STEP 4: artifact return karo ─────────────────────
            artifact = DataTransformationArtifact(
                x_train_padded_path = self.config.x_train_padded_path,
                x_test_padded_path  = self.config.x_test_padded_path,
            )

            logging.info(">>> DataTransformation completed <<<")
            return artifact

        except Exception as e:
            raise MovieSentimentException(e, sys)


# ─────────────────────────────────────────────────────────────────
# DRY RUN — full flow
#
# transformation = DataTransformation(
#     transformation_config,
#     ingestion_artifact,
#     validation_artifact   ← status=True → gate pass
# )
# artifact = transformation.initiate_data_transformation()
#
# artifact.x_train_padded_path → "Artifacts/.../transformed/X_train_padded.npy"
# artifact.x_test_padded_path  → "Artifacts/.../transformed/X_test_padded.npy"
#
# shapes:
# X_train_padded → (25000, 500)
# X_test_padded  → (25000, 500)
# ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────
# COMPARISON — Car Price vs Movie Sentiment
#
# ┌──────────────────┬──────────────────────┬─────────────────────┐
# │ Feature          │ Car Price            │ Movie Sentiment      │
# ├──────────────────┼──────────────────────┼─────────────────────┤
# │ Transformer      │ ColumnTransformer    │ pad_sequences       │
# │                  │ StandardScaler+OHE   │ maxlen=500          │
# │ Fit on train?    │ Yes — fit_transform  │ No — stateless      │
# │                  │ on train only        │ same maxlen both    │
# │ Output           │ .npy arrays          │ .npy arrays (same)  │
# │ X shape before   │ (12283, 10)          │ (25000,) var len    │
# │ X shape after    │ (12283, 44)          │ (25000, 500) fixed  │
# │ y transform      │ log1p(selling_price) │ nothing — already   │
# │                  │                      │ 0/1 integers        │
# │ Saved artifact   │ preprocessor.pkl     │ — (no fitted obj)   │
# └──────────────────┴──────────────────────┴─────────────────────┘
# ─────────────────────────────────────────────────────────────────