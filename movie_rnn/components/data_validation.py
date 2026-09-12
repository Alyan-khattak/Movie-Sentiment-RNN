# ═══════════════════════════════════════════════════════════════════
# movie_rnn/components/data_validation.py
# ═══════════════════════════════════════════════════════════════════
# Pipeline ka DOOSRA component — data quality check karta hai
#
# WHY DATA VALIDATION?
# Car Price + Network Security mein same pattern:
#   DataIngestion ke baad blindly transform mat karo
#   Pehle check karo data expected shape mein hai ya nahi
#
# CHECKS JO HOTE HAIN:
#   1. Train samples = 25000?
#   2. Test samples  = 25000?
#   3. word_index loaded? keys hain?
#   4. Koi bhi sequence index 10000 se zyada nahi?
#   5. Labels sirf 0 ya 1 hain?
#
# FLOW:
# DataIngestionArtifact (paths)
#       ↓
# arrays load karo (.npy)
#       ↓
# har check run karo → pass/fail
#       ↓
# validation_report.yaml save karo
#       ↓
# DataValidationArtifact return karo
#
# IMP: agar koi bhi check fail → status=False → pipeline rukti hai
#      training_pipeline.py mein check hoga
# ═══════════════════════════════════════════════════════════════════

import os
import sys
import logging
import numpy as np

from movie_rnn.entity.config_entity import DataValidationConfig
from movie_rnn.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from movie_rnn.utils.main_utils.utils import (
    load_numpy_array,
    save_yaml,
    load_object,
)
from movie_rnn.exception.exception import MovieSentimentException


class DataValidation:
    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        """
        Parameters:
            data_validation_config   : paths + expected values
            data_ingestion_artifact  : ingestion ke output paths
                                       yahan se arrays load karenge
        """
        try:
            self.config   = data_validation_config
            self.artifact = data_ingestion_artifact
            logging.info("DataValidation initialized")
        except Exception as e:
            raise MovieSentimentException(e, sys)


    def validate_shape(
        self,
        X_train, X_test,
        y_train, y_test
    ) -> dict:
        """
        Sample counts check karta hai

        CHECKS:
            X_train.shape[0] == 25000 ?
            X_test.shape[0]  == 25000 ?
            y_train.shape[0] == 25000 ?
            y_test.shape[0]  == 25000 ?

        Returns:
            dict → {"shape_check": True/False, "message": "..."}
        """
        try:
            train_ok = (
                X_train.shape[0] == self.config.expected_train_samples and
                y_train.shape[0] == self.config.expected_train_samples
            )
            test_ok = (
                X_test.shape[0] == self.config.expected_test_samples and
                y_test.shape[0] == self.config.expected_test_samples
            )
            # DRY RUN:
            # X_train.shape[0] = 25000 == 25000 → True
            # X_test.shape[0]  = 25000 == 25000 → True
            # train_ok = True, test_ok = True

            status = train_ok and test_ok
            message = (
                "Shape check passed"
                if status
                else (
                    f"Shape mismatch — "
                    f"train: {X_train.shape[0]} "
                    f"test: {X_test.shape[0]}"
                )
            )
            logging.info(f"Shape check: {status} | {message}")
            return {"shape_check": status, "message": message}

        except Exception as e:
            raise MovieSentimentException(e, sys)


    def validate_labels(self, y_train, y_test) -> dict:
        """
        Labels sirf 0 ya 1 hain check karta hai

        CHECKS:
            unique values in y_train ⊆ {0, 1}
            unique values in y_test  ⊆ {0, 1}

        Returns:
            dict → {"label_check": True/False, "message": "..."}
        """
        try:
            valid_labels = {0, 1}

            train_labels = set(np.unique(y_train))
            test_labels  = set(np.unique(y_test))
            # DRY RUN:
            # np.unique(y_train) → [0, 1]
            # train_labels = {0, 1}
            # {0,1} ⊆ {0,1} → True

            status = (
                train_labels.issubset(valid_labels) and
                test_labels.issubset(valid_labels)
            )
            message = (
                "Label check passed"
                if status
                else (
                    f"Invalid labels found — "
                    f"train: {train_labels} "
                    f"test: {test_labels}"
                )
            )
            logging.info(f"Label check: {status} | {message}")
            return {"label_check": status, "message": message}

        except Exception as e:
            raise MovieSentimentException(e, sys)


    def validate_vocab(self, X_train, X_test) -> dict:
        """
        Sequence indices 10000 se zyada nahi hone chahiye

        CHECKS:
            max index in X_train <= max_features (10000)
            max index in X_test  <= max_features (10000)

        IMP: imdb.load_data(num_words=10000) ne already
             10000 se upar ke words ko index 2 se replace kiya
             yeh check verify karta hai woh sahi hua

        Returns:
            dict → {"vocab_check": True/False, "message": "..."}
        """
        try:
            # X_train = array of variable-length lists
            # max index dhundhne ke liye sab lists flatten karo
            train_max = max(max(seq) for seq in X_train)
            test_max  = max(max(seq) for seq in X_test)
            # DRY RUN:
            # X_train[0] = [1, 14, 22, 530, 973]
            # max(X_train[0]) = 973
            # train_max = max across all 25000 reviews
            # → should be <= 10000

            status = (
                train_max <= self.config.expected_max_index and
                test_max  <= self.config.expected_max_index
            )
            message = (
                "Vocab check passed"
                if status
                else (
                    f"Index out of range — "
                    f"train max: {train_max} "
                    f"test max: {test_max}"
                )
            )
            logging.info(f"Vocab check: {status} | {message}")
            return {"vocab_check": status, "message": message}

        except Exception as e:
            raise MovieSentimentException(e, sys)


    def validate_word_index(self, word_index: dict) -> dict:
        """
        word_index dict properly loaded hua check karta hai

        CHECKS:
            word_index is not None
            len(word_index) > 0
            "the" in word_index  ← most common English word hona chahiye

        Returns:
            dict → {"word_index_check": True/False, "message": "..."}
        """
        try:
            status = (
                word_index is not None and
                len(word_index) > 0 and
                "the" in word_index
            )
            # DRY RUN:
            # word_index = {"the": 1, "and": 2, ...}
            # len = 88584
            # "the" in word_index → True
            # status = True

            message = (
                f"Word index check passed | vocab: {len(word_index)}"
                if status
                else "Word index empty or corrupted"
            )
            logging.info(f"Word index check: {status} | {message}")
            return {"word_index_check": status, "message": message}

        except Exception as e:
            raise MovieSentimentException(e, sys)


    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Main method — sab checks run karta hai

        FLOW:
        arrays load
              ↓
        shape check
              ↓
        label check
              ↓
        vocab check
              ↓
        word_index check
              ↓
        report.yaml save
              ↓
        DataValidationArtifact return
        """
        try:
            logging.info(">>> DataValidation started <<<")

            # ── STEP 1: arrays load karo ──────────────────────────
            X_train = load_numpy_array(self.artifact.x_train_path)
            X_test  = load_numpy_array(self.artifact.x_test_path)
            y_train = load_numpy_array(self.artifact.y_train_path)
            y_test  = load_numpy_array(self.artifact.y_test_path)
            word_index = load_object(self.artifact.word_index_path)

            # ── STEP 2: sab checks run karo ───────────────────────
            shape_result      = self.validate_shape(X_train, X_test, y_train, y_test)
            label_result      = self.validate_labels(y_train, y_test)
            vocab_result      = self.validate_vocab(X_train, X_test)
            word_index_result = self.validate_word_index(word_index)

            # ── STEP 3: overall status ────────────────────────────
            all_passed = (
                shape_result["shape_check"]           and
                label_result["label_check"]           and
                vocab_result["vocab_check"]           and
                word_index_result["word_index_check"]
            )
            # True sirf tab jab SARE checks pass hon

            # ── STEP 4: report banao ──────────────────────────────
            report = {
                "validation_status": all_passed,
                "checks": {
                    "shape_check":      shape_result,
                    "label_check":      label_result,
                    "vocab_check":      vocab_result,
                    "word_index_check": word_index_result,
                }
            }

            # ── STEP 5: report save karo ──────────────────────────
            save_yaml(self.config.validation_report_path, report)
            # → "Artifacts/timestamp/data_validation/validation_report.yaml"

            # ── STEP 6: artifact return karo ─────────────────────
            artifact = DataValidationArtifact(
                validation_status       = all_passed,
                validation_report_path  = self.config.validation_report_path,
                message = "All validation checks passed" if all_passed
                          else "One or more validation checks failed"
            )

            logging.info(f">>> DataValidation completed | status: {all_passed} <<<")
            return artifact

        except Exception as e:
            raise MovieSentimentException(e, sys)


# ─────────────────────────────────────────────────────────────────
# DRY RUN — full flow
#
# validation = DataValidation(validation_config, ingestion_artifact)
# artifact   = validation.initiate_data_validation()
#
# artifact.validation_status      → True
# artifact.validation_report_path → "Artifacts/.../validation_report.yaml"
# artifact.message                → "All validation checks passed"
#
# validation_report.yaml contents:
# validation_status: true
# checks:
#   shape_check:
#     shape_check: true
#     message: Shape check passed
#   label_check:
#     label_check: true
#     message: Label check passed
#   vocab_check:
#     vocab_check: true
#     message: Vocab check passed
#   word_index_check:
#     word_index_check: true
#     message: Word index check passed | vocab: 88584
# ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────
# COMPARISON — Car Price + Network Security vs Movie Sentiment
#
# ┌──────────────────┬──────────────────────┬─────────────────────┐
# │ Check            │ Car Price/NetSec     │ Movie Sentiment      │
# ├──────────────────┼──────────────────────┼─────────────────────┤
# │ Schema check     │ column names/types   │ sample counts       │
# │ Drift check      │ KS test per column   │ vocab index range   │
# │ Extra check      │ missing values       │ label values {0,1}  │
# │ Report format    │ validation_report    │ validation_report   │
# │                  │ .yaml                │ .yaml (same)        │
# │ On failure       │ status=False →       │ status=False →      │
# │                  │ pipeline stops       │ pipeline stops      │
# └──────────────────┴──────────────────────┴─────────────────────┘
# ─────────────────────────────────────────────────────────────────