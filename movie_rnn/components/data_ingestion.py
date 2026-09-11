# ═══════════════════════════════════════════════════════════════════
# movie_rnn/components/data_ingestion.py
# ═══════════════════════════════════════════════════════════════════
# Pipeline ka PEHLA component — data download + save karta hai
#
# WHY DATA INGESTION?
# Har ML project ka pehla step — raw data lao aur save karo
# Car Price   → CSV file padhta tha → train/test split → .csv save
# Network Sec → MongoDB se pull karta tha → .csv save
# Movie RNN   → keras.datasets.imdb se load → .npy + .pkl save
#
# IMP: Yahan koi preprocessing nahi hoti
#      Sirf raw data lao → disk pe save karo → artifact return karo
#      Padding (pad_sequences) → DataTransformation mein hogi
#
# FLOW:
# imdb.load_data(num_words=10000)
#       ↓
# X_train, X_test, y_train, y_test  (raw integer sequences)
#       ↓
# imdb.get_word_index()  →  word_index dict
#       ↓
# save X_train.npy, X_test.npy, y_train.npy, y_test.npy
# save word_index.pkl
#       ↓
# DataIngestionArtifact return
# ═══════════════════════════════════════════════════════════════════

import os
import sys
import numpy as np
from tensorflow.keras.datasets import imdb

from movie_rnn.entity.config_entity import DataIngestionConfig
from movie_rnn.entity.artifact_entity import DataIngestionArtifact
from movie_rnn.utils.main_utils.utils import save_numpy_array, save_object
from movie_rnn.exception.exception import MovieSentimentException
from movie_rnn.logging.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Parameters:
            data_ingestion_config (DataIngestionConfig):
                config object — saare paths iske andar hain
                component khud koi path nahi banata

        IMP: Car Price + Network Security mein bhi same pattern tha
             Component → config inject → config se paths lo
             Khud kuch hardcode mat karo
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            logging.info("DataIngestion initialized")
        except Exception as e:
            raise MovieSentimentException(e, sys)


    def download_data(self) -> tuple:
        """
        keras.datasets.imdb se data load karta hai

        Returns:
            (X_train, y_train, X_test, y_test, word_index)

        FLOW:
        imdb.load_data(num_words=10000)
              ↓
        X_train → list of lists  (25000 reviews, variable length)
                  e.g. X_train[0] = [1, 14, 22, 16, 43, 530, ...]
        y_train → array of 0/1   (25000 labels)
                  0 = Negative, 1 = Positive
              ↓
        imdb.get_word_index()
              ↓
        word_index → {"the": 1, "and": 2, "movie": 45, ...}
                     10000 words total

        IMP: num_words=10000 → sirf top 10k frequent words load honge
             baaki sab → index 2 (unknown token) se replace honge
        """
        try:
            logging.info("Loading IMDB dataset from keras...")

            # ── LOAD DATASET ──────────────────────────────────────
            (X_train, y_train), (X_test, y_test) = imdb.load_data(
                num_words=self.data_ingestion_config.max_features
            )
            # DRY RUN:
            # max_features = 10000
            # X_train.shape → (25000,)   ← 25000 reviews
            # y_train.shape → (25000,)   ← 25000 labels (0 or 1)
            # X_test.shape  → (25000,)
            # y_test.shape  → (25000,)
            #
            # X_train[0] → [1, 14, 22, 16, 43, 530, 973, ...]
            # y_train[0] → 1  (Positive)
            #
            # IMP: X_train dtype = object (variable length sequences)
            #      y_train dtype = int64

            logging.info(
                f"Dataset loaded | "
                f"X_train: {X_train.shape} | "
                f"X_test: {X_test.shape}"
            )

            # ── LOAD WORD INDEX ───────────────────────────────────
            word_index = imdb.get_word_index()
            # word_index = {"the": 1, "and": 2, "a": 3, ...}
            # IMP: yeh dictionary 88000+ words rakhti hai
            #      lekin humne sirf 10000 load kiye hain
            #      isliye prediction pe bhi sirf top 10k kaam karenge

            logging.info(f"Word index loaded | vocab size: {len(word_index)}")

            return X_train, y_train, X_test, y_test, word_index

        except Exception as e:
            raise MovieSentimentException(e, sys)


    def save_data(
        self,
        X_train, y_train,
        X_test,  y_test,
        word_index
    ) -> DataIngestionArtifact:
        """
        Downloaded data ko disk pe save karta hai

        Returns:
            DataIngestionArtifact — saare saved file paths

        SAVES:
        raw_data/
        ├── X_train.npy       ← 25000 variable-length sequences
        ├── X_test.npy        ← 25000 variable-length sequences
        ├── y_train.npy       ← 25000 labels (0/1)
        ├── y_test.npy        ← 25000 labels (0/1)
        └── word_index.pkl    ← {"the": 1, "movie": 45, ...}
        """
        try:
            logging.info("Saving ingested data to disk...")

            cfg = self.data_ingestion_config

            # ── SAVE ARRAYS ───────────────────────────────────────
            save_numpy_array(cfg.x_train_path, X_train)
            # → "Artifacts/.../raw_data/X_train.npy"

            save_numpy_array(cfg.x_test_path, X_test)
            # → "Artifacts/.../raw_data/X_test.npy"

            save_numpy_array(cfg.y_train_path, y_train)
            # → "Artifacts/.../raw_data/y_train.npy"

            save_numpy_array(cfg.y_test_path, y_test)
            # → "Artifacts/.../raw_data/y_test.npy"

            # ── SAVE WORD INDEX ───────────────────────────────────
            save_object(cfg.word_index_path, word_index)
            # → "Artifacts/.../raw_data/word_index.pkl"
            # IMP: word_index dict → dill se serialize
            #      prediction time pe text → integers ke liye zaroori

            logging.info("All data saved successfully")

            # ── BUILD ARTIFACT ────────────────────────────────────
            artifact = DataIngestionArtifact(
                x_train_path    = cfg.x_train_path,
                x_test_path     = cfg.x_test_path,
                y_train_path    = cfg.y_train_path,
                y_test_path     = cfg.y_test_path,
                word_index_path = cfg.word_index_path,
            )
            # IMP: artifact mein sirf paths hain — data nahi
            #      next component artifact se path lega → khud load karega

            logging.info(f"DataIngestionArtifact created: {artifact}")
            return artifact

        except Exception as e:
            raise MovieSentimentException(e, sys)


    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Main method — training pipeline yahi call karega

        FLOW:
        download_data()
              ↓
        X_train, y_train, X_test, y_test, word_index
              ↓
        save_data()
              ↓
        DataIngestionArtifact
        """
        try:
            logging.info(">>> DataIngestion started <<<")

            # ── STEP 1: download ──────────────────────────────────
            X_train, y_train, X_test, y_test, word_index = self.download_data()

            # ── STEP 2: save ──────────────────────────────────────
            artifact = self.save_data(
                X_train, y_train,
                X_test,  y_test,
                word_index
            )

            logging.info(">>> DataIngestion completed <<<")
            return artifact

        except Exception as e:
            raise MovieSentimentException(e, sys)


# ─────────────────────────────────────────────────────────────────
# DRY RUN — full flow
#
# pipeline_cfg  = TrainingPipelineConfig()
# ingestion_cfg = DataIngestionConfig(pipeline_cfg)
# ingestion     = DataIngestion(ingestion_cfg)
# artifact      = ingestion.initiate_data_ingestion()
#
# artifact.x_train_path    → "Artifacts/.../raw_data/X_train.npy"
# artifact.x_test_path     → "Artifacts/.../raw_data/X_test.npy"
# artifact.y_train_path    → "Artifacts/.../raw_data/y_train.npy"
# artifact.y_test_path     → "Artifacts/.../raw_data/y_test.npy"
# artifact.word_index_path → "Artifacts/.../raw_data/word_index.pkl"
# ─────────────────────────────────────────────────────────────────

