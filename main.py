# ═══════════════════════════════════════════════════════════════════
# main.py
# ═══════════════════════════════════════════════════════════════════
# Project root mein hai — pipeline ka entry point
#
# ABHI KE LIYE:
#   Sirf DataIngestion test karta hai
#   Jaise jaise components bante jayenge → yahan add hote jayenge
#
# FINAL MEIN HOGA:
#   TrainingPipeline().run_pipeline() → sab components ek saath
#
# IMP: Isko run karo → Artifacts/ folder check karo
#      5 files dikh jayengi → ingestion kaam kar rahi hai
# ═══════════════════════════════════════════════════════════════════

import sys
from movie_rnn.exception.exception import MovieSentimentException
from movie_rnn.logging.logger import logging

from movie_rnn.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
)
from movie_rnn.components.data_ingestion import DataIngestion


try:
    logging.info("═" * 60)
    logging.info(">>> PIPELINE STARTED <<<")
    logging.info("═" * 60)

    # ── STEP 1: pipeline config → timestamp folder banao ─────────
    pipeline_config = TrainingPipelineConfig()
    logging.info(f"Artifact dir: {pipeline_config.artifact_dir}")
    # → "Artifacts/09_11_2026_14_32_00"

    # ── STEP 2: ingestion config ──────────────────────────────────
    ingestion_config = DataIngestionConfig(pipeline_config)

    # ── STEP 3: run ingestion ─────────────────────────────────────
    data_ingestion = DataIngestion(ingestion_config)
    artifact = data_ingestion.initiate_data_ingestion()

    # ── STEP 4: print artifact paths ─────────────────────────────
    logging.info("DataIngestionArtifact:")
    logging.info(f"  x_train_path    → {artifact.x_train_path}")
    logging.info(f"  x_test_path     → {artifact.x_test_path}")
    logging.info(f"  y_train_path    → {artifact.y_train_path}")
    logging.info(f"  y_test_path     → {artifact.y_test_path}")
    logging.info(f"  word_index_path → {artifact.word_index_path}")

    logging.info(">>> DataIngestion: PASSED <<<")

except Exception as e:
    raise MovieSentimentException(e, sys)