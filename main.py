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
    DataValidationConfig,
    DataTransformationConfig
)
from movie_rnn.components.data_ingestion import DataIngestion
from movie_rnn.components.data_validation import DataValidation
from movie_rnn.components.data_transformation import DataTransformation

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
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    # ── STEP 4: print artifact paths ─────────────────────────────
    logging.info("DataIngestionArtifact:")
    logging.info(f"  x_train_path    → {data_ingestion_artifact.x_train_path}")
    logging.info(f"  x_test_path     → {data_ingestion_artifact.x_test_path}")
    logging.info(f"  y_train_path    → {data_ingestion_artifact.y_train_path}")
    logging.info(f"  y_test_path     → {data_ingestion_artifact.y_test_path}")
    logging.info(f"  word_index_path → {data_ingestion_artifact.word_index_path}")

    logging.info(">>> DataIngestion: PASSED <<<")



    validation_config  = DataValidationConfig(pipeline_config)  # add

    # validation
    data_validation = DataValidation(validation_config, data_ingestion_artifact)
    validation_artifact = data_validation.initiate_data_validation()
    logging.info(f"Validation status: {validation_artifact.validation_status}")
    logging.info(f"Validation message: {validation_artifact.message}")


    transformation_config = DataTransformationConfig(pipeline_config)
    data_transformation = DataTransformation(
        transformation_config,
        data_ingestion_artifact,
        validation_artifact
    )
    transformation_artifact = data_transformation.initiate_data_transformation()
    logging.info(f"X_train_padded: {transformation_artifact.x_train_padded_path}")
    logging.info(f"X_test_padded:  {transformation_artifact.x_test_padded_path}")
except Exception as e:
    raise MovieSentimentException(e, sys)