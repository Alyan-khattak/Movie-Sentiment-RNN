# ═══════════════════════════════════════════════════════════════════
# main.py
# ═══════════════════════════════════════════════════════════════════
# Project root mein hai — pipeline ka entry point
#
# ABHI KE LIYE:
#   Sab 4 components test karta hai ek saath
#   DataIngestion → DataValidation → DataTransformation → ModelTrainer
#
# FINAL MEIN HOGA:
#   TrainingPipeline().run_pipeline() → sab components ek saath
#   Abhi directly call kar rahe hain — pipeline file baad mein
# ═══════════════════════════════════════════════════════════════════

import sys
import logging
from dotenv import load_dotenv

load_dotenv()

from movie_rnn.exception.exception import MovieSentimentException

from movie_rnn.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from movie_rnn.components.data_ingestion import DataIngestion
from movie_rnn.components.data_validation import DataValidation
from movie_rnn.components.data_transformation import DataTransformation
from movie_rnn.components.model_trainer import ModelTrainer


try:
    logging.info("═" * 60)
    logging.info(">>> PIPELINE STARTED <<<")
    logging.info("═" * 60)

    # ── STEP 1: pipeline config → timestamp folder ────────────────
    pipeline_config = TrainingPipelineConfig()
    logging.info(f"Artifact dir: {pipeline_config.artifact_dir}")
    # → "Artifacts/09_11_2026_14_32_00"

    # ─────────────────────────────────────────────────────────────
    # DATA INGESTION
    # ─────────────────────────────────────────────────────────────
    ingestion_config  = DataIngestionConfig(pipeline_config)
    data_ingestion    = DataIngestion(ingestion_config)
    ingestion_artifact = data_ingestion.initiate_data_ingestion()

    logging.info(f"Ingestion done | x_train: {ingestion_artifact.x_train_path}")

    # ─────────────────────────────────────────────────────────────
    # DATA VALIDATION
    # ─────────────────────────────────────────────────────────────
    validation_config   = DataValidationConfig(pipeline_config)
    data_validation     = DataValidation(validation_config, ingestion_artifact)
    validation_artifact = data_validation.initiate_data_validation()

    logging.info(f"Validation done | status: {validation_artifact.validation_status}")

    # ─────────────────────────────────────────────────────────────
    # DATA TRANSFORMATION
    # ─────────────────────────────────────────────────────────────
    transformation_config   = DataTransformationConfig(pipeline_config)
    data_transformation     = DataTransformation(
        transformation_config,
        ingestion_artifact,
        validation_artifact,
    )
    transformation_artifact = data_transformation.initiate_data_transformation()

    logging.info(f"Transformation done | x_train_padded: {transformation_artifact.x_train_padded_path}")

    # ─────────────────────────────────────────────────────────────
    # MODEL TRAINER
    # ─────────────────────────────────────────────────────────────
    trainer_config   = ModelTrainerConfig(pipeline_config)
    model_trainer    = ModelTrainer(
        trainer_config,
        ingestion_artifact,
        transformation_artifact,
    )
    trainer_artifact = model_trainer.initiate_model_trainer()

    logging.info("─" * 60)
    logging.info(f"Train accuracy : {trainer_artifact.train_accuracy:.4f}")
    logging.info(f"Test  accuracy : {trainer_artifact.test_accuracy:.4f}")
    logging.info(f"Model accepted : {trainer_artifact.is_model_accepted}")
    logging.info(f"Model path     : {trainer_artifact.trained_model_path}")
    logging.info("─" * 60)
    logging.info(">>> PIPELINE COMPLETED <<<")
    logging.info("═" * 60)

except Exception as e:
    raise MovieSentimentException(e, sys)