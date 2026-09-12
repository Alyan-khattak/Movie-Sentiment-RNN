# ═══════════════════════════════════════════════════════════════════
# movie_rnn/pipeline/training_pipeline.py
# ═══════════════════════════════════════════════════════════════════
# Poore pipeline ka ORCHESTRATOR — sab components ko order mein chalata hai
#
# WHY TRAINING PIPELINE?
# Car Price + Network Security mein same pattern:
#   main.py mein sab directly call karna → messy
#   TrainingPipeline class → ek method → sab kuch
#   app.py bhi yahi call karta hai jab /train route hit hota hai
#
# FLOW:
# TrainingPipeline.run_pipeline()
#       ↓
# DataIngestion.initiate_data_ingestion()
#       ↓
# DataValidation.initiate_data_validation()
#       ↓
# DataTransformation.initiate_data_transformation()
#       ↓
# ModelTrainer.initiate_model_trainer()
#       ↓
# ModelTrainerArtifact return
# ═══════════════════════════════════════════════════════════════════

import sys
import logging

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

from movie_rnn.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from movie_rnn.exception.exception import MovieSentimentException


class TrainingPipeline:
    def __init__(self):
        """
        TrainingPipeline initialize karta hai
        TrainingPipelineConfig banata hai — timestamp generate hota hai
        Sab component configs isko inject karte hain
        """
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
            logging.info(
                f"TrainingPipeline initialized | "
                f"artifact_dir: {self.training_pipeline_config.artifact_dir}"
            )
        except Exception as e:
            raise MovieSentimentException(e, sys)


    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        DataIngestion component run karta hai

        Returns:
            DataIngestionArtifact
        """
        try:
            logging.info("─" * 55)
            logging.info(">>> Stage 1: DataIngestion <<<")

            ingestion_config   = DataIngestionConfig(self.training_pipeline_config)
            data_ingestion     = DataIngestion(ingestion_config)
            ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info(f"DataIngestion done: {ingestion_artifact}")
            return ingestion_artifact

        except Exception as e:
            raise MovieSentimentException(e, sys)


    def start_data_validation(
        self,
        ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """
        DataValidation component run karta hai

        Parameters:
            ingestion_artifact : DataIngestion ka output

        Returns:
            DataValidationArtifact
        """
        try:
            logging.info("─" * 55)
            logging.info(">>> Stage 2: DataValidation <<<")

            validation_config   = DataValidationConfig(self.training_pipeline_config)
            data_validation     = DataValidation(
                validation_config,
                ingestion_artifact
            )
            validation_artifact = data_validation.initiate_data_validation()

            logging.info(f"DataValidation done: {validation_artifact}")
            return validation_artifact

        except Exception as e:
            raise MovieSentimentException(e, sys)


    def start_data_transformation(
        self,
        ingestion_artifact:  DataIngestionArtifact,
        validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        """
        DataTransformation component run karta hai

        Parameters:
            ingestion_artifact  : DataIngestion ka output
            validation_artifact : DataValidation ka output
                                  IMP: status=False → transform rok do

        Returns:
            DataTransformationArtifact
        """
        try:
            logging.info("─" * 55)
            logging.info(">>> Stage 3: DataTransformation <<<")

            transformation_config   = DataTransformationConfig(self.training_pipeline_config)
            data_transformation     = DataTransformation(
                transformation_config,
                ingestion_artifact,
                validation_artifact,
            )
            transformation_artifact = data_transformation.initiate_data_transformation()

            logging.info(f"DataTransformation done: {transformation_artifact}")
            return transformation_artifact

        except Exception as e:
            raise MovieSentimentException(e, sys)


    def start_model_trainer(
        self,
        ingestion_artifact:       DataIngestionArtifact,
        transformation_artifact:  DataTransformationArtifact,
    ) -> ModelTrainerArtifact:
        """
        ModelTrainer component run karta hai

        Parameters:
            ingestion_artifact      : y_train, y_test, word_index paths
            transformation_artifact : X_train_padded, X_test_padded paths

        Returns:
            ModelTrainerArtifact
        """
        try:
            logging.info("─" * 55)
            logging.info(">>> Stage 4: ModelTrainer <<<")

            trainer_config   = ModelTrainerConfig(self.training_pipeline_config)
            model_trainer    = ModelTrainer(
                trainer_config,
                ingestion_artifact,
                transformation_artifact,
            )
            trainer_artifact = model_trainer.initiate_model_trainer()

            logging.info(f"ModelTrainer done: {trainer_artifact}")
            return trainer_artifact

        except Exception as e:
            raise MovieSentimentException(e, sys)


    def run_pipeline(self) -> ModelTrainerArtifact:
        """
        Poora pipeline ek saath chalata hai

        FLOW:
        start_data_ingestion()
              ↓
        start_data_validation()
              ↓
        start_data_transformation()
              ↓
        start_model_trainer()
              ↓
        ModelTrainerArtifact return

        IMP: har stage ka artifact next stage ko pass hota hai
             Car Price + Network Security mein bhi same pattern tha
        """
        try:
            logging.info("═" * 55)
            logging.info(">>> TRAINING PIPELINE STARTED <<<")
            logging.info("═" * 55)

            # ── Stage 1 ───────────────────────────────────────────
            ingestion_artifact = self.start_data_ingestion()

            # ── Stage 2 ───────────────────────────────────────────
            validation_artifact = self.start_data_validation(
                ingestion_artifact
            )

            # ── Stage 3 ───────────────────────────────────────────
            transformation_artifact = self.start_data_transformation(
                ingestion_artifact,
                validation_artifact,
            )

            # ── Stage 4 ───────────────────────────────────────────
            trainer_artifact = self.start_model_trainer(
                ingestion_artifact,
                transformation_artifact,
            )

            logging.info("═" * 55)
            logging.info(">>> TRAINING PIPELINE COMPLETED <<<")
            logging.info(
                f"Train accuracy : {trainer_artifact.train_accuracy:.4f} | "
                f"Test accuracy  : {trainer_artifact.test_accuracy:.4f}"
            )
            logging.info("═" * 55)

            return trainer_artifact

        except Exception as e:
            raise MovieSentimentException(e, sys)


# ─────────────────────────────────────────────────────────────────
# DRY RUN
#
# pipeline = TrainingPipeline()
# artifact = pipeline.run_pipeline()
#
# → Stage 1: DataIngestion    → X_train.npy, y_train.npy, word_index.pkl
# → Stage 2: DataValidation   → validation_report.yaml (status: True)
# → Stage 3: DataTransformation → X_train_padded.npy (25000, 500)
# → Stage 4: ModelTrainer     → model.keras → MLflow → HF push
#
# artifact.train_accuracy → 0.9154
# artifact.test_accuracy  → 0.7726
# artifact.is_model_accepted → True
# ─────────────────────────────────────────────────────────────────


