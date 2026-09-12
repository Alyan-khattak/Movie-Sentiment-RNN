# ═══════════════════════════════════════════════════════════════════
# movie_rnn/cloud/hf_syncer.py
# ═══════════════════════════════════════════════════════════════════
# HuggingFace Hub se model push/pull karta hai
# ModelTrainer yahan se import karta hai — cloud logic alag file mein
#
# WHY ALAG FILE?
# model_trainer.py → training logic
# hf_syncer.py     → cloud sync logic
# Separation of concerns — Car Price + Network Security mein bhi same tha
#
# PUSH: final_model/ folder → HF repo
#   final_model/
#   ├── model.keras       ← trained RNN weights
#   └── word_index.pkl    ← text → integer mapping
#
# PULL: HF repo → final_model/ folder (app.py startup pe)
# ═══════════════════════════════════════════════════════════════════

import os
import sys
import logging

from movie_rnn.exception.exception import MovieSentimentException
from movie_rnn.constants.training_pipeline import (
    HF_REPO_ID,
    HF_REPO_TYPE,
    HF_MODEL_DIR,
)


def push_model_to_huggingface(
    folder_path: str = HF_MODEL_DIR,
    repo_id:     str = HF_REPO_ID,
    repo_type:   str = HF_REPO_TYPE,
    private:     bool = False,
) -> str:
    """
    final_model/ folder ko HuggingFace Hub pe push karta hai

    Parameters:
        folder_path (str)  : local folder → default: "final_model/"
                             model.keras + word_index.pkl yahan hone chahiye
        repo_id     (str)  : HF repo → default: constants se
        repo_type   (str)  : "model"
        private     (bool) : False → public repo

    Returns:
        str : HF repo URL

    FLOW:
    HfApi() banao
          ↓
    create_repo() → exist_ok=True → already exist toh error nahi
          ↓
    upload_folder(final_model/) → HF repo
          ↓
    URL return

    IMP: upload_folder → poora folder ek saath push karta hai
         Car Price mein bhi same tha — individual files nahi
    """
    try:
        from huggingface_hub import HfApi

        logging.info(f"Pushing to HuggingFace: {repo_id}")

        api = HfApi()

        # ── STEP 1: repo banao agar exist nahi karta ─────────────
        api.create_repo(
            repo_id  = repo_id,
            repo_type= repo_type,
            private  = private,
            exist_ok = True,
            # exist_ok=True → already exist → error nahi aayega
        )
        logging.info(f"Repo ready: huggingface.co/{repo_id}")

        # ── STEP 2: folder push karo ──────────────────────────────
        api.upload_folder(
            folder_path = folder_path,
            repo_id     = repo_id,
            repo_type   = repo_type,
        )
        # DRY RUN:
        # folder_path = "final_model/"
        # final_model/model.keras      → HF repo/model.keras
        # final_model/word_index.pkl   → HF repo/word_index.pkl

        url = f"https://huggingface.co/{repo_id}"
        logging.info(f">>> Model pushed to HF: {url} <<<")
        return url

    except Exception as e:
        raise MovieSentimentException(e, sys)


def pull_model_from_huggingface(
    repo_id:   str = HF_REPO_ID,
    repo_type: str = HF_REPO_TYPE,
    save_dir:  str = HF_MODEL_DIR,
) -> str:
    """
    HuggingFace se model download karta hai
    app.py lifespan event mein call hoga — startup pe

    Parameters:
        repo_id   (str) : HF repo naam
        repo_type (str) : "model"
        save_dir  (str) : local folder jahan save karna hai
                          default: "final_model/"

    Returns:
        str : local folder path jahan files save hui

    FLOW:
    snapshot_download() → poora repo download
          ↓
    final_model/
    ├── model.keras
    └── word_index.pkl
          ↓
    local_path return → app.py wahan se load karega

    IMP: snapshot_download → poora repo ek saath download karta hai
         Car Price mein bhi same tha
    """
    try:
        from huggingface_hub import snapshot_download

        logging.info(f"Pulling from HuggingFace: {repo_id}")

        os.makedirs(save_dir, exist_ok=True)
        # final_model/ folder banao agar exist nahi karta

        local_path = snapshot_download(
            repo_id   = repo_id,
            repo_type = repo_type,
            local_dir = save_dir,
        )
        # DRY RUN:
        # repo_id    = "alyan-ktk/movie-sentiment-rnn"
        # local_dir  = "final_model/"
        # downloads:
        #   final_model/model.keras
        #   final_model/word_index.pkl
        # local_path = "final_model/"

        logging.info(f">>> Model pulled from HF to: {local_path} <<<")
        return local_path

    except Exception as e:
        raise MovieSentimentException(e, sys)


# ─────────────────────────────────────────────────────────────────
# DRY RUN — push
#
# push_model_to_huggingface()
# → HfApi()
# → create_repo("alyan-ktk/movie-sentiment-rnn", exist_ok=True)
# → upload_folder("final_model/" → "alyan-ktk/movie-sentiment-rnn")
#   → model.keras uploaded
#   → word_index.pkl uploaded
# → "https://huggingface.co/alyan-ktk/movie-sentiment-rnn"
#
# DRY RUN — pull
#
# pull_model_from_huggingface()
# → makedirs("final_model/")
# → snapshot_download("alyan-ktk/movie-sentiment-rnn" → "final_model/")
#   → final_model/model.keras
#   → final_model/word_index.pkl
# → "final_model/"
# ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────
# COMPARISON — Car Price vs Movie Sentiment
#
# ┌──────────────────┬──────────────────────┬─────────────────────┐
# │ Feature          │ Car Price            │ Movie Sentiment      │
# ├──────────────────┼──────────────────────┼─────────────────────┤
# │ Push method      │ upload_folder()      │ upload_folder()     │
# │ Pull method      │ snapshot_download()  │ snapshot_download() │
# │ Files pushed     │ model.keras +        │ model.keras +       │
# │                  │ preprocessor.pkl     │ word_index.pkl      │
# │ Local dir        │ final_model/         │ final_model/        │
# │ Repo type        │ "model"              │ "model"             │
# └──────────────────┴──────────────────────┴─────────────────────┘
# ─────────────────────────────────────────────────────────────────