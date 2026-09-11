# ═══════════════════════════════════════════════════════════════════
# movie_rnn/utils/main_utils/utils.py
# ═══════════════════════════════════════════════════════════════════
# Common helper functions — poore project mein use hongi
#
# WHY UTILS?
# Car Price + Network Security mein same pattern:
#   Har component mein save/load repeat nahi karte
#   Ek jagah likhte hain → sab import karte hain
#
# ABHI SIRF jo DataIngestion ko chahiye:
#   - save_numpy_array  → .npy files save karna
#   - save_object       → word_index.pkl save karna
# Baaki functions baad mein add honge jab zaroorat hogi
# ═══════════════════════════════════════════════════════════════════

import os
import sys
import dill
import numpy as np

from movie_rnn.exception.exception import MovieSentimentException
from movie_rnn.logging.logger import logging


# ══════════════════════════════════════════════════════════════════
# FUNCTION 1: save_numpy_array
# ══════════════════════════════════════════════════════════════════
# X_train, X_test, y_train, y_test → .npy files save karta hai

def save_numpy_array(file_path: str, array: np.ndarray) -> None:
    """
    Numpy array ko disk pe save karta hai (.npy format)

    Parameters:
        file_path (str) : jahan save karna hai
                          e.g. "Artifacts/.../raw_data/X_train.npy"
        array (np.ndarray) : save karna wala array

    Flow:
        file_path → folder exist? → nahi → banao
                  → np.save(file_path, array)
                  → "X_train.npy saved" log
    """
    try:
        # ── STEP 1: folder banao agar exist nahi karta ───────────
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        # os.makedirs("Artifacts/.../raw_data", exist_ok=True)
        # exist_ok=True → folder already exist kare toh error nahi

        # ── STEP 2: array save karo ──────────────────────────────
        np.save(file_path, array)
        # IMP: np.save automatically ".npy" append karta hai
        #      agar file_path mein ".npy" nahi hai toh
        #      "X_train" → "X_train.npy" ban jaata hai
        #      isliye constants mein "X_train.npy" rakha hai
        #      taaki confusion na ho

        logging.info(f"Numpy array saved at: {file_path} | shape: {array.shape}")

    except Exception as e:
        raise MovieSentimentException(e, sys)


# ══════════════════════════════════════════════════════════════════
# FUNCTION 2: save_object
# ══════════════════════════════════════════════════════════════════
# word_index.pkl save karta hai — dill use karta hai pickle ki jagah

def save_object(file_path: str, obj: object) -> None:
    """
    Python object ko disk pe save karta hai (dill/pickle format)

    Parameters:
        file_path (str) : jahan save karna hai
                          e.g. "Artifacts/.../raw_data/word_index.pkl"
        obj (object)    : save karna wala object
                          e.g. word_index dict

    WHY DILL over pickle?
        dill → lambda functions, closures bhi serialize kar sakta hai
        pickle → sirf basic objects
        Car Price + Network Security mein bhi dill use kiya tha

    Flow:
        file_path → folder exist? → nahi → banao
                  → dill.dump(obj, file)
                  → "object saved" log
    """
    try:
        # ── STEP 1: folder banao ─────────────────────────────────
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # ── STEP 2: object save karo ─────────────────────────────
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            # IMP: dill.dump(obj, file_obj) — argument order


        logging.info(f"Object saved at: {file_path}")

    except Exception as e:
        raise MovieSentimentException(e, sys)


# ─────────────────────────────────────────────────────────────────
# DRY RUN — save_numpy_array
#
# save_numpy_array(
#     file_path = "Artifacts/09_11_2026/data_ingestion/raw_data/X_train.npy",
#     array     = X_train   # shape: (25000,) — variable length sequences
# )
# → makedirs("Artifacts/.../raw_data")
# → np.save("...X_train.npy", X_train)
# → log: "Numpy array saved | shape: (25000,)"
#
# DRY RUN — save_object
#
# save_object(
#     file_path = "Artifacts/.../raw_data/word_index.pkl",
#     obj       = word_index   # {"the": 1, "movie": 2, ...} 10000 keys
# )
# → makedirs("Artifacts/.../raw_data")
# → dill.dump(word_index, file_obj)
# → log: "Object saved at: .../word_index.pkl"
# ─────────────────────────────────────────────────────────────────