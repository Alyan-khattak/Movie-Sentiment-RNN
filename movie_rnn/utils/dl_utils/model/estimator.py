# ═══════════════════════════════════════════════════════════════════
# movie_rnn/utils/dl_utils/model/estimator.py
# ═══════════════════════════════════════════════════════════════════
# IMDBSentimentModel — model + word_index ek saath wrap karta hai
#
# WHY ESTIMATOR?
# Car Price mein CarPriceModel tha:
#   preprocessor.pkl + model.keras → ek object
#   predict() → raw input → preprocessor → model → output
#
# Yahan bhi same:
#   word_index dict + model.keras → ek object
#   predict() → raw text → encode → pad → model → label + score
#
# IMP: Yahi object HuggingFace pe jayega
#      FastAPI startup pe yahi load hoga
#      word_index aur model ALAG save honge HF pe
#      load hone ke baad yahan wrap honge
# ═══════════════════════════════════════════════════════════════════

import sys
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from movie_rnn.exception.exception import MovieSentimentException


class IMDBSentimentModel:
    def __init__(
        self,
        word_index: dict,
        model,
        max_features: int = 10000,
        max_len: int = 500
    ):
        """
        Parameters:
            word_index   : {"the": 1, "movie": 2, ...} — 88k words
            model        : trained keras Sequential model
            max_features : vocab size = 10000
            max_len      : pad length = 500
        """
        try:
            self.word_index   = word_index
            self.model        = model
            self.max_features = max_features
            self.max_len      = max_len

        except Exception as e:
            raise MovieSentimentException(e, sys)


    def preprocess(self, text: str) -> np.ndarray:
        """
        Raw text → padded numpy array

        FLOW:
        "This movie was amazing"
              ↓
        lowercase + split
        ["this", "movie", "was", "amazing"]
              ↓
        word_index lookup + offset +3
        [15, 48, 7, 94]
              ↓
        pad_sequences(maxlen=500)
        [[0, 0, ..., 0, 15, 48, 7, 94]]  shape: (1, 500)

        WHY +3 OFFSET?
        imdb dataset reserves first 3 indices:
            0 → padding token
            1 → start of sequence token
            2 → unknown word token
        isliye actual word indices 3 se start hote hain
        word_index.get(word, 2) + 3
            agar word exist kare  → actual index + 3
            agar word nahi mile   → 2 + 3 = 5 (unknown)
        """
        try:
            # ── STEP 1: lowercase + split ─────────────────────────
            words = text.lower().split()
            # "This Movie was AMAZING" → ["this", "movie", "was", "amazing"]

            # ── STEP 2: word → integer ────────────────────────────
            encoded = [
                self.word_index.get(word, 2) + 3
                for word in words
            ]
            # DRY RUN:
            # "movie" → word_index.get("movie", 2) = 17
            # 17 + 3 = 20
            # "xyz123" → word_index.get("xyz123", 2) = 2
            # 2 + 3 = 5  ← unknown token

            # ── STEP 3: pad ───────────────────────────────────────
            padded = pad_sequences(
                [encoded],
                maxlen=self.max_len
            )
            # [encoded] → list of one sequence
            # output shape → (1, 500)

            return padded

        except Exception as e:
            raise MovieSentimentException(e, sys)


    def predict(self, text: str) -> tuple:
        """
        Raw text → sentiment label + score

        Parameters:
            text (str) : user ka raw review
                         e.g. "This movie was absolutely fantastic"

        Returns:
            (label, score)
            label : "Positive" or "Negative"
            score : float 0.0 → 1.0  (probability of Positive)

        FLOW:
        text → preprocess() → (1, 500) array
                    ↓
             model.predict()
                    ↓
             [[0.87]]
                    ↓
             score = 0.87
             label = "Positive" if score > 0.5 else "Negative"
        """
        try:
            # ── STEP 1: preprocess ────────────────────────────────
            padded = self.preprocess(text)
            # padded.shape → (1, 500)

            # ── STEP 2: model predict ─────────────────────────────
            prediction = self.model.predict(padded, verbose=0)
            # prediction → [[0.87]]
            # verbose=0  → progress bar mat dikha (FastAPI mein clutters logs)

            # ── STEP 3: score + label ─────────────────────────────
            score = float(prediction[0][0])
            # [[0.87]] → [0.87] → 0.87 → float(0.87)

            label = "Positive" if score > 0.5 else "Negative"
            # 0.87 > 0.5 → "Positive"
            # 0.23 > 0.5 → False → "Negative"

            return label, score

        except Exception as e:
            raise MovieSentimentException(e, sys)


# ─────────────────────────────────────────────────────────────────
# DRY RUN
#
# model_wrapper = IMDBSentimentModel(word_index, keras_model)
#
# label, score = model_wrapper.predict("This movie was fantastic")
# → preprocess("This movie was fantastic")
#   → ["this", "movie", "was", "fantastic"]
#   → [15, 48, 7, 231]  (after +3 offset)
#   → padded shape: (1, 500)
# → model.predict([[0,0,...,15,48,7,231]])
#   → [[0.87]]
# → score = 0.87
# → label = "Positive"
# → return ("Positive", 0.87)
# ─────────────────────────────────────────────────────────────────