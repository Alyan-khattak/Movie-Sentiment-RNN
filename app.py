# ═══════════════════════════════════════════════════════════════════
# app.py
# ═══════════════════════════════════════════════════════════════════
# FastAPI app — Movie Sentiment RNN ka serving layer
#
# CAR PRICE + NETWORK SECURITY SE SAME PATTERN:
#   lifespan event → startup pe HF se model pull
#   /train route   → TrainingPipeline().run_pipeline()
#   /predict route → IMDBSentimentModel.predict()
#
# FLOW:
# startup
#   ↓
# pull_model_from_huggingface()
#   → final_model/model.keras + final_model/word_index.pkl
#   ↓
# IMDBSentimentModel(word_index, model) → app.state.model
#   ↓
# FastAPI ready
#
# ROUTES:
#   GET  /          → index.html (landing page)
#   GET  /predict   → predict.html (text input form)
#   POST /predict   → inference → result.html
#   GET  /train     → TrainingPipeline().run_pipeline()
# ═══════════════════════════════════════════════════════════════════

import os
import sys
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

import dill
from tensorflow import keras
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from movie_rnn.cloud.hf_syncer import pull_model_from_huggingface
from movie_rnn.utils.dl_utils.model.estimator import IMDBSentimentModel
from movie_rnn.pipeline.training_pipeline import TrainingPipeline
from movie_rnn.exception.exception import MovieSentimentException
from movie_rnn.constants.training_pipeline import HF_MODEL_DIR

load_dotenv()


# ══════════════════════════════════════════════════════════════════
# LIFESPAN — startup pe model load karta hai
# ══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event — startup + shutdown handle karta hai

    STARTUP FLOW:
    pull_model_from_huggingface()
          ↓
    final_model/model.keras     → keras.models.load_model()
    final_model/word_index.pkl  → dill.load()
          ↓
    IMDBSentimentModel(word_index, model)
          ↓
    app.state.model → har request pe available

    IMP: app.state.model → thread-safe way to share model
         Car Price + Network Security mein bhi same tha
    """
    try:
        logging.info(">>> App startup — pulling model from HuggingFace <<<")

        # ── STEP 1: HF se pull karo ───────────────────────────────
        pull_model_from_huggingface()
        # → final_model/ mein save hoga
        # → final_model/model.keras
        # → final_model/word_index.pkl

        # ── STEP 2: model load karo ───────────────────────────────
        model_path = os.path.join(HF_MODEL_DIR, "model.keras")
        model      = keras.models.load_model(model_path)
        logging.info("model.keras loaded")

        # ── STEP 3: word_index load karo ─────────────────────────
        wi_path = os.path.join(HF_MODEL_DIR, "word_index.pkl")
        with open(wi_path, "rb") as f:
            word_index = dill.load(f)
        logging.info(f"word_index loaded | vocab: {len(word_index)}")

        # ── STEP 4: wrapper banao → app.state mein store ─────────
        app.state.model = IMDBSentimentModel(
            word_index   = word_index,
            model        = model,
            max_features = 10000,
            max_len      = 500,
        )
        # IMP: app.state.model → global state
        #      har request handler isko access kar sakta hai
        #      request.app.state.model se

        logging.info(">>> Model ready — app startup complete <<<")

    except Exception as e:
        logging.error(f"Startup failed: {e}")
        raise MovieSentimentException(e, sys)

    yield
    # ── SHUTDOWN ──────────────────────────────────────────────────
    logging.info(">>> App shutdown <<<")


# ══════════════════════════════════════════════════════════════════
# APP INIT
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title      = "Movie Sentiment RNN",
    description= "IMDB Movie Review Sentiment Analysis — SimpleRNN",
    version    = "1.0.0",
    lifespan   = lifespan,
)

templates = Jinja2Templates(directory="templates")


# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════

# ── GET / — landing page ──────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Landing page"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# ── GET /predict — text input form ───────────────────────────────
@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    """Prediction form — user review input karta hai"""
    return templates.TemplateResponse(
        "predict.html",
        {"request": request}
    )


# ── POST /predict — inference ────────────────────────────────────
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    review: str = Form(...)
    # Form(...) → HTML form se text aata hai
    # name="review" → predict.html ke input field se match karna chahiye
):
    """
    User ka review leta hai → sentiment predict karta hai

    FLOW:
    review (str) → request.app.state.model.predict()
          ↓
    (label, score) → result.html render
    """
    try:
        model_wrapper = request.app.state.model
        # app.state.model → lifespan mein load hua tha

        label, score = model_wrapper.predict(review)
        # label → "Positive" or "Negative"
        # score → 0.0 to 1.0

        # score ko percentage mein convert karo display ke liye
        confidence = round(score * 100, 2) if label == "Positive" \
                     else round((1 - score) * 100, 2)
        # DRY RUN:
        # label = "Positive", score = 0.87
        # confidence = 87.0%
        # label = "Negative", score = 0.23
        # confidence = (1-0.23)*100 = 77.0%

        return templates.TemplateResponse(
            "result.html",
            {
                "request"   : request,
                "review"    : review,
                "label"     : label,
                "score"     : score,
                "confidence": confidence,
            }
        )

    except Exception as e:
        raise MovieSentimentException(e, sys)


# ── GET /train — trigger training pipeline ───────────────────────
@app.get("/train", response_class=HTMLResponse)
async def train(request: Request):
    """
    Training pipeline trigger karta hai

    IMP: production mein yeh route auth se protect karna chahiye
         Car Price + Network Security mein bhi same tha — open rakha
         demo ke liye
    """
    try:
        logging.info(">>> /train route hit — starting pipeline <<<")

        pipeline = TrainingPipeline()
        artifact = pipeline.run_pipeline()

        return HTMLResponse(
            content=(
                f"<h2>Training Complete</h2>"
                f"<p>Train Accuracy : {artifact.train_accuracy:.4f}</p>"
                f"<p>Test  Accuracy : {artifact.test_accuracy:.4f}</p>"
                f"<p>Model Accepted : {artifact.is_model_accepted}</p>"
            )
        )

    except Exception as e:
        raise MovieSentimentException(e, sys)


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host = "0.0.0.0",
        port = 8080,
    )
    # run karo:
    # python app.py
    # ya
    # uvicorn app:app --host 0.0.0.0 --port 8080 --reload


# ─────────────────────────────────────────────────────────────────
# COMPARISON — Car Price + Network Security vs Movie Sentiment
#
# ┌──────────────────┬──────────────────────┬─────────────────────┐
# │ Feature          │ Car Price/NetSec     │ Movie Sentiment      │
# ├──────────────────┼──────────────────────┼─────────────────────┤
# │ Lifespan         │ pull model → load    │ pull model → load   │
# │ Startup files    │ model.keras +        │ model.keras +       │
# │                  │ preprocessor.pkl     │ word_index.pkl      │
# │ app.state.model  │ CarPriceModel/       │ IMDBSentimentModel  │
# │                  │ NetworkModel         │                     │
# │ /predict input   │ form fields (nums)   │ textarea (text)     │
# │ /predict output  │ price (float)        │ label + confidence  │
# │ /train route     │ yes                  │ yes (same)          │
# └──────────────────┴──────────────────────┴─────────────────────┘
# ─────────────────────────────────────────────────────────────────