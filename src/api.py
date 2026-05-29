from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

try:
    from .inference import predict
except ImportError:
    from inference import predict

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # allows file:// and localhost origins
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

class PredictRequest(BaseModel):
    text: str
    context: List[str] = []      # list of prior sentences, oldest first

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_ui():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.post("/predict")
def run_predict(req: PredictRequest):
    # Join context list into a single string (matches inference.py's expected input)
    # Training used a single shifted sentence, but joining multiple is harmless
    context_str = " ".join(req.context) if req.context else ""

    result = predict(req.text, context=context_str)

    # Map inference.py's emoji labels to the UI's label keys
    raw_label = result["label"]
    if "Sarcastic 😏" in raw_label:
        ui_label = "sarcastic"
    elif "Not Sarcastic" in raw_label:
        ui_label = "not_sarcastic"
    else:
        ui_label = "uncertain"

    confidence_pct = round(result["confidence"] * 100)

    # Build explanation string
    if ui_label == "sarcastic":
        explanation = (
            f"Model is {confidence_pct}% confident this is sarcastic. "
            "Positive framing in a likely negative context was detected."
        )
    elif ui_label == "not_sarcastic":
        explanation = (
            f"Model is {100 - confidence_pct}% confident this is a sincere statement. "
            "No strong sarcasm signals were found."
        )
    else:
        explanation = (
            f"Model is uncertain (sarcasm probability: {confidence_pct}%). "
            "Try adding prior context sentences for a better result."
        )

    # Derive tags
    tags = []
    if confidence_pct > 80:
        tags.append("high-confidence")
    if ui_label == "sarcastic":
        tags.append("irony-detected")
    elif ui_label == "not_sarcastic":
        tags.append("sincere")
    else:
        tags.append("ambiguous")
    if context_str:
        tags.append("context-used")

    return {
        "label": ui_label,
        "confidence": confidence_pct,
        "explanation": explanation,
        "tags": tags
    }

