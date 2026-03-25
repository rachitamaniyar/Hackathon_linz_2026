import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from transformers import pipeline

from pose_detection import detect_pose_from_landmarks
from pose_feedback import generate_pose_feedback


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
IMAGES_DIR = BASE_DIR / "images"
MODEL_ID = os.getenv("YOGA_MODEL_ID", "dima806/yoga_pose_image_classification")
CONFIDENCE_THRESHOLD = float(os.getenv("YOGA_CONFIDENCE_THRESHOLD", "0.45"))
LANDMARK_CONFIDENCE_THRESHOLD = float(os.getenv("YOGA_LANDMARK_THRESHOLD", "0.58"))


class PredictRequest(BaseModel):
    image_base64: str = Field(..., description="Data URL or raw base64-encoded image")
    landmarks: list[dict[str, Any]] = Field(default_factory=list)
    source: str = Field(default="webcam")


@lru_cache(maxsize=1)
def get_classifier():
    return pipeline("image-classification", model=MODEL_ID)


def decode_base64_image(image_base64: str) -> Image.Image:
    payload = image_base64
    if "," in image_base64:
        payload = image_base64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload.") from exc

    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Image payload could not be opened.") from exc


app = FastAPI(title="Yoga Pose Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "landmark_confidence_threshold": LANDMARK_CONFIDENCE_THRESHOLD,
    }


@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/analyze")
def analyze_pose(payload: PredictRequest):
    if not payload.landmarks:
        return {
            "label": None,
            "confidence": 0.0,
            "pose_score": 0,
            "status": "no_landmarks",
            "message": "No pose landmarks detected. Move fully into the camera frame.",
            "threshold": CONFIDENCE_THRESHOLD,
            "suggestions": [],
        }

    classifier = get_classifier()
    image = decode_base64_image(payload.image_base64)
    predictions = classifier(image)

    if not predictions:
        raise HTTPException(status_code=500, detail="The classifier returned no predictions.")

    top_prediction = predictions[0]
    landmark_prediction = detect_pose_from_landmarks(payload.landmarks)
    final_label = None
    final_confidence = top_prediction["score"]
    recognition_method = "image"

    if top_prediction["score"] >= CONFIDENCE_THRESHOLD:
        final_label = top_prediction["label"]
    elif landmark_prediction and landmark_prediction["confidence"] >= LANDMARK_CONFIDENCE_THRESHOLD:
        final_label = landmark_prediction["label"]
        final_confidence = landmark_prediction["confidence"]
        recognition_method = "landmarks"
    else:
        recognition_method = "none"

    feedback = generate_pose_feedback(
        final_label,
        payload.landmarks,
    )

    return {
        "label": final_label,
        "confidence": final_confidence,
        "pose_score": feedback["pose_score"],
        "status": "determined" if final_label else "uncertain",
        "message": (
            final_label
            if final_label
            else "Pose cannot be correctly determined."
        ),
        "threshold": CONFIDENCE_THRESHOLD,
        "recognition_method": recognition_method,
        "landmark_prediction": landmark_prediction,
        "suggestions": feedback["suggestions"],
        "summary": feedback["summary"],
        "landmark_count": len(payload.landmarks),
        "source": payload.source,
        "top_predictions": predictions[:3],
    }
