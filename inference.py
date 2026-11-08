"""Shared VisionGuard image preprocessing and inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = (160, 160)


class InvalidImageError(ValueError):
    """Raised when image bytes cannot be decoded into an RGB image."""


def load_model(model_path: str | Path) -> tf.keras.Model:
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"VisionGuard model file not found: {path}")
    return tf.keras.models.load_model(path, compile=False)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )
        image.set_shape((None, None, 3))
        image = tf.image.resize(
            image,
            IMAGE_SIZE,
            method="bilinear",
            antialias=True,
        )
        image = tf.cast(image, tf.float32)
    except (tf.errors.InvalidArgumentError, ValueError, TypeError) as error:
        raise InvalidImageError("The uploaded file is not a decodable image.") from error
    if image.shape.rank != 3 or image.shape[-1] != 3:
        raise InvalidImageError("The uploaded image must decode as RGB.")
    return np.expand_dims(image.numpy(), axis=0)


def predict(
    model: tf.keras.Model,
    image_bytes: bytes,
    threshold: float,
) -> dict[str, Any]:
    batch = preprocess_image(image_bytes)
    output = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
    if output.size != 1 or not np.isfinite(output[0]):
        raise RuntimeError("The model returned an invalid prediction.")
    ai_probability = float(np.clip(output[0], 0.0, 1.0))
    is_ai = ai_probability >= threshold
    return {
        "label": "AI-generated" if is_ai else "Real",
        "confidence": ai_probability if is_ai else 1.0 - ai_probability,
        "ai_probability": ai_probability,
    }


def environment_model_config() -> tuple[Path, float]:
    model_path = os.environ.get("MODEL_PATH", "").strip()
    threshold_text = os.environ.get("THRESHOLD", "").strip()
    if not model_path:
        raise RuntimeError("MODEL_PATH must point to a trained VisionGuard .keras model.")
    if not threshold_text:
        raise RuntimeError("THRESHOLD must be set to the validation-selected threshold.")
    try:
        threshold = float(threshold_text)
    except ValueError as error:
        raise RuntimeError("THRESHOLD must be a number between 0 and 1.") from error
    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError("THRESHOLD must be between 0 and 1.")
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path, threshold
