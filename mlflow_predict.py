"""Load the local VisionGuard champion alias and classify one image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import mlflow
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageOps

import visionguard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model-uri", default="models:/VisionGuard@champion"
    )
    parser.add_argument("--tracking-uri")
    args = parser.parse_args()

    load_dotenv(visionguard.ROOT / ".env")
    tracking_uri = (
        args.tracking_uri
        or f"sqlite:///{(visionguard.ROOT / 'mlflow_local.db').as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    registry = json.loads(
        (visionguard.OUTPUT_DIR / "mlflow_registry_results.json").read_text(
            encoding="utf-8"
        )
    )
    threshold = float(registry["threshold"])
    model = mlflow.pyfunc.load_model(args.model_uri)
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = image.resize(visionguard.IMAGE_SIZE, Image.Resampling.BILINEAR)
        batch = np.asarray(image, dtype=np.float32)[None, ...]
    prediction = np.asarray(model.predict(batch)).reshape(-1)
    probability = float(prediction[0])
    label = "AI-generated" if probability >= threshold else "Real"
    print(
        f"Model: {args.model_uri}\n"
        f"Image: {args.image}\n"
        f"AI probability: {probability:.6f}\n"
        f"Threshold: {threshold:.6f}\n"
        f"Prediction: {label}"
    )


if __name__ == "__main__":
    main()
