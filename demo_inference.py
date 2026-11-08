"""Print predictions for one real and one AI test image."""

from __future__ import annotations

import csv
from pathlib import Path

from dotenv import load_dotenv

from inference import ROOT, environment_model_config, load_model, predict


def main() -> None:
    load_dotenv(ROOT / ".env")
    model_path, threshold = environment_model_config()
    split_file = ROOT / "visionguard_outputs" / "splits" / "test.csv"
    if not split_file.is_file():
        raise FileNotFoundError(f"Test split list is missing: {split_file}")

    known_images: dict[int, Path] = {}
    with split_file.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            label = int(row["label"])
            if label not in known_images:
                known_images[label] = ROOT / row["path"]
            if len(known_images) == 2:
                break
    if set(known_images) != {0, 1}:
        raise RuntimeError("The test split must contain both Real and AI images.")

    model = load_model(model_path)
    for label, path in known_images.items():
        if not path.is_file():
            raise FileNotFoundError(f"Known test image is missing: {path}")
        result = predict(model, path.read_bytes(), threshold)
        actual = "Real" if label == 0 else "AI-generated"
        print(
            f"Image: {path.name}\n"
            f"Actual label: {actual}\n"
            f"Prediction: {result['label']}\n"
            f"Confidence: {result['confidence']:.6f}\n"
            f"AI probability: {result['ai_probability']:.6f}\n"
            f"Threshold: {threshold:.6f}\n",
            flush=True,
        )


if __name__ == "__main__":
    main()
