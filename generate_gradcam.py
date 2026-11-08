"""Generate optional Grad-CAM examples on validation data without re-evaluating test data."""

from __future__ import annotations

import json

import tensorflow as tf

import visionguard


def main() -> None:
    results_path = visionguard.OUTPUT_DIR / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError("Run visionguard.py before generating Grad-CAM.")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    model_name = results["selected_model"]
    model_path = visionguard.OUTPUT_DIR / f"{model_name}_best.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Selected model checkpoint not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    backbone = (
        model.get_layer("mobilenetv2_1.00_160")
        if model_name == "transfer_finetune"
        else None
    )
    labels, probabilities = visionguard.infer_once(
        model, visionguard.make_dataset("validation")
    )
    threshold = results["evaluation"][model_name]["threshold"]
    visionguard.make_gradcam_figure(
        model,
        backbone,
        labels,
        probabilities,
        threshold,
        split_name="validation",
    )
    print(
        f"Grad-CAM figure saved to "
        f"{visionguard.OUTPUT_DIR / 'gradcam_examples.png'} using validation examples."
    )


if __name__ == "__main__":
    main()
