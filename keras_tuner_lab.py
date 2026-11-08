"""Keras Tuner demonstrations: Fashion-MNIST toy search and VisionGuard search."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import keras_tuner as kt
import tensorflow as tf

import visionguard


OUTPUT_DIR = visionguard.OUTPUT_DIR
TUNER_DIR = OUTPUT_DIR / "keras_tuner"


def save_trials(tuner: kt.Tuner, name: str, max_trials: int) -> list[dict[str, Any]]:
    trials = tuner.oracle.get_best_trials(num_trials=max_trials)
    rows = []
    for trial in trials:
        rows.append(
            {
                **trial.hyperparameters.values,
                "objective": float(trial.score) if trial.score is not None else None,
                "trial_id": trial.trial_id,
            }
        )
    if not rows:
        raise RuntimeError(f"Keras Tuner produced no completed trials for {name}.")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (OUTPUT_DIR / f"{name}_tuner_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_DIR / f"{name}_tuner_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return rows


def run_toy(max_trials: int, epochs: int) -> dict[str, Any]:
    tf.keras.utils.set_random_seed(visionguard.SEED)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train, x_validation = x_train[:-6000], x_train[-6000:]
    y_train, y_validation = y_train[:-6000], y_train[-6000:]

    def build_model(hp: kt.HyperParameters) -> tf.keras.Model:
        units = hp.Int("units", min_value=32, max_value=256, step=32)
        learning_rate = hp.Float(
            "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
        )
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(28, 28)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ],
            name="fashion_mnist_toy",
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    tuner = kt.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=max_trials,
        seed=visionguard.SEED,
        directory=str(TUNER_DIR),
        project_name="fashion_mnist",
        overwrite=True,
    )
    tuner.search(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=128,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=2, mode="max"
            )
        ],
        verbose=2,
    )
    rows = save_trials(tuner, "toy", max_trials)
    best_model = tuner.get_best_models(num_models=1)[0]
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=0)
    results = {
        "dataset": "Fashion-MNIST",
        "train_examples": int(len(x_train)),
        "validation_examples": int(len(x_validation)),
        "test_examples": int(len(x_test)),
        "trials": len(rows),
        "best_settings": tuner.get_best_hyperparameters(1)[0].values,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }
    (OUTPUT_DIR / "toy_tuner_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def run_visionguard(max_trials: int, epochs: int) -> dict[str, Any]:
    visionguard.set_seeds()
    visionguard.create_splits()
    train_data = visionguard.make_dataset("train", training=True)
    validation_data = visionguard.make_dataset("validation")

    def build_model(hp: kt.HyperParameters) -> tf.keras.Model:
        learning_rate = hp.Float(
            "learning_rate", min_value=1e-5, max_value=1e-3, sampling="log"
        )
        dropout = hp.Float("dropout", min_value=0.20, max_value=0.50, step=0.10)
        unfrozen_layers = hp.Int("unfrozen_layers", min_value=0, max_value=15, step=5)
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(*visionguard.IMAGE_SIZE, 3),
        )
        backbone.trainable = True
        first_trainable_index = len(backbone.layers) - unfrozen_layers
        for index, layer in enumerate(backbone.layers):
            layer.trainable = (
                index >= first_trainable_index
                and not isinstance(layer, tf.keras.layers.BatchNormalization)
            )
        inputs = tf.keras.Input(shape=(*visionguard.IMAGE_SIZE, 3))
        x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
        x = backbone(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs, outputs, name="visionguard_tuned_mobilenetv2")
        visionguard.compile_model(model, learning_rate=learning_rate)
        return model

    tuner = kt.RandomSearch(
        hypermodel=build_model,
        objective="val_loss",
        max_trials=max_trials,
        seed=visionguard.SEED,
        directory=str(TUNER_DIR),
        project_name="visionguard",
        overwrite=True,
    )
    tuner.search(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        shuffle=False,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, mode="min", restore_best_weights=True
            )
        ],
        verbose=2,
    )
    rows = save_trials(tuner, "visionguard", max_trials)
    best = tuner.get_best_hyperparameters(1)[0]
    summary = {
        "dataset": "VisionGuard archive",
        "trials": len(rows),
        "objective": "minimum validation binary cross-entropy",
        "best_validation_loss": float(tuner.oracle.get_best_trials(1)[0].score),
        "best_settings": best.values,
        "epoch_limit_per_trial": epochs,
        "unfrozen_layers_note": (
            "Number of final MobileNetV2 layer objects opened for tuning; "
            "BatchNormalization layers remain frozen."
        ),
    }
    (OUTPUT_DIR / "visionguard_tuner_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("toy", "applied", "all"), default="all")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if args.trials < 8:
        parser.error("--trials must be at least 8 to meet the assignment requirement.")
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.mode in ("toy", "all"):
        results["toy"] = run_toy(args.trials, args.epochs)
    if args.mode in ("applied", "all"):
        results["visionguard"] = run_visionguard(args.trials, args.epochs)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
