"""Local/DagsHub MLflow exercises for Task 3 (toy, VisionGuard, comparison, registry)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.metrics import roc_curve

import visionguard


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = visionguard.OUTPUT_DIR
LOCAL_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow_local.db').as_posix()}"
LOCAL_ARTIFACT_ROOT = ROOT / "mlartifacts"
EXPERIMENT_NAME = "VisionGuard-Task-3"
REGISTERED_MODEL_NAME = "VisionGuard"

COMPARISON_VARIANTS = (
    ("lr_1e-5", 1e-5),
    ("lr_3e-5", 3e-5),
    ("lr_1e-4", 1e-4),
    ("lr_3e-4", 3e-4),
    ("lr_1e-3", 1e-3),
    ("lr_3e-3", 3e-3),
)


def git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def git_metadata() -> dict[str, str]:
    commit = git_value("rev-parse", "HEAD")
    dirty = bool(git_value("status", "--porcelain"))
    return {
        "git_commit": commit or "unavailable",
        "git_worktree": "dirty" if dirty else "clean",
    }


def dvc_data_version() -> str:
    dvc_file = ROOT / "archive.dvc"
    if not dvc_file.is_file():
        return "archive.dvc-missing"
    match = re.search(r"(?m)^\s*-\s*md5:\s*([0-9a-f]+)", dvc_file.read_text())
    return match.group(1) if match else "archive.dvc-hash-missing"


def configure_tracking(destination: str) -> str:
    load_dotenv(ROOT / ".env")
    if destination == "dagshub":
        uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
        username = os.environ.get("MLFLOW_TRACKING_USERNAME", "").strip()
        token = os.environ.get("MLFLOW_TRACKING_PASSWORD", "").strip()
        if not (uri and username and token):
            raise RuntimeError(
                "DagsHub tracking needs MLFLOW_TRACKING_URI, "
                "MLFLOW_TRACKING_USERNAME, and MLFLOW_TRACKING_PASSWORD. "
                "Set them in an ignored .env file or your process environment; "
                "do not put the token in source code."
            )
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    else:
        uri = LOCAL_TRACKING_URI
        LOCAL_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri


def standard_tags(task: str, logging_method: str) -> dict[str, str]:
    return {
        **git_metadata(),
        "dvc_data_version": dvc_data_version(),
        "seed": str(visionguard.SEED),
        "task": task,
        "logging_method": logging_method,
    }


def write_history_plot(
    path: Path,
    history: dict[str, list[float]],
    title: str,
) -> None:
    epochs = range(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for metric, axis, label in (
        ("loss", axes[0], "Loss"),
        ("accuracy", axes[1], "Accuracy"),
    ):
        axis.plot(epochs, history[metric], label="Train")
        val_key = f"val_{metric}"
        if val_key in history:
            axis.plot(epochs, history[val_key], label="Validation")
        axis.set_title(label)
        axis.set_xlabel("Epoch")
        axis.legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def log_manual_history(history: dict[str, list[float]]) -> None:
    for epoch, values in enumerate(zip(*history.values()), start=1):
        mlflow.log_metrics(
            {name: float(value) for name, value in zip(history, values)},
            step=epoch,
        )


def run_toy(destination: str, learning_rates: list[float], epochs: int) -> None:
    tracking_uri = configure_tracking(destination)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train, x_validation = x_train[:-6000], x_train[-6000:]
    y_train, y_validation = y_train[:-6000], y_train[-6000:]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for index, learning_rate in enumerate(learning_rates):
        random.seed(visionguard.SEED)
        np.random.seed(visionguard.SEED)
        tf.keras.utils.set_random_seed(visionguard.SEED)
        with mlflow.start_run(run_name=f"fashion_lr_{learning_rate:g}") as run:
            mlflow.log_params(
                {
                    "dataset": "Fashion-MNIST",
                    "learning_rate": learning_rate,
                    "hidden_units": 128,
                    "batch_size": 128,
                    "epochs": epochs,
                    "seed": visionguard.SEED,
                }
            )
            mlflow.set_tags(
                {
                    **standard_tags("3.1-toy", "manual"),
                    "learning_rate_comparison": "true",
                    "comparison_index": str(index + 1),
                }
            )
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(28, 28)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dropout(0.20),
                    tf.keras.layers.Dense(10, activation="softmax"),
                ],
                name="fashion_mnist_toy",
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            result = model.fit(
                x_train,
                y_train,
                validation_data=(x_validation, y_validation),
                batch_size=128,
                epochs=epochs,
                verbose=2,
            )
            history = {
                name: [float(value) for value in values]
                for name, values in result.history.items()
            }
            log_manual_history(history)
            test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
            mlflow.log_metrics(
                {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}
            )
            plot_path = OUTPUT_DIR / f"mlflow_toy_{index + 1}_curves.png"
            write_history_plot(plot_path, history, f"Fashion-MNIST lr={learning_rate:g}")
            mlflow.log_artifact(str(plot_path), artifact_path="plots")
            model_path = OUTPUT_DIR / f"mlflow_toy_{index + 1}.keras"
            model.save(model_path)
            mlflow.log_artifact(str(model_path), artifact_path="models")
            print(
                f"Toy run {run.info.run_id}: lr={learning_rate:g}, "
                f"test_accuracy={test_accuracy:.4f}, tracking={tracking_uri}"
            )


def load_visionguard_data() -> tuple[
    tf.keras.Model,
    tf.keras.Model,
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    started = time.perf_counter()
    print(
        "Preparing VisionGuard data and frozen MobileNetV2 feature extractor...",
        flush=True,
    )
    visionguard.set_seeds()
    visionguard.create_splits()
    full_model, backbone = visionguard.build_transfer_model()
    backbone.trainable = False
    feature_model = tf.keras.Model(
        full_model.input,
        full_model.get_layer("global_average_pool").output,
        name="frozen_mobilenet_feature_extractor",
    )

    def extract(split_name: str, training: bool = False) -> tuple[np.ndarray, np.ndarray]:
        dataset = visionguard.make_dataset(split_name, training=training)
        total_batches = int(tf.data.experimental.cardinality(dataset).numpy())
        print(
            f"Extracting MobileNetV2 features for {split_name}: "
            f"{total_batches} batches...",
            flush=True,
        )
        progress = tf.keras.utils.Progbar(total_batches, stateful_metrics=[])
        feature_batches = []
        label_batches = []
        for batch_index, (images, labels) in enumerate(dataset, start=1):
            feature_batches.append(feature_model(images, training=False).numpy())
            label_batches.append(labels.numpy().reshape(-1).astype(np.int32))
            progress.update(batch_index)
        features = np.concatenate(feature_batches)
        labels = np.concatenate(label_batches)
        print(
            f"Finished {split_name}: {len(labels)} images, "
            f"feature shape={features.shape[1:]}",
            flush=True,
        )
        return features, labels

    train_features, train_labels = extract("train", training=True)
    validation_features, validation_labels = extract("validation")
    test_features, test_labels = extract("test")
    print(
        f"VisionGuard data prepared in {time.perf_counter() - started:.1f}s. "
        "The MobileNetV2 backbone is frozen; each run will train its classification head.",
        flush=True,
    )
    return (
        full_model,
        backbone,
        (train_features, train_labels),
        (validation_features, validation_labels),
        (test_features, test_labels),
    )


def make_trainable_head(
    full_model: tf.keras.Model, backbone: tf.keras.Model
) -> tf.keras.Model:
    feature_input = tf.keras.Input(
        shape=(int(backbone.output.shape[-1]),), name="pooled_features"
    )
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_features")(
        feature_input
    )
    x = tf.keras.layers.Dropout(0.30, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", name="fake_probability"
    )(x)
    return tf.keras.Model(feature_input, outputs, name="visionguard_trainable_head")


def transfer_head_weights(
    head: tf.keras.Model, full_model: tf.keras.Model
) -> None:
    for layer_name in ("dense_features", "fake_probability"):
        full_model.get_layer(layer_name).set_weights(
            head.get_layer(layer_name).get_weights()
        )


def save_confusion_plot(path: Path, matrix: np.ndarray, title: str) -> None:
    figure, axis = plt.subplots(figsize=(4.5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["Predicted real", "Predicted AI"])
    axis.set_yticks([0, 1], labels=["Actual real", "Actual AI"])
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_roc_plot(path: Path, labels: np.ndarray, scores: np.ndarray) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    figure, axis = plt.subplots(figsize=(4.5, 4))
    axis.plot(false_positive_rate, true_positive_rate)
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("VisionGuard ROC")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_comparison_plot(rows: list[dict[str, Any]], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: row["learning_rate"])
    labels = [f"{row['learning_rate']:g}" for row in ordered]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(labels, [row["validation_weighted_error_cost"] for row in ordered])
    axes[0].set_title("Validation weighted error cost (lower is better)")
    axes[0].set_xlabel("Learning rate")
    axes[0].set_ylabel("2 x FP + FN")
    axes[1].plot(labels, [row["f1"] for row in ordered], "o-", label="Validation F1")
    axes[1].plot(
        labels, [row["roc_auc"] for row in ordered], "s-", label="Validation ROC-AUC"
    )
    axes[1].set_title("Validation ranking and threshold metrics")
    axes[1].set_xlabel("Learning rate")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def log_required_visionguard_artifacts(
    confusion_path: Path, roc_path: Path
) -> None:
    required = [
        confusion_path,
        roc_path,
        visionguard.SPLIT_DIR / "train.csv",
        visionguard.SPLIT_DIR / "validation.csv",
        visionguard.SPLIT_DIR / "test.csv",
        ROOT / "requirements_frozen.txt",
    ]
    for artifact in required:
        if not artifact.is_file():
            raise FileNotFoundError(f"Required MLflow artifact is missing: {artifact}")
        mlflow.log_artifact(str(artifact), artifact_path="assignment_artifacts")


def log_full_model(
    full_model: tf.keras.Model,
    images: np.ndarray,
) -> None:
    predictions = full_model.predict(images[:2], verbose=0)
    signature = infer_signature(images[:2], predictions)
    mlflow.tensorflow.log_model(
        full_model,
        name="visionguard_model",
        signature=signature,
        input_example=images[:1],
        pip_requirements=[
            f"tensorflow=={tf.__version__}",
            f"numpy=={np.__version__}",
        ],
    )


def train_logged_head(
    destination: str,
    logging_method: str,
    learning_rate: float,
    epochs: int,
    cached: tuple[
        tf.keras.Model,
        tf.keras.Model,
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ]
    | None = None,
) -> tuple[str, tf.keras.Model, float, dict[str, float]]:
    print(
        f"\nStarting VisionGuard training: method={logging_method}, "
        f"destination={destination}, epochs={epochs}, learning_rate={learning_rate:g}",
        flush=True,
    )
    configure_tracking(destination)
    if cached is None:
        cached = load_visionguard_data()
    full_model, backbone, train, validation, test = cached
    train_features, train_labels = train
    validation_features, validation_labels = validation
    test_features, test_labels = test
    validation_images, _ = next(iter(visionguard.make_dataset("validation")))
    validation_images = validation_images.numpy()
    visionguard.set_seeds()
    head = make_trainable_head(full_model, backbone)
    visionguard.compile_model(
        head, learning_rate=learning_rate
    )

    with mlflow.start_run(
        run_name=f"visionguard_{logging_method}_lr_{learning_rate:g}"
    ) as run:
        mlflow.log_params(
            {
                "model": "MobileNetV2 frozen backbone + trainable head",
                "learning_rate": learning_rate,
                "dropout": 0.30,
                "unfrozen_layers": 0,
                "image_height": visionguard.IMAGE_SIZE[0],
                "image_width": visionguard.IMAGE_SIZE[1],
                "batch_size": visionguard.BATCH_SIZE,
                "epochs": epochs,
                "augmentation": "training-only random flip/rotation/zoom/contrast",
                "seed": visionguard.SEED,
            }
        )
        mlflow.set_tags(standard_tags("3.3-visionguard", logging_method))
        callbacks: list[tf.keras.callbacks.Callback] = []
        if logging_method == "manual":
            class MlflowEpochLogger(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
                    if not logs:
                        return
                    mlflow.log_metrics(
                        {
                            key: float(value)
                            for key, value in logs.items()
                            if value is not None
                        },
                        step=epoch + 1,
                    )

            callbacks.append(MlflowEpochLogger())
        else:
            mlflow.tensorflow.autolog(
                log_models=True,
                log_every_epoch=True,
                checkpoint=False,
                extra_tags={"logging_method": "tensorflow_autolog"},
            )

        print(
            f"Training classification head for {logging_method} run "
            f"{run.info.run_id} (MobileNetV2 backbone frozen)...",
            flush=True,
        )
        history_obj = head.fit(
            train_features,
            train_labels,
            validation_data=(validation_features, validation_labels),
            batch_size=visionguard.BATCH_SIZE,
            epochs=epochs,
            callbacks=callbacks,
            shuffle=False,
            verbose=2,
        )
        print(
            f"Training finished for {logging_method}; computing validation-selected "
            "threshold and held-out test metrics...",
            flush=True,
        )
        history = {
            key: [float(value) for value in values]
            for key, values in history_obj.history.items()
        }
        if logging_method == "manual":
            history_path = OUTPUT_DIR / "manual_visionguard_history.csv"
            with history_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", *history])
                for epoch in range(len(history["loss"])):
                    writer.writerow(
                        [epoch + 1, *(history[key][epoch] for key in history)]
                    )
            mlflow.log_artifact(str(history_path), artifact_path="training")
        else:
            mlflow.tensorflow.autolog(disable=True)

        transfer_head_weights(head, full_model)
        assembled = tf.keras.Model(
            full_model.input, full_model.output, name="visionguard_transfer_model"
        )
        validation_scores = head.predict(validation_features, verbose=0).reshape(-1)
        threshold, validation_cost = visionguard.choose_threshold(
            validation_labels, validation_scores
        )
        test_scores = head.predict(test_features, verbose=0).reshape(-1)
        test_metrics = visionguard.calculate_metrics(
            test_labels, test_scores, threshold
        )
        mlflow.log_metrics(
            {
                "validation_threshold": float(threshold),
                "validation_weighted_error_cost": int(
                    validation_cost["weighted_error_cost"]
                ),
                "best_validation_loss": float(min(history["val_loss"])),
                **{
                    f"test_{key}": value
                    for key, value in test_metrics.items()
                    if isinstance(value, (int, float))
                },
            }
        )
        mlflow.set_tag("validation_threshold", f"{threshold:.8f}")

        confusion_path = OUTPUT_DIR / f"visionguard_{logging_method}_confusion.png"
        save_confusion_plot(
            confusion_path,
            np.asarray(test_metrics["confusion_matrix"]),
            f"VisionGuard {logging_method} test confusion",
        )
        roc_path = OUTPUT_DIR / f"visionguard_{logging_method}_roc.png"
        save_roc_plot(roc_path, test_labels, test_scores)
        model_path = OUTPUT_DIR / f"visionguard_{logging_method}.keras"
        assembled.save(model_path)
        mlflow.log_artifact(str(model_path), artifact_path="models")
        plot_path = OUTPUT_DIR / f"visionguard_{logging_method}_curves.png"
        write_history_plot(plot_path, history, f"VisionGuard ({logging_method})")
        mlflow.log_artifact(str(plot_path), artifact_path="plots")

        log_full_model(assembled, validation_images)
        print(
            f"Uploading model and evaluation artifacts to {destination}...",
            flush=True,
        )
        log_required_visionguard_artifacts(confusion_path, roc_path)
        result = {
            "threshold": float(threshold),
            "validation_weighted_error_cost": int(
                validation_cost["weighted_error_cost"]
            ),
            "validation_loss": float(min(history["val_loss"])),
        }
        return run.info.run_id, assembled, threshold, result


def run_visionguard_logging(destination: str, epochs: int) -> None:
    cached = load_visionguard_data()
    for method in ("manual", "autolog"):
        run_id, _, threshold, metrics = train_logged_head(
            destination,
            method,
            learning_rate=1e-4,
            epochs=epochs,
            cached=cached,
        )
        print(
            f"Completed {method} VisionGuard run={run_id}, threshold={threshold:.4f}, "
            f"validation_cost={metrics['validation_weighted_error_cost']}",
            flush=True,
        )


def run_comparison(destination: str, epochs: int) -> dict[str, Any]:
    configure_tracking(destination)
    cached = load_visionguard_data()
    full_model, backbone, train, validation, test = cached
    train_features, train_labels = train
    validation_features, validation_labels = validation
    test_features, test_labels = test
    validation_images, _ = next(iter(visionguard.make_dataset("validation")))
    all_runs: list[dict[str, Any]] = []
    model_by_run: dict[str, tf.keras.Model] = {}

    for name, learning_rate in COMPARISON_VARIANTS:
        visionguard.set_seeds()
        head = make_trainable_head(full_model, backbone)
        visionguard.compile_model(head, learning_rate)
        with mlflow.start_run(run_name=f"comparison_{name}") as run:
            mlflow.log_params(
                {
                    "comparison_group": "learning_rate_only",
                    "model": "MobileNetV2 frozen backbone + trainable head",
                    "learning_rate": learning_rate,
                    "dropout": 0.30,
                    "unfrozen_layers": 0,
                    "image_height": 160,
                    "image_width": 160,
                    "augmentation": "same training-only augmentation",
                    "epochs": epochs,
                    "seed": visionguard.SEED,
                    "batch_size": visionguard.BATCH_SIZE,
                }
            )
            mlflow.set_tags(standard_tags("3.5-controlled-comparison", "manual"))
            history_obj = head.fit(
                train_features,
                train_labels,
                validation_data=(validation_features, validation_labels),
                batch_size=visionguard.BATCH_SIZE,
                epochs=epochs,
                shuffle=False,
                verbose=2,
            )
            history = {
                key: [float(value) for value in values]
                for key, values in history_obj.history.items()
            }
            log_manual_history(history)
            transfer_head_weights(head, full_model)
            run_model = tf.keras.Model(
                full_model.input,
                full_model.output,
                name="visionguard_transfer_model",
            )
            log_full_model(run_model, validation_images.numpy())
            validation_scores = head.predict(validation_features, verbose=0).reshape(-1)
            threshold, validation_cost = visionguard.choose_threshold(
                validation_labels, validation_scores
            )
            validation_metrics = visionguard.calculate_metrics(
                validation_labels, validation_scores, threshold
            )
            mlflow.log_metrics(
                {
                    "validation_threshold": float(threshold),
                    "validation_weighted_error_cost": int(
                        validation_cost["weighted_error_cost"]
                    ),
                    "validation_accuracy": validation_metrics["accuracy"],
                    "validation_f1": validation_metrics["f1"],
                    "validation_roc_auc": validation_metrics["roc_auc"],
                    "best_validation_loss": float(min(history["val_loss"])),
                }
            )
            plot_path = OUTPUT_DIR / f"comparison_{name}_curves.png"
            write_history_plot(plot_path, history, f"VisionGuard {name}")
            mlflow.log_artifact(str(plot_path), artifact_path="plots")
            all_runs.append(
                {
                    "run_id": run.info.run_id,
                    "run_name": name,
                    "learning_rate": learning_rate,
                    "threshold": threshold,
                    **validation_metrics,
                    "validation_weighted_error_cost": validation_cost[
                        "weighted_error_cost"
                    ],
                    "best_validation_loss": float(min(history["val_loss"])),
                }
            )
            model_by_run[run.info.run_id] = head

    selected = min(
        all_runs,
        key=lambda row: (
            row["validation_weighted_error_cost"],
            -row["f1"],
            -row["roc_auc"],
        ),
    )
    selected_head = model_by_run[selected["run_id"]]
    transfer_head_weights(selected_head, full_model)
    test_scores = selected_head.predict(test_features, verbose=0).reshape(-1)
    test_metrics = visionguard.calculate_metrics(
        test_labels, test_scores, selected["threshold"]
    )
    selected_model = tf.keras.Model(
        full_model.input, full_model.output, name="visionguard_transfer_model"
    )
    with mlflow.start_run(run_id=selected["run_id"]):
        mlflow.log_metrics(
            {f"test_{key}": value for key, value in test_metrics.items() if isinstance(value, (int, float))}
        )
        model_path = OUTPUT_DIR / "visionguard_comparison_champion.keras"
        selected_model.save(model_path)
        mlflow.log_artifact(str(model_path), artifact_path="models")
        confusion_path = OUTPUT_DIR / "mlflow_visionguard_confusion.png"
        matrix = np.asarray(test_metrics["confusion_matrix"])
        save_confusion_plot(confusion_path, matrix, "Selected run test confusion matrix")
        roc_path = OUTPUT_DIR / "mlflow_visionguard_roc.png"
        save_roc_plot(roc_path, test_labels, test_scores)
        mlflow.log_artifact(str(confusion_path), artifact_path="evaluation")
        mlflow.log_artifact(str(roc_path), artifact_path="evaluation")
        mlflow.log_params(
            {
                "model_registry_image_size": "160x160",
                "selected_validation_threshold": selected["threshold"],
            }
        )
        log_required_visionguard_artifacts(confusion_path, roc_path)

    summary = {
        "experiment": EXPERIMENT_NAME,
        "comparison_factor": "learning_rate only",
        "epochs_per_run": epochs,
        "runs": all_runs,
        "selected_run_id": selected["run_id"],
        "selected_run_name": selected["run_name"],
        "selected_validation_weighted_error_cost": selected[
            "validation_weighted_error_cost"
        ],
        "selected_test_metrics": test_metrics,
    }
    summary_path = OUTPUT_DIR / "mlflow_comparison_results.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUTPUT_DIR / "mlflow_comparison_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(all_runs[0]))
        writer.writeheader()
        writer.writerows(all_runs)
    save_comparison_plot(all_runs, OUTPUT_DIR / "mlflow_comparison_chart.png")
    print(json.dumps(summary, indent=2))
    return summary


def register_two_versions(
    destination: str, selected_run_id: str | None = None
) -> dict[str, Any]:
    configure_tracking(destination)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment does not exist: {EXPERIMENT_NAME}")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED' and tags.task = '3.5-controlled-comparison'",
        order_by=["metrics.validation_weighted_error_cost ASC"],
    )
    if runs.empty:
        raise RuntimeError(
            "No completed comparison runs found; run the six-run comparison first."
        )
    if selected_run_id is None:
        selected_run_id = str(runs.iloc[0]["run_id"])
    if selected_run_id not in set(runs["run_id"].astype(str)):
        raise ValueError(f"Selected run is not in the comparison: {selected_run_id}")
    second_run = next(
        (str(run_id) for run_id in runs["run_id"].astype(str) if run_id != selected_run_id),
        None,
    )
    if second_run is None:
        raise RuntimeError("Need at least two comparison runs to create two model versions.")

    client = mlflow.MlflowClient()
    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(REGISTERED_MODEL_NAME)

    registered = []
    for run_id in (selected_run_id, second_run):
        model_uri = f"runs:/{run_id}/visionguard_model"
        model_version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        registered.append(
            {
                "run_id": run_id,
                "version": str(model_version.version),
            }
        )
    champion_version = next(
        item["version"] for item in registered if item["run_id"] == selected_run_id
    )
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, "champion", champion_version
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, champion_version, "selection", "lowest validation error cost"
    )
    result = {
        "registered_model": REGISTERED_MODEL_NAME,
        "champion_alias_version": champion_version,
        "versions": registered,
        "threshold": float(
            mlflow.get_run(selected_run_id).data.metrics["validation_threshold"]
        ),
    }
    (OUTPUT_DIR / "mlflow_registry_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=(
            "toy-local",
            "toy-dagshub",
            "visionguard-local",
            "visionguard-dagshub",
            "compare-local",
            "compare-dagshub",
            "register-local",
        ),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--selected-run-id")
    parser.add_argument("--learning-rates", nargs="+", type=float)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.task.startswith("toy-"):
        rates = args.learning_rates or [1e-3, 1e-4]
        if len(rates) < 2:
            parser.error("Toy tracking needs at least two learning rates.")
        destination = "dagshub" if args.task.endswith("dagshub") else "local"
        run_toy(destination, rates, args.epochs)
    elif args.task.startswith("visionguard-"):
        destination = "dagshub" if args.task.endswith("dagshub") else "local"
        run_visionguard_logging(destination, args.epochs)
    elif args.task.startswith("compare-"):
        destination = "dagshub" if args.task.endswith("dagshub") else "local"
        run_comparison(destination, args.epochs)
    else:
        register_two_versions("local", args.selected_run_id)


if __name__ == "__main__":
    main()
