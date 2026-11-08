"""VisionGuard image pipeline, CNN/transfer training, and single-pass evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "archive"
OUTPUT_DIR = ROOT / "visionguard_outputs"
SPLIT_DIR = OUTPUT_DIR / "splits"
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42
CLASS_NAMES = {0: "RealArt", 1: "AiArtData"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_splits(force: bool = False) -> dict[str, Any]:
    """Create reproducible 70/15/15 lists, removing byte-identical duplicates."""
    split_files = {name: SPLIT_DIR / f"{name}.csv" for name in ("train", "validation", "test")}
    if not force and all(path.exists() for path in split_files.values()):
        return read_split_counts()

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    unique_by_hash: dict[str, tuple[str, int, str]] = {}
    discovered = 0
    duplicate_count = 0
    for folder, label in (("RealArt", 0), ("AiArtData", 1)):
        class_dir = DATA_DIR / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class directory not found: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            discovered += 1
            with Image.open(path) as image:
                image.verify()
            file_hash = sha256_file(path)
            if file_hash in unique_by_hash:
                duplicate_count += 1
                continue
            unique_by_hash[file_hash] = (path.relative_to(ROOT).as_posix(), label, folder)

    records = list(unique_by_hash.values())
    paths = np.asarray([row[0] for row in records])
    labels = np.asarray([row[1] for row in records], dtype=np.int32)
    classes = np.asarray([row[2] for row in records])
    train_paths, remainder_paths, train_labels, remainder_labels, train_classes, remainder_classes = (
        train_test_split(
            paths,
            labels,
            classes,
            test_size=0.30,
            random_state=SEED,
            stratify=labels,
        )
    )
    validation_paths, test_paths, validation_labels, test_labels, validation_classes, test_classes = (
        train_test_split(
            remainder_paths,
            remainder_labels,
            remainder_classes,
            test_size=0.50,
            random_state=SEED,
            stratify=remainder_labels,
        )
    )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    split_data = {
        "train": (train_paths, train_labels, train_classes),
        "validation": (validation_paths, validation_labels, validation_classes),
        "test": (test_paths, test_labels, test_classes),
    }
    for name, (split_paths, split_labels, split_classes) in split_data.items():
        with (SPLIT_DIR / f"{name}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("path", "label", "class_name"))
            writer.writerows(zip(split_paths, split_labels, split_classes))

    counts = {
        name: {
            "images": int(len(data[0])),
            "real": int(np.sum(data[1] == 0)),
            "ai": int(np.sum(data[1] == 1)),
        }
        for name, data in split_data.items()
    }
    dataset_info = {
        "seed": SEED,
        "image_size": list(IMAGE_SIZE),
        "split_percentages": {"train": 70, "validation": 15, "test": 15},
        "discovered_files": discovered,
        "duplicate_files_removed": duplicate_count,
        "unique_images": len(records),
        "counts": counts,
        "labels": {"0": "RealArt (real)", "1": "AiArtData (AI-generated)"},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2), encoding="utf-8"
    )
    return dataset_info


def read_split_counts() -> dict[str, Any]:
    info_path = OUTPUT_DIR / "dataset_info.json"
    if info_path.exists():
        return json.loads(info_path.read_text(encoding="utf-8"))
    counts: dict[str, Any] = {}
    for name in ("train", "validation", "test"):
        with (SPLIT_DIR / f"{name}.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        counts[name] = {
            "images": len(rows),
            "real": sum(int(row["label"]) == 0 for row in rows),
            "ai": sum(int(row["label"]) == 1 for row in rows),
        }
    info = {
        "seed": SEED,
        "image_size": list(IMAGE_SIZE),
        "split_percentages": {"train": 70, "validation": 15, "test": 15},
        "counts": counts,
        "labels": {"0": "RealArt (real)", "1": "AiArtData (AI-generated)"},
    }
    return info


def read_split(name: str) -> tuple[list[str], np.ndarray]:
    with (SPLIT_DIR / f"{name}.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    paths = [str(ROOT / Path(row["path"])) for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    return paths, labels


def decode_and_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear", antialias=True)
    return tf.cast(image, tf.float32), tf.cast(label, tf.float32)


def make_augmenter() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.08, seed=SEED + 1),
            tf.keras.layers.RandomZoom(0.10, seed=SEED + 2),
            tf.keras.layers.RandomContrast(0.10, seed=SEED + 3),
        ],
        name="training_augmentation",
    )


def make_dataset(name: str, training: bool = False) -> tf.data.Dataset:
    paths, labels = read_split(name)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True
        )
    dataset = dataset.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        augmenter = make_augmenter()
        dataset = dataset.map(
            lambda image, label: (
                tf.clip_by_value(augmenter(image, training=True), 0.0, 255.0),
                label,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def save_augmented_batch(dataset: tf.data.Dataset) -> None:
    images, labels = next(iter(dataset))
    figure, axes = plt.subplots(4, 4, figsize=(10, 10))
    for index, axis in enumerate(axes.flat):
        image = tf.clip_by_value(images[index], 0, 255).numpy().astype(np.uint8)
        axis.imshow(image)
        axis.set_title("AI-generated" if int(labels[index]) == 1 else "Real")
        axis.axis("off")
    figure.suptitle("Training batch after random augmentation")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "augmented_batch.png", dpi=150)
    plt.close(figure)


def build_baseline() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    for index, filters in enumerate((32, 64, 128), start=1):
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False, name=f"conv{index}"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn{index}")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu{index}")(x)
        x = tf.keras.layers.MaxPooling2D(name=f"pool{index}")(x)
        if index == 2:
            x = tf.keras.layers.Dropout(0.20, name="spatial_dropout")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(0.40, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="fake_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="visionguard_baseline_cnn")
    compile_model(model, learning_rate=1e-3)
    return model


def build_transfer_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5, offset=-1, name="mobilenetv2_preprocessing"
    )(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(0.30, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="fake_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="visionguard_mobilenetv2")
    compile_model(model, learning_rate=1e-3)
    return model, backbone


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
        ],
    )


def trainable_weight_layers(layers: list[tf.keras.layers.Layer]) -> list[str]:
    return [layer.name for layer in layers if layer.trainable and layer.count_params() > 0]


def make_callbacks(name: str, patience: int = 4) -> list[tf.keras.callbacks.Callback]:
    checkpoint = OUTPUT_DIR / f"{name}_best.keras"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def record_history(name: str, history: tf.keras.callbacks.History) -> dict[str, Any]:
    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    with (OUTPUT_DIR / f"{name}_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *history_dict])
        for epoch in range(len(history.epoch)):
            writer.writerow(
                [epoch + 1, *(history_dict[key][epoch] for key in history_dict)]
            )
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    return {"best_epoch": best_epoch, "epochs_run": len(history.epoch), **history_dict}


def save_model_summary(model: tf.keras.Model, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        model.summary(print_fn=lambda line: file.write(line + "\n"))


def save_layer_table(model: tf.keras.Model) -> None:
    with (OUTPUT_DIR / "baseline_layer_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(("layer", "output_shape", "parameters"))
        for layer in model.layers:
            output_shape = getattr(layer, "output", None)
            shape = str(tuple(output_shape.shape)) if output_shape is not None else "n/a"
            writer.writerow((layer.name, shape, layer.count_params()))


def train_model(
    model: tf.keras.Model,
    name: str,
    train_data: tf.data.Dataset,
    validation_data: tf.data.Dataset,
    epochs: int,
    patience: int,
) -> dict[str, Any]:
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=make_callbacks(name, patience),
        shuffle=False,
        verbose=2,
    )
    best_path = OUTPUT_DIR / f"{name}_best.keras"
    best_model = tf.keras.models.load_model(best_path)
    history_result = record_history(name, history)
    save_training_curves(name, history_result)
    return {"model": best_model, "history": history_result}


def save_training_curves(name: str, history: dict[str, Any]) -> None:
    epochs = range(1, history["epochs_run"] + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].axvline(history["best_epoch"], color="black", linestyle="--", label="Best epoch")
    axes[0].set_title(f"{name}: loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, history["accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].axvline(history["best_epoch"], color="black", linestyle="--", label="Best epoch")
    axes[1].set_title(f"{name}: accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{name}_curves.png", dpi=150)
    plt.close(figure)


def infer_once(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for images, batch_labels in dataset:
        batch_scores = model(images, training=False).numpy().reshape(-1)
        labels.append(batch_labels.numpy().reshape(-1).astype(np.int32))
        probabilities.append(batch_scores)
    return np.concatenate(labels), np.concatenate(probabilities)


def choose_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, dict[str, float | int]]:
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    best: tuple[float, float, int, int] | None = None
    for threshold in candidates:
        predictions = probabilities >= threshold
        false_positive = int(np.sum((labels == 0) & predictions))
        false_negative = int(np.sum((labels == 1) & ~predictions))
        cost = 2 * false_positive + false_negative
        candidate = (float(cost), -float(threshold), false_positive, false_negative)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("Could not select a validation decision threshold.")
    threshold = -best[1]
    return threshold, {
        "weighted_error_cost": int(best[0]),
        "false_positive": best[2],
        "false_negative": best[3],
    }


def calculate_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int32)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "weighted_error_cost": int(2 * matrix[0, 1] + matrix[1, 0]),
    }


def save_evaluation_figures(
    name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    metrics: dict[str, Any],
) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["Predicted real", "Predicted AI"])
    axis.set_yticks([0, 1], labels=["Actual real", "Actual AI"])
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center")
    axis.set_title(f"{name} confusion matrix")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(figure)

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(false_positive_rate, true_positive_rate, label=f"AUC={metrics['roc_auc']:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(f"{name} ROC curve")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{name}_roc.png", dpi=150)
    plt.close(figure)


def save_misclassified_grid(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> None:
    paths, _ = read_split("test")
    predictions = (probabilities >= threshold).astype(np.int32)
    error_indices = np.flatnonzero(predictions != labels)[:10]
    with (OUTPUT_DIR / "misclassified_test.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(("path", "actual", "predicted", "ai_probability", "hypothesis"))
        for index in error_indices:
            writer.writerow(
                (
                    Path(paths[index]).relative_to(ROOT).as_posix(),
                    CLASS_NAMES[int(labels[index])],
                    CLASS_NAMES[int(predictions[index])],
                    f"{probabilities[index]:.6f}",
                    "Inspect blur, resampling, watermark/text, style, and fine texture.",
                )
            )

    if len(error_indices) == 0:
        return
    columns = 5
    rows = int(np.ceil(len(error_indices) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3.8 * rows))
    axes = np.asarray(axes).reshape(-1)
    for axis, index in zip(axes, error_indices):
        with Image.open(paths[index]) as image_file:
            image = ImageOps.exif_transpose(image_file).convert("RGB")
            image.thumbnail((360, 260))
            axis.imshow(image)
        axis.set_title(
            f"True: {CLASS_NAMES[int(labels[index])]}\n"
            f"Pred: {CLASS_NAMES[int(predictions[index])]} "
            f"(pAI={probabilities[index]:.2f})"
        )
        axis.axis("off")
    for axis in axes[len(error_indices) :]:
        axis.axis("off")
    figure.suptitle("Misclassified test images (selected model)")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "misclassified_test_grid.png", dpi=150)
    plt.close(figure)


def make_gradcam_figure(
    model: tf.keras.Model,
    backbone: tf.keras.Model | None,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    split_name: str = "test",
) -> None:
    paths, _ = read_split(split_name)
    predictions = (probabilities >= threshold).astype(np.int32)
    correct = np.flatnonzero(predictions == labels)[:4]
    wrong = np.flatnonzero(predictions != labels)[:4]
    selected = list(correct) + list(wrong)
    if not selected:
        return
    grad_model = (
        None
        if backbone is not None
        else tf.keras.Model(
            model.inputs, [model.get_layer("conv3").output, model.output]
        )
    )
    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    for position, axis in enumerate(axes.flat):
        if position >= len(selected):
            axis.axis("off")
            continue
        index = selected[position]
        with Image.open(paths[index]) as image_file:
            source = ImageOps.exif_transpose(image_file).convert("RGB")
            resized = source.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            image = np.asarray(resized, dtype=np.float32)
        tensor = tf.convert_to_tensor(image[None, ...])
        with tf.GradientTape() as tape:
            if backbone is not None:
                feature_maps = backbone(
                    model.get_layer("mobilenetv2_preprocessing")(tensor),
                    training=False,
                )
                x = model.get_layer("global_average_pool")(feature_maps)
                x = model.get_layer("dense_features")(x)
                x = model.get_layer("head_dropout")(x, training=False)
                score = model.get_layer("fake_probability")(x)
            else:
                if grad_model is None:
                    raise RuntimeError("Grad-CAM graph was not initialized.")
                feature_maps, score = grad_model(tensor, training=False)
            target = score[:, 0]
        gradients = tape.gradient(target, feature_maps)
        weights = tf.reduce_mean(gradients, axis=(1, 2))
        heatmap = tf.reduce_sum(feature_maps * weights[:, None, None, :], axis=-1)[0]
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
        heatmap = tf.image.resize(
            heatmap[..., tf.newaxis], IMAGE_SIZE, method="bilinear"
        ).numpy()[..., 0]
        axis.imshow(resized)
        axis.imshow(heatmap, cmap="jet", alpha=0.40)
        kind = "Correct" if predictions[index] == labels[index] else "Wrong"
        axis.set_title(f"{kind}: true {labels[index]}, pred {predictions[index]}")
        axis.axis("off")
    figure.suptitle("Grad-CAM for correct (top row) and wrong (bottom row) cases")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "gradcam_examples.png", dpi=150)
    plt.close(figure)


def write_results_json(results: dict[str, Any]) -> None:
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


def run_training(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_info = create_splits(force=args.make_splits)
    if args.prepare_only:
        save_augmented_batch(make_dataset("train", training=True))
        print(json.dumps(dataset_info, indent=2))
        return

    set_seeds()
    train_data = make_dataset("train", training=True)
    validation_data = make_dataset("validation")
    test_data = make_dataset("test")
    save_augmented_batch(train_data)
    results: dict[str, Any] = {
        "dataset": dataset_info,
        "training": {},
        "evaluation": {},
    }

    baseline = build_baseline()
    save_model_summary(baseline, OUTPUT_DIR / "baseline_model_summary.txt")
    save_layer_table(baseline)
    baseline_result = train_model(
        baseline,
        "baseline",
        train_data,
        validation_data,
        args.epochs,
        args.patience,
    )
    results["training"]["baseline"] = baseline_result["history"]
    results["training"]["baseline"]["trainable_layers"] = trainable_weight_layers(
        baseline_result["model"].layers
    )

    transfer_model, backbone = build_transfer_model()
    save_model_summary(transfer_model, OUTPUT_DIR / "transfer_model_summary.txt")
    frozen_trainable = {
        "backbone": trainable_weight_layers(backbone.layers),
        "head": trainable_weight_layers(
            [
                layer
                for layer in transfer_model.layers
                if layer.name != backbone.name
            ]
        ),
    }
    frozen_result = train_model(
        transfer_model,
        "transfer_frozen",
        train_data,
        validation_data,
        args.epochs,
        args.patience,
    )
    fine_model = frozen_result["model"]
    backbone = fine_model.get_layer(backbone.name)
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = layer.name.startswith("block_16_") and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
    fine_tune_layers = trainable_weight_layers(backbone.layers)
    if not fine_tune_layers:
        raise RuntimeError("MobileNetV2 block_16 layers were not found for fine-tuning.")
    compile_model(fine_model, learning_rate=1e-5)
    finetune_result = train_model(
        fine_model,
        "transfer_finetune",
        train_data,
        validation_data,
        args.epochs,
        args.patience,
    )
    results["training"]["transfer_frozen"] = frozen_result["history"]
    results["training"]["transfer_finetune"] = finetune_result["history"]
    results["training"]["transfer_frozen"]["trainable_layers"] = (
        frozen_trainable["head"]
    )
    results["training"]["transfer_finetune"]["trainable_layers"] = (
        fine_tune_layers
        + trainable_weight_layers(
            [
                layer
                for layer in finetune_result["model"].layers
                if layer.name != backbone.name
            ]
        )
    )
    (OUTPUT_DIR / "trainable_layers.json").write_text(
        json.dumps(
            {
                "transfer_frozen": frozen_trainable,
                "transfer_finetune": {
                    "backbone": fine_tune_layers,
                    "head": results["training"]["transfer_finetune"][
                        "trainable_layers"
                    ][len(fine_tune_layers) :],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    validation_labels, validation_scores = infer_once(
        baseline_result["model"], validation_data
    )
    baseline_threshold, baseline_validation_cost = choose_threshold(
        validation_labels, validation_scores
    )
    final_transfer = finetune_result["model"]
    transfer_labels, transfer_scores = infer_once(final_transfer, validation_data)
    transfer_threshold, transfer_validation_cost = choose_threshold(
        transfer_labels, transfer_scores
    )

    selected_model_name = min(
        (
            ("baseline", baseline_validation_cost, baseline_threshold),
            ("transfer_finetune", transfer_validation_cost, transfer_threshold),
        ),
        key=lambda item: (item[1]["weighted_error_cost"], item[0]),
    )[0]
    results["selected_model"] = selected_model_name
    results["evaluation"]["threshold_selection_cost"] = {
        "baseline": baseline_validation_cost,
        "transfer_finetune": transfer_validation_cost,
    }

    models = {
        "baseline": (baseline_result["model"], baseline_threshold),
        "transfer_finetune": (final_transfer, transfer_threshold),
    }
    test_predictions: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for name, (model, threshold) in models.items():
        test_labels, test_scores = infer_once(model, test_data)
        metrics = calculate_metrics(test_labels, test_scores, threshold)
        results["evaluation"][name] = metrics
        save_evaluation_figures(name, test_labels, test_scores, threshold, metrics)
        test_predictions[name] = (test_labels, test_scores, threshold)

    selected_labels, selected_scores, selected_threshold = test_predictions[
        selected_model_name
    ]
    save_misclassified_grid(selected_labels, selected_scores, selected_threshold)
    if args.gradcam:
        selected_backbone = (
            final_transfer.get_layer("mobilenetv2_1.00_160")
            if selected_model_name == "transfer_finetune"
            else None
        )
        selected_model = models[selected_model_name][0]
        make_gradcam_figure(
            selected_model,
            selected_backbone,
            selected_labels,
            selected_scores,
            selected_threshold,
        )
    write_results_json(results)
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--make-splits", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--gradcam", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive integers.")
    run_training(args)


if __name__ == "__main__":
    main()
