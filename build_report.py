"""Build an editable Word report from VisionGuard training artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ROW_HEIGHT_RULE, WD_CELL_VERTICAL_ALIGNMENT
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "visionguard_outputs"
REPORT_PATH = ROOT / "VisionGuard_Assignment_Report.docx"


def load_json(filename: str) -> dict[str, Any]:
    path = OUTPUT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Run the training/tuning scripts first; missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(filename: str) -> list[dict[str, str]]:
    path = OUTPUT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required report data is missing: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def set_document_style(document: Document) -> None:
    normal = document.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10)
    for style_name in ("Heading 1", "Heading 2", "Heading 3"):
        document.styles[style_name].font.name = "Aptos Display"


def add_title(document: Document, title: str, subtitle: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(title)
    run.bold = True
    run.font.size = Pt(26)
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run(subtitle).italic = True


def add_table(
    document: Document,
    headers: list[str],
    rows: list[list[str]],
    font_size: int = 9,
) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(font_size)
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(font_size)


def add_picture(document: Document, filename: str, caption: str, width: float = 6.2) -> None:
    path = OUTPUT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required figure is missing: {path}")
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Inches(width))
    caption_paragraph = document.add_paragraph(caption)
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption_paragraph.runs:
        caption_paragraph.runs[0].italic = True
        caption_paragraph.runs[0].font.size = Pt(9)


def add_bullet(document: Document, text: str) -> None:
    document.add_paragraph(text, style="List Bullet")


def add_screenshot_placeholder(document: Document, title: str, instructions: str) -> None:
    table = document.add_table(rows=1, cols=1)
    table.style = "Light Shading Accent 1"
    row = table.rows[0]
    row.height = Inches(1.7)
    row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    cell = row.cells[0]
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(f"[PASTE SCREENSHOT HERE]\n{title}\n{instructions}")
    run.bold = True
    run.font.size = Pt(10)


def add_baseline_summary_figure(summary_path: Path, figure_path: Path) -> None:
    text = summary_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    height = max(4.0, min(11.0, 0.19 * len(lines)))
    figure, axis = plt.subplots(figsize=(11, height))
    axis.axis("off")
    axis.text(
        0.01,
        0.99,
        "\n".join(lines),
        va="top",
        family="monospace",
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def add_tuner_table(document: Document, filename: str, columns: list[str]) -> list[dict[str, str]]:
    rows = load_csv(filename)
    visible_rows = [
        [row.get(column, "") for column in columns]
        for row in rows[:8]
    ]
    add_table(document, columns, visible_rows, font_size=8)
    return rows


def summarize_trial_effect(rows: list[dict[str, str]]) -> tuple[str, str]:
    parameters = [
        key
        for key in ("learning_rate", "dropout", "unfrozen_layers")
        if any(key in row for row in rows)
    ]
    effects: list[tuple[float, str, str]] = []
    for parameter in parameters:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            score = row.get("objective")
            value = row.get(parameter)
            if score not in (None, "") and value not in (None, ""):
                grouped.setdefault(value, []).append(float(score))
        means = [sum(values) / len(values) for values in grouped.values()]
        if len(means) > 1:
            effects.append(
                (max(means) - min(means), parameter, f"{min(means):.4f} to {max(means):.4f}")
            )
    if not effects:
        return "Not estimable", "The sampled trials did not provide repeated parameter levels."
    spread, parameter, score_range = max(effects)
    return parameter, f"largest sampled difference in mean validation loss: {spread:.4f} ({score_range})"


def add_mlflow_task3_section(document: Document) -> None:
    comparison_path = OUTPUT_DIR / "mlflow_comparison_results.json"
    registry_path = OUTPUT_DIR / "mlflow_registry_results.json"
    if not comparison_path.is_file() or not registry_path.is_file():
        return

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    local_uri = f"sqlite:///{(ROOT / 'mlflow_local.db').as_posix()}"
    mlflow.set_tracking_uri(local_uri)
    experiment = mlflow.get_experiment_by_name("VisionGuard-Task-3")
    if experiment is None:
        raise RuntimeError("The local MLflow experiment is missing.")
    runs = mlflow.search_runs([experiment.experiment_id])

    def tagged_runs(task: str) -> list[dict[str, Any]]:
        return [
            row.to_dict()
            for _, row in runs.iterrows()
            if row.get("tags.task") == task
        ]

    def add_run_details_table(task: str, headers: list[str], columns: list[str]) -> None:
        selected_runs = tagged_runs(task)
        add_table(
            document,
            headers,
            [
                [
                    str(run.get(column, ""))[:80]
                    for column in columns
                ]
                for run in selected_runs
            ],
            font_size=7,
        )

    document.add_page_break()
    document.add_heading("3. MLflow tracking and model registry", level=1)
    document.add_paragraph(
        "MLflow was configured with a local SQLite tracking database at mlflow_local.db "
        "and its local artifact store. Runs were verified through MLflow's tracking client "
        "and the local interface at http://127.0.0.1:5050."
    )
    document.add_heading("3.1 Local toy runs and terminology", level=2)
    for definition in (
        "Experiment: a named group of related training runs.",
        "Run: one recorded execution, with its own parameters, metrics, tags, and artifacts.",
        "Parameter: an input setting such as learning rate or batch size.",
        "Metric: a measured number such as validation loss at a particular epoch.",
        "Artifact: a file produced or saved by a run, such as a chart or model.",
        "Tag: searchable descriptive metadata, for example logging method or data version.",
    ):
        add_bullet(document, definition)
    toy_runs = tagged_runs("3.1-toy")
    toy_rows = [
        [
            str(run.get("tags.mlflow.runName", "")),
            str(run.get("params.learning_rate", "")),
            f"{float(run.get('metrics.test_accuracy', 0)):.4f}",
            f"{float(run.get('metrics.test_loss', 0)):.4f}",
        ]
        for run in toy_runs
    ]
    add_table(document, ["Run", "Learning rate", "Test accuracy", "Test loss"], toy_rows)
    for index in (1, 2):
        if (OUTPUT_DIR / f"mlflow_toy_{index}_curves.png").is_file():
            add_picture(
                document,
                f"mlflow_toy_{index}_curves.png",
                f"Figure 11.{index}. Fashion-MNIST training and validation metrics for toy run {index}.",
                width=5.8,
            )
    if (OUTPUT_DIR / "mlflow_toy_comparison_ui.png").is_file():
        add_picture(
            document,
            "mlflow_toy_comparison_ui.png",
            "Figure 12. Local MLflow comparison view configured for the two Fashion-MNIST runs.",
            width=6.2,
        )
    if (OUTPUT_DIR / "mlflow_runs_ui.png").is_file():
        add_picture(
            document,
            "mlflow_runs_ui.png",
            "Figure 13. Local MLflow run list with six comparison runs, both VisionGuard methods, and the toy runs.",
            width=6.2,
        )

    document.add_heading("3.2 DagsHub configuration", level=2)
    document.add_paragraph(
        "The DagsHub project tracking URI supplied for this project is "
        "https://dagshub.com/saboor-sarfraz/mlops-a2.mlflow. Credentials are configured "
        "locally in the ignored .env file. The two-run Fashion-MNIST toy experiment completed "
        "successfully on DagsHub. The command output reported test accuracy 0.8624 at learning "
        "rate 0.001 and 0.8198 at learning rate 0.0001. The DagsHub MLflow run links are:"
    )
    add_table(
        document,
        ["Learning rate", "Test accuracy", "DagsHub run"],
        [
            [
                "0.001",
                "0.8624",
                "ae3749b59a3e4d889ad9b14c250698cf",
            ],
            [
                "0.0001",
                "0.8198",
                "eddc8a1a27aa41968fcec8b4be052e7f",
            ],
        ],
        font_size=8,
    )
    for run_id in (
        "ae3749b59a3e4d889ad9b14c250698cf",
        "eddc8a1a27aa41968fcec8b4be052e7f",
    ):
        paragraph = document.add_paragraph()
        paragraph.add_run(
            f"https://dagshub.com/saboor-sarfraz/mlops-a2.mlflow/"
            f"#/experiments/0/runs/{run_id}"
        )
    document.add_paragraph(
        "The training completed in three epochs for each run. TensorFlow oneDNN and dataset "
        "NodeDef messages in the console were informational compatibility notices; both "
        "training runs finished and returned their DagsHub run links. The DagsHub MLflow UI is "
        "hosted remotely; you do not need to start a local MLflow server to view these runs. "
        "Open the project and run links above for screenshots. Never commit .env or include "
        "the access token in screenshots."
    )
    add_screenshot_placeholder(
        document,
        "DagsHub project page",
        "Capture the repository/project page showing saboor-sarfraz/mlops-a2.",
    )
    add_screenshot_placeholder(
        document,
        "DagsHub toy experiment run list",
        "Capture both Fashion-MNIST runs and their names in the hosted MLflow interface.",
    )
    add_screenshot_placeholder(
        document,
        "DagsHub run details: learning rate 0.001",
        "Capture parameters and test metrics for run ae3749b59a3e4d889ad9b14c250698cf.",
    )
    add_screenshot_placeholder(
        document,
        "DagsHub run details: learning rate 0.0001",
        "Capture parameters and test metrics for run eddc8a1a27aa41968fcec8b4be052e7f.",
    )
    add_screenshot_placeholder(
        document,
        "Proof that .env is git-ignored",
        "In PowerShell run: git check-ignore -v .env. Capture the rule/output, not the .env file.",
    )
    document.add_paragraph(
        "Optional: to start the LOCAL MLflow UI and inspect local runs, run this from "
        "S:\\A2 in PowerShell, then open http://127.0.0.1:5050 in a browser. Keep that terminal "
        "open while using the UI; press Ctrl+C to stop the server. This is separate from the "
        "hosted DagsHub UI."
    )
    local_server = document.add_paragraph()
    run = local_server.add_run(
        r"S:\MLOps\venv\python.exe -m mlflow ui --backend-store-uri "
        r"sqlite:///S:/A2/mlflow_local.db --host 127.0.0.1 --port 5050"
    )
    run.font.name = "Consolas"
    run.font.size = Pt(8)

    document.add_heading("3.3 Manual and autolog VisionGuard runs", level=2)
    add_run_details_table(
        "3.3-visionguard",
        ["Logging method", "Run", "Trainable LR", "Validation loss", "Test accuracy", "Test F1"],
        [
            "tags.logging_method",
            "tags.mlflow.runName",
            "params.learning_rate",
            "metrics.best_validation_loss",
            "metrics.test_accuracy",
            "metrics.test_f1",
        ],
    )
    document.add_paragraph(
        "The manual callback logs every training and validation metric at each epoch; explicit "
        "code logs the final validation threshold, test metrics, tags, model, plot, confusion "
        "matrix, ROC curve, split CSVs, and package list. TensorFlow autolog captures the Keras "
        "fit history and a training model automatically, while explicit logging is still used "
        "for final test metrics, tags, the full image-input model, and assignment artifacts. "
        "Both runs train the same frozen-MobileNetV2 feature-extractor/head setup."
    )
    document.add_paragraph(
        "To run the DagsHub VisionGuard training and see terminal progress, execute "
        "S:\\MLOps\\venv\\python.exe mlflow_tasks.py --task visionguard-dagshub --epochs 2 "
        "from S:\\A2. The script prints feature-extraction progress by split, Keras training "
        "loss/accuracy after every epoch for each logging method, evaluation/upload stages, "
        "and the completed DagsHub run IDs. This run trains the classification head while "
        "the pretrained MobileNetV2 feature-extractor remains frozen."
    )

    document.add_heading("3.4 Logged artifacts", level=2)
    add_bullet(
        document,
        "The VisionGuard run's assignment_artifacts folder contains confusion matrix, ROC curve, three split CSVs, and requirements_frozen.txt.",
    )
    add_bullet(
        document,
        "The models folder contains the full image-input Keras model; plots and training history are logged separately.",
    )

    document.add_heading("3.5 Six-run experiment comparison", level=2)
    table_rows = []
    for run in sorted(
        comparison["runs"], key=lambda item: item["learning_rate"]
    ):
        table_rows.append(
            [
                f"{run['learning_rate']:g}",
                str(run["validation_weighted_error_cost"]),
                f"{run['accuracy']:.3f}",
                f"{run['f1']:.3f}",
                f"{run['roc_auc']:.3f}",
                f"{run['best_validation_loss']:.3f}",
            ]
        )
    add_table(
        document,
        [
            "Learning rate",
            "Val cost",
            "Val accuracy",
            "Val F1",
            "Val ROC-AUC",
            "Val loss",
        ],
        table_rows,
        font_size=8,
    )
    document.add_paragraph(
        f"Only learning rate varied; backbone, image size, augmentation, batch size, seed, "
        f"and three-epoch budget were held fixed. The validation-cost winner was "
        f"{comparison['selected_run_name']} (weighted cost "
        f"{comparison['selected_validation_weighted_error_cost']}). On its final test pass it "
        f"achieved accuracy {comparison['selected_test_metrics']['accuracy']:.3f}, F1 "
        f"{comparison['selected_test_metrics']['f1']:.3f}, and ROC-AUC "
        f"{comparison['selected_test_metrics']['roc_auc']:.3f}. The cost penalizes a real image "
        "incorrectly flagged as AI twice as much as an AI image missed."
    )
    chart = OUTPUT_DIR / "mlflow_comparison_chart.png"
    if chart.is_file():
        add_picture(
            document,
            chart.name,
            "Figure 14. Six-run validation comparison from the local MLflow experiment.",
            width=6.2,
        )

    document.add_heading("3.6 Model registry and alias prediction", level=2)
    add_table(
        document,
        ["Registered model", "Version", "Source run", "Alias"],
        [
            [
                registry["registered_model"],
                version["version"],
                version["run_id"],
                "champion" if version["version"] == registry["champion_alias_version"] else "",
            ]
            for version in registry["versions"]
        ],
    )
    document.add_paragraph(
        "The separate mlflow_predict.py script loaded models:/VisionGuard@champion and "
        "successfully classified an archive image. The test image produced AI probability "
        "0.175510 and was labeled Real at the validation-selected threshold "
        f"{registry['threshold']:.6f}. The model signature expects a float RGB image tensor "
        "with shape 160 x 160 x 3."
    )
    if (OUTPUT_DIR / "mlflow_registry_ui.png").is_file():
        add_picture(
            document,
            "mlflow_registry_ui.png",
            "Figure 15. Local MLflow registry showing the VisionGuard champion alias.",
            width=6.2,
        )


def main() -> None:
    results = load_json("results.json")
    dataset = results["dataset"]
    evaluation = results["evaluation"]
    toy = load_json("toy_tuner_summary.json")
    applied_tuner = load_json("visionguard_tuner_summary.json")

    document = Document()
    set_document_style(document)
    add_baseline_summary_figure(
        OUTPUT_DIR / "baseline_model_summary.txt",
        OUTPUT_DIR / "baseline_model_summary.png",
    )
    add_title(
        document,
        "VisionGuard",
        "AI-generated versus real-art image classification | Editable assignment report",
    )
    document.add_paragraph(
        "Replace the course/student details below as needed. The report tables and explanatory "
        "text are editable in Word; charts and image grids are inserted as figures."
    )
    add_table(
        document,
        ["Course / assignment", "Student", "Date"],
        [["Machine Learning / Assignment 2", "[Edit name and ID]", "[Edit date]"]],
    )

    document.add_heading("Executive summary", level=1)
    selected = results["selected_model"]
    document.add_paragraph(
        f"The project compares a small convolutional neural network with a pretrained "
        f"MobileNetV2. The model selected using validation data was {selected}. The held-out "
        "test set was reserved for one final prediction pass per model; its results are shown "
        "in the evaluation section. A predicted AI probability is the sigmoid output, and "
        "label 1 denotes AI-generated artwork."
    )

    document.add_heading("2.1 Input pipeline", level=1)
    counts = dataset["counts"]
    add_table(
        document,
        ["Split", "Images", "Real (label 0)", "AI-generated (label 1)"],
        [
            [
                split.title(),
                str(counts[split]["images"]),
                str(counts[split]["real"]),
                str(counts[split]["ai"]),
            ]
            for split in ("train", "validation", "test")
        ],
    )
    document.add_paragraph(
        f"The archive contains {dataset['discovered_files']} recognized image files. "
        f"{dataset.get('duplicate_files_removed', 0)} byte-identical duplicate was removed "
        "before splitting so identical files cannot leak across subsets. Split lists are saved "
        "as CSV files under visionguard_outputs/splits and are reused on subsequent runs. "
        "The split is stratified, reproducible (seed 42), and approximately 70%/15%/15%."
    )
    for item in (
        "The CSV split list supplies a path and binary label; label 0 is RealArt and label 1 is AiArtData.",
        "tf.io.read_file loads each file and tf.io.decode_image decodes it into three RGB channels; GIFs use their first frame.",
        "tf.image.resize converts every decoded image to 160 x 160 pixels using bilinear interpolation with antialiasing. Images are stretched directly to a square: aspect ratio is not preserved, so circles/faces can be slightly distorted. No crop is used.",
        "Images remain float32 in [0, 255] in the input pipeline. Only training data is shuffled and receives random horizontal flips, rotations, zoom, and contrast variation. Validation and test data are not augmented or shuffled.",
        "The examples are grouped into batches of 32 and prefetched with tf.data.AUTOTUNE to overlap input work with model execution.",
    ):
        add_bullet(document, item)
    add_picture(
        document,
        "augmented_batch.png",
        "Figure 1. Example training batch after random augmentation (labels shown above each image).",
    )

    document.add_heading("2.2 Baseline CNN", level=1)
    document.add_paragraph(
        "The baseline is a compact Keras CNN trained from scratch. Binary cross-entropy is "
        "appropriate for the single sigmoid probability and binary labels. Adam starts at "
        "1e-3 to provide adaptive updates. Accuracy is easy to interpret; precision and recall "
        "expose the two error types; ROC-AUC measures ranking over thresholds."
    )
    for item in (
        "Rescaling maps RGB pixels from [0, 255] to [0, 1], making optimization numerically stable.",
        "Three 3 x 3 convolutions (32, 64, 128 filters) learn increasingly complex local patterns; batch normalization stabilizes activations and ReLU adds non-linearity.",
        "Max pooling reduces spatial resolution and computation. Dropout after the second block and after the dense layer reduce overfitting on this small dataset.",
        "Global average pooling replaces a large flatten layer with a compact feature vector, reducing parameters; a 128-unit dense layer combines learned features.",
        "A final one-unit sigmoid returns P(AI-generated), the only output required for binary classification.",
    ):
        add_bullet(document, item)
    add_picture(
        document,
        "baseline_model_summary.png",
        "Figure 2. Keras baseline model.summary() rendered as an image for quick inspection.",
    )
    layer_rows = load_csv("baseline_layer_table.csv")
    add_table(
        document,
        ["Layer", "Output shape", "Parameters"],
        [[row["layer"], row["output_shape"], row["parameters"]] for row in layer_rows],
        font_size=8,
    )

    document.add_heading("2.3 Transfer learning", level=1)
    document.add_paragraph(
        "MobileNetV2 is loaded from keras.applications with include_top=False and ImageNet "
        "weights. Its expected input transform maps RGB pixels to [-1, 1], implemented as "
        "Rescaling(scale=1/127.5, offset=-1) inside the model. This keeps preprocessing "
        "consistent with the pretrained feature extractor."
    )
    add_bullet(
        document,
        "Stage 1: freeze the complete pretrained backbone and train a new global-average-pooling, 128-unit ReLU, 0.30-dropout, sigmoid head.",
    )
    add_bullet(
        document,
        "Stage 2: unfreeze only MobileNetV2 block_16 (the last inverted-residual block), keep its batch-normalization layers frozen, and continue with Adam at 1e-5.",
    )
    transfer_layers = load_json("trainable_layers.json")
    frozen_backbone = ", ".join(transfer_layers["transfer_frozen"]["backbone"]) or "None"
    frozen_head = ", ".join(transfer_layers["transfer_frozen"]["head"])
    fine_backbone = ", ".join(transfer_layers["transfer_finetune"]["backbone"])
    fine_head = ", ".join(transfer_layers["transfer_finetune"]["head"])
    add_table(
        document,
        ["Training stage", "Trainable backbone layers", "Trainable head layers"],
        [
            ["Frozen backbone", frozen_backbone, frozen_head],
            ["Fine-tuning", fine_backbone, fine_head],
        ],
        font_size=7,
    )
    add_picture(
        document,
        "transfer_frozen_curves.png",
        f"Figure 3. Transfer-learning stage 1; best validation-loss epoch: "
        f"{results['training']['transfer_frozen']['best_epoch']}.",
    )
    add_picture(
        document,
        "transfer_finetune_curves.png",
        f"Figure 4. Transfer-learning fine-tuning stage; best validation-loss epoch: "
        f"{results['training']['transfer_finetune']['best_epoch']}.",
    )

    document.add_heading("2.4 Training discipline", level=1)
    document.add_paragraph(
        f"Python, NumPy, and TensorFlow/Keras seeds are fixed at {dataset['seed']}; TensorFlow "
        "deterministic operations are enabled. All models monitor validation binary cross-entropy. "
        "ModelCheckpoint saves the lowest-validation-loss model, EarlyStopping stops after four "
        "non-improving epochs and restores best weights, and ReduceLROnPlateau reduces the "
        "learning rate by a factor of 0.2 after two stagnant epochs. Best epochs are defined as "
        "the epoch with minimum validation loss. Training ran on the CPU because TensorFlow "
        "reported no usable GPU in the native Windows environment."
    )
    for name, title in (
        ("baseline", "Baseline CNN"),
        ("transfer_frozen", "Transfer stage 1"),
        ("transfer_finetune", "Transfer fine-tuning"),
    ):
        add_picture(
            document,
            f"{name}_curves.png",
            f"{title}: best epoch {results['training'][name]['best_epoch']} of "
            f"{results['training'][name]['epochs_run']} epochs.",
        )

    document.add_heading("2.5 Evaluation", level=1)
    document.add_paragraph(
        "For each model, the decision threshold is chosen using validation predictions only by "
        "minimizing 2 x false positives + false negatives; ties choose the higher threshold. "
        "A false positive here means calling a real image AI-generated. For a tool that may "
        "label an authentic photograph as fake, that accusation is the worse product error, "
        "so it receives twice the cost of allowing an AI image through. Test data is not used "
        "to choose thresholds or select a model. Each model is run once on the test dataset, "
        "and all thresholded metrics, ROC values, and error examples reuse that pass."
    )
    document.add_paragraph(
        "The optional Grad-CAM figure is generated from validation examples in this report. "
        "It uses additional forward/gradient passes only for up to eight selected examples and "
        "does not alter the threshold, model selection, or test metrics."
    )
    metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc", "threshold")
    metric_rows = []
    for model_name, label in (
        ("baseline", "Baseline CNN"),
        ("transfer_finetune", "MobileNetV2 fine-tuned"),
    ):
        model_metrics = evaluation[model_name]
        metric_rows.append(
            [
                label,
                *[f"{model_metrics[metric]:.4f}" for metric in metric_names],
                str(model_metrics["false_positive"]),
                str(model_metrics["false_negative"]),
            ]
        )
    add_table(
        document,
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "Threshold",
            "FP",
            "FN",
        ],
        metric_rows,
        font_size=8,
    )
    document.add_paragraph(
        "In this run the baseline's validation-selected threshold produced zero AI detections "
        "on the test set (F1 = 0). MobileNetV2 was selected using validation cost (51 versus "
        "80 for the baseline), but it still missed 29 AI images and falsely flagged 7 real "
        "images on test. Treat these numbers as evidence from this small dataset, not a "
        "deployment guarantee."
    )
    add_picture(
        document,
        "baseline_confusion_matrix.png",
        "Figure 5. Baseline confusion matrix at its validation-selected threshold.",
        width=4.8,
    )
    add_picture(
        document,
        "baseline_roc.png",
        "Figure 6. Baseline ROC curve; AUC is threshold-independent.",
        width=4.8,
    )
    add_picture(
        document,
        "transfer_finetune_confusion_matrix.png",
        "Figure 7. MobileNetV2 confusion matrix at its validation-selected threshold.",
        width=4.8,
    )
    add_picture(
        document,
        "transfer_finetune_roc.png",
        "Figure 8. MobileNetV2 ROC curve; AUC is threshold-independent.",
        width=4.8,
    )

    document.add_heading("2.6 Error analysis", level=1)
    document.add_paragraph(
        f"The validation-cost-selected model ({selected}) supplies the displayed error examples. "
        "The image-specific explanations below are hypotheses to verify visually, not proven "
        "causes. They can be edited after reviewing the images."
    )
    document.add_paragraph(
        "The reviewed grid contains photo-like AI portraits/animals that were called real, "
        "and a genuine waterfall and historical painting collage called AI. This suggests "
        "substantial style overlap: the classifier may use visual style rather than reliable "
        "generation provenance, so it should not be treated as an authenticity detector."
    )
    error_rows = load_csv("misclassified_test.csv")
    if error_rows:
        add_picture(
            document,
            "misclassified_test_grid.png",
            "Figure 9. Up to 10 test-set misclassifications from the selected model.",
        )
        hypotheses = {
            "robot-monitoring-dairy-farm-1200x800-1.jpg": (
                "The detailed farm scene, plausible lighting, and robot/cow context may "
                "have made this AI image look like a documentary photograph."
            ),
            "midjourney-v5-drafts-Artboard-4-800x400.jpg": (
                "The realistic skin, portrait lighting, and side-by-side photo layout may "
                "have obscured generation cues."
            ),
            "ures-created-on-midjourney-midjourney-curated-feature-stirworld-220930050044.jpg": (
                "Natural-looking bark, architecture, and sunset lighting may have made this "
                "AI concept image resemble a real location photograph."
            ),
            "nature-scenery.jpg": (
                "The smooth long-exposure waterfall and saturated foliage may resemble "
                "synthetic landscape rendering."
            ),
            "1000_F_570271411_Tqbdg5z0mL3qV3eaDS0XGrjMMwk6aNTf.jpg": (
                "Detailed fur, shallow depth of field, and a stock-style watermark may make "
                "this AI animal portrait look photographic."
            ),
            "cat-with-ducklings.jpg": (
                "The unusual animal grouping may look compositionally improbable, even though "
                "the image is labeled real."
            ),
            "HD-wallpaper-ai-landscape-2023-digital-art.jpg": (
                "The smooth sunset gradient and silhouette may imitate a conventional "
                "landscape photograph and hide painterly cues."
            ),
            "Earth-Species-project-generative-ai-unders.max-2100x2100.jpg": (
                "Dark exposure, close cropping, and dense fur-like texture may resemble a "
                "real wildlife close-up."
            ),
            "Dottemannen_photo_of_a_blonde_swede_5d222c23-1088-4336-93c6-0fbd355e8150.png": (
                "Photorealistic faces and skin detail in the portrait collage may have hidden "
                "synthetic inconsistencies."
            ),
            "most-famous-paintings-by-jan-van-eyck.jpg": (
                "Painterly texture and the collage of historical portraits may have triggered "
                "style cues associated with generated artwork."
            ),
        }
        table_rows = []
        for index, row in enumerate(error_rows, start=1):
            filename = Path(row["path"]).name
            hypothesis = hypotheses.get(
                filename,
                (
                    "Possible processed appearance, blur, compression, or watermark may "
                    "have raised the AI score."
                    if row["actual"] == "RealArt"
                    else "Convincing photographic texture or lighting may have lowered the "
                    "AI score."
                ),
            )
            table_rows.append(
                [
                    str(index),
                    filename,
                    f"{row['actual']} -> {row['predicted']}",
                    row["ai_probability"],
                    hypothesis,
                ]
            )
        add_table(
            document,
            ["#", "Image", "Actual -> predicted", "P(AI)", "Editable hypothesis"],
            table_rows,
            font_size=7,
        )
    else:
        document.add_paragraph(
            "The selected model made no test errors at the chosen threshold, so there are no "
            "misclassified examples to display."
        )

    document.add_heading("2.7 Tool lab: Keras Tuner", level=1)
    document.add_paragraph(
        f"The toy search uses RandomSearch on Fashion-MNIST with {toy['trials']} trials, tuning "
        "the hidden-unit count and Adam learning rate. It selected "
        f"{toy['best_settings']} and achieved test accuracy {toy['test_accuracy']:.4f}."
    )
    toy_rows = load_csv("toy_tuner_results.csv")
    add_table(
        document,
        ["Trial", "Units", "Learning rate", "Best validation accuracy"],
        [
            [
                row["trial_id"],
                row["units"],
                f"{float(row['learning_rate']):.6g}",
                f"{float(row['objective']):.4f}",
            ]
            for row in toy_rows
        ],
        font_size=8,
    )
    document.add_paragraph(
        f"The applied search uses {applied_tuner['trials']} RandomSearch trials on VisionGuard, "
        "with MobileNetV2, tuning learning rate, dropout, and the number of final backbone "
        "layers opened for fine-tuning (batch-normalization layers stay frozen). Its best "
        "validation-loss setting was "
        f"{applied_tuner['best_settings']} (validation loss "
        f"{applied_tuner['best_validation_loss']:.4f})."
    )
    applied_rows = add_tuner_table(
        document,
        "visionguard_tuner_results.csv",
        ["trial_id", "learning_rate", "dropout", "unfrozen_layers", "objective"],
    )
    most_sensitive, sensitivity = summarize_trial_effect(applied_rows)
    parameter_reasons = {
        "learning_rate": (
            "it directly changes update size, so values that are too large can destabilize "
            "fine-tuning while values that are too small adapt slowly"
        ),
        "dropout": (
            "it changes regularization strength and therefore the balance between fitting the "
            "small training set and retaining enough head capacity"
        ),
        "unfrozen_layers": (
            "it controls how much pretrained representation can adapt to this dataset, trading "
            "domain adaptation against overfitting and extra computation"
        ),
    }
    reason = parameter_reasons.get(
        most_sensitive, "the current random sample does not isolate a parameter effect"
    )
    document.add_paragraph(
        f"Within these trials, {most_sensitive} had the largest observed grouped mean "
        f"validation-loss spread ({sensitivity}). It may matter most because {reason}. "
        "This is an exploratory signal, not a controlled causal comparison: random trials can "
        "confound hyperparameters and training noise."
    )
    document.add_paragraph(
        "Selecting the best result after many trials can overfit the validation set because "
        "each trial indirectly adapts choices to that same validation data. The test set remains "
        "untouched until the final evaluation, and a larger study should use nested validation "
        "or a separate tuning holdout."
    )

    gradcam_path = OUTPUT_DIR / "gradcam_examples.png"
    if gradcam_path.is_file():
        document.add_heading("Bonus: Grad-CAM", level=1)
        document.add_paragraph(
            "Grad-CAM highlights feature-map regions that locally contribute to the AI score. "
            "In these examples attention often falls on central subjects or bright, high-contrast "
            "textures (including portrait clothing/neck areas and landscape terrain), rather "
            "than one consistent generation artifact. This may indicate reliance on content or "
            "style cues; the maps are qualitative and do not establish causal evidence."
        )
        add_picture(
            document,
            "gradcam_examples.png",
            "Figure 10. Up to four correct and four incorrect predictions with Grad-CAM overlays.",
        )

    document.add_heading("Reproduction", level=1)
    document.add_paragraph(
        "From the repository root, activate the supplied conda environment and run:"
    )
    for command in (
        r"S:\MLOps\venv\python.exe visionguard.py",
        r"S:\MLOps\venv\python.exe keras_tuner_lab.py --mode all --trials 8",
        r"S:\MLOps\venv\python.exe generate_gradcam.py",
        r"S:\MLOps\venv\python.exe build_report.py",
    ):
        paragraph = document.add_paragraph()
        run = paragraph.add_run(command)
        run.font.name = "Consolas"
        run.font.size = Pt(9)
    document.add_paragraph("MLflow Task 3 commands:")
    for command in (
        r"S:\MLOps\venv\python.exe -m mlflow ui --backend-store-uri sqlite:///S:/A2/mlflow_local.db --host 127.0.0.1 --port 5050",
        r"S:\MLOps\venv\python.exe mlflow_tasks.py --task toy-local",
        r"S:\MLOps\venv\python.exe mlflow_tasks.py --task visionguard-local --epochs 2",
        r"S:\MLOps\venv\python.exe mlflow_tasks.py --task compare-local --epochs 3",
        r"S:\MLOps\venv\python.exe mlflow_tasks.py --task register-local",
        r"S:\MLOps\venv\python.exe mlflow_tasks.py --task toy-dagshub",
        r"S:\MLOps\venv\python.exe mlflow_tasks.py --task visionguard-dagshub --epochs 2",
        r"S:\MLOps\venv\python.exe mlflow_tasks.py --task compare-dagshub --epochs 3",
        r"S:\MLOps\venv\python.exe mlflow_predict.py archive\AiArtData\robot-monitoring-dairy-farm-1200x800-1.jpg",
    ):
        paragraph = document.add_paragraph()
        run = paragraph.add_run(command)
        run.font.name = "Consolas"
        run.font.size = Pt(8)
    document.add_paragraph(
        "Use --prepare-only to create split lists and the augmented-batch figure without "
        "training. Run generate_gradcam.py after training to produce Grad-CAM from validation "
        "examples without re-running test evaluation. Outputs are stored in visionguard_outputs/."
    )
    add_mlflow_task3_section(document)

    document.save(REPORT_PATH)
    print(f"Editable report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
