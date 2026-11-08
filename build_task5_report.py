"""Build the editable Tasks 1-5 report from the existing Tasks 1-3 DOCX."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parent
SOURCE_REPORT = ROOT / "23L-0522_VisionGuard_Tasks1-3.docx"
OUTPUT_REPORT = ROOT / "23L-0522_VisionGuard_Tasks1-5.docx"


def add_table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Normal Table"
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(9)
    for values in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, values):
            cell.text = value
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(8)


def add_screenshot_placeholder(
    document: Document,
    title: str,
    description: str,
    height: float = 1.25,
) -> None:
    table = document.add_table(rows=1, cols=1)
    table.style = "Normal Table"
    row = table.rows[0]
    row.height = Inches(height)
    row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    cell = row.cells[0]
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(f"[PASTE SCREENSHOT HERE]\n{title}\n{description}")
    run.bold = True
    run.font.size = Pt(9)


def add_screenshot_evidence(
    document: Document,
    filename: str,
    title: str,
    description: str,
    width: float = 5.3,
) -> None:
    image_path = ROOT / "visionguard_outputs" / filename
    if not image_path.is_file():
        add_screenshot_placeholder(document, title, description)
        return
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(image_path), width=Inches(width))
    caption = document.add_paragraph(f"{title}. {description}")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in caption.runs:
        run.italic = True
        run.font.size = Pt(8)


def add_code_block(document: Document, value: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(0.25)
    run = paragraph.add_run(value)
    run.font.name = "Consolas"
    run.font.size = Pt(8)


def main() -> None:
    if not SOURCE_REPORT.is_file():
        raise FileNotFoundError(f"Existing Tasks 1-3 report not found: {SOURCE_REPORT}")
    document = Document(SOURCE_REPORT)
    if any(paragraph.text.strip() == "4. Flask Web Application" for paragraph in document.paragraphs):
        raise RuntimeError(
            f"Task 4 already exists in {SOURCE_REPORT.name}; refusing to append a duplicate."
        )

    document.add_page_break()
    document.add_heading("4. Flask Web Application", level=1)
    document.add_paragraph(
        "VisionGuard is served as a small Flask application. A user selects one image in a "
        "browser, receives a Real or AI-generated verdict and confidence, and can review "
        "prediction metadata in the history page. Every result includes a caveat: this "
        "probabilistic classifier can make mistakes and is not proof of an image's origin."
    )

    document.add_heading("4.1 Inference module", level=2)
    document.add_paragraph(
        "The separate inference.py module loads the .keras champion model once when the Flask "
        "application starts and keeps it in app.extensions for subsequent requests. Loading "
        "model weights on every request repeats expensive disk I/O, model construction, and "
        "TensorFlow initialization, adding latency and avoidable resource use. It also makes "
        "concurrent requests compete to initialize duplicate models."
    )
    document.add_paragraph(
        "For each request, uploaded bytes are decoded as three-channel RGB with TensorFlow's "
        "decode_image (the first GIF frame is used), resized to 160 x 160 with bilinear "
        "interpolation and antialiasing, and cast to float32 in [0,255]. This matches "
        "visionguard.py's training input pipeline; the saved MobileNetV2 model itself performs "
        "the expected [-1,1] rescaling. Keeping decoding, size, interpolation, channels, and "
        "numeric range consistent is necessary because a preprocessing mismatch changes the "
        "distribution the trained weights receive and can make scores unreliable."
    )
    document.add_paragraph(
        "The validation-selected threshold is 0.856983. Scores at or above it are labeled "
        "AI-generated; lower scores are labeled Real. Displayed confidence is the model score "
        "for the returned label (AI probability for AI, one minus AI probability for Real). "
        "This score is not a calibrated guarantee."
    )
    document.add_paragraph("Known test-image inference output (one image from each class):")
    add_code_block(
        document,
        "AI test image (64efd48f24bacac40feae0b4_qFcGNfN66Fp4nrvBHhQp89Kg425rczlCMKCbW2cNCCs.jpeg): "
        "prediction=AI-generated, confidence=0.977487, "
        "AI probability=0.977487, threshold=0.856983\n"
        "Real test image (lemuranimaheader3.jpg): prediction=Real, confidence=0.770438, "
        "AI probability=0.229562, threshold=0.856983",
    )
    add_screenshot_evidence(
        document,
        "task4_console_output.png",
        "Console output for two known images",
        "Run `.\\run_demo.ps1` from S:\\A2 and replace this box with the terminal screenshot.",
    )

    document.add_heading("4.2 Routes, validation, and privacy", level=2)
    add_table(
        document,
        ["Route", "Purpose"],
        [
            ["/", "Image-upload page"],
            ["/predict (POST)", "Validate and predict without saving the uploaded image"],
            ["/result/<id>", "Display a stored prediction result"],
            ["/history", "Browse paginated prediction metadata from SQLite"],
            ["/health", "Return model and database health as JSON"],
        ],
    )
    document.add_paragraph(
        "Only JPG/JPEG, PNG, BMP, and GIF filename extensions are accepted, and TensorFlow "
        "must successfully decode the content. The maximum request size is 8 MiB. Filenames "
        "are normalized with Werkzeug secure_filename. A memory-only request stream prevents "
        "Werkzeug from spooling uploaded files to disk; image bytes are processed in memory "
        "and are not stored. History keeps only the sanitized filename, timestamp, verdict, "
        "confidence, AI probability, and configured model identity."
    )
    document.add_paragraph(
        "Automated Flask tests passed for a valid prediction, upload/history flow, the health "
        "endpoint, unsupported extension, corrupt image bytes, and an over-limit upload."
    )
    for heading, description in (
        ("Wrong file type (HTTP 400)", "Submit a .txt file from the upload form."),
        ("Oversized file (HTTP 413)", "Submit an image larger than the configured 8 MiB limit."),
        ("Undecodable image (HTTP 400)", "Rename non-image bytes to .png and submit them."),
    ):
        add_screenshot_evidence(
            document,
            {
                "Wrong file type (HTTP 400)": "task4_error_wrong_type.png",
                "Oversized file (HTTP 413)": "task4_error_oversized.png",
                "Undecodable image (HTTP 400)": "task4_error_corrupt.png",
            }[heading],
            heading,
            f"Friendly browser error page demonstrating that the app rejects the upload. {description}",
            width=4.8,
        )

    document.add_heading("4.3 HTML and CSS", level=2)
    document.add_paragraph(
        "Jinja templates and a responsive stylesheet provide the upload form, result page, "
        "navigation, and history page. Results display the verdict, label confidence, raw AI "
        "probability, threshold, and an uncertainty caveat. The history page reads only "
        "metadata from the SQLite database; uploaded pictures are not retained."
    )
    for heading, description in (
        ("Upload page", "Capture the initial VisionGuard image-upload form."),
        ("Result page", "Capture a verdict, confidence, AI score, and the caveat."),
        ("Prediction history", "Capture a history row loaded from SQLite."),
    ):
        filename = {
            "Upload page": "task4_upload_page.png",
            "Result page": "task4_result_page.png",
            "Prediction history": "task4_history_page.png",
        }[heading]
        add_screenshot_evidence(
            document,
            filename,
            heading,
            {
                "Upload page": "VisionGuard upload form and model caveat.",
                "Result page": "Real verdict, confidence, AI score, threshold, and caveat.",
                "Prediction history": "Stored prediction metadata displayed from SQLite.",
            }[heading],
        )

    document.add_heading("4.4 Configuration and server", level=2)
    add_table(
        document,
        ["Environment variable", "Example value / purpose"],
        [
            [
                "MODEL_PATH",
                "S:/A2/visionguard_outputs/visionguard_comparison_champion.keras",
            ],
            ["THRESHOLD", "0.8569827079772949 (chosen on validation data)"],
            ["MODEL_NAME", "Human-readable model identity stored with each prediction"],
            [
                "DATABASE_URL",
                "sqlite:///S:/A2/visionguard_outputs/prediction_history.db",
            ],
            ["MAX_UPLOAD_BYTES", "8388608"],
            ["HOST / PORT", "127.0.0.1 / 5000"],
        ],
    )
    document.add_paragraph(
        "The application reads model location, threshold, and database URL from environment "
        "variables (python-dotenv loads the ignored local .env when present). run_flask.ps1 "
        "supplies safe local defaults if a variable is absent. Flask starts with debug=False "
        "and use_reloader=False: debug mode can expose an interactive debugger and sensitive "
        "process details, while the development reloader can initialize the model more than "
        "once. Flask's built-in server is for this assignment/local demonstration; a public "
        "production deployment should use a production WSGI server behind HTTPS."
    )
    add_code_block(
        document,
        "From S:\\A2 in PowerShell:\n"
        ".\\run_flask.ps1\n"
        "Open http://127.0.0.1:5000 in a browser. Keep the terminal open; "
        "press Ctrl+C to stop the server.",
    )
    add_screenshot_evidence(
        document,
        "task4_health.png",
        "Health-check response",
        "JSON response confirms the model is loaded and the database is available.",
        width=5.3,
    )

    document.add_heading("Task 4 reproduction and tests", level=2)
    add_code_block(
        document,
        "S:\\MLOps\\venv\\python.exe -m unittest discover -s tests -v\n"
        "From PowerShell: .\\run_demo.ps1\n"
        "From PowerShell: .\\run_flask.ps1",
    )

    document.add_page_break()
    document.add_heading("5. Database Trail", level=1)
    document.add_paragraph(
        "Each successful inference creates a durable SQLite row before the browser is "
        "redirected to its result page. This audit trail records the decision, UTC time, "
        "sanitized original filename, confidence, raw AI probability, and configured model "
        "identity. Image contents are deliberately not persisted. Failed uploads or "
        "inference requests do not create prediction rows."
    )
    document.add_heading("5.1 Schema and persistence", level=2)
    add_table(
        document,
        ["Field", "Meaning"],
        [
            ["id", "Unique prediction identifier used by the result page"],
            ["created_at", "UTC timestamp captured after successful inference"],
            ["filename", "Sanitized client filename; no image bytes are stored"],
            ["label", "Decision: Real or AI-generated"],
            ["confidence", "Model score for the selected label"],
            ["ai_probability", "Raw probability used with the decision threshold"],
            ["model_name", "MODEL_NAME identity, defaulting to the model filename"],
        ],
    )
    document.add_paragraph(
        "The table is initialized on application startup. Existing Task 4 databases are "
        "migrated in place: if model_name is missing, SQLite ALTER TABLE adds it without "
        "dropping previous predictions. Legacy rows remain queryable and are shown as "
        "\"Unknown (older record)\" because their original model identity was not recorded. "
        "A descending created_at index supports the newest-first history view. Results are "
        "paginated 50 rows at a time, so older records remain accessible instead of being "
        "silently hidden by a fixed recent-record limit."
    )
    add_code_block(
        document,
        "SELECT id, created_at, filename, label, confidence, ai_probability, model_name\n"
        "FROM predictions ORDER BY created_at DESC;",
    )
    document.add_heading("5.2 History menu and evidence", level=2)
    document.add_paragraph(
        "The History navigation link opens the SQLite-backed /history page. Each row shows "
        "UTC time, filename, decision, label confidence, AI probability, and model identity; "
        "the View link opens the stored result. Navigation links allow users to browse all "
        "pages. The upload page continues to retain no image files."
    )
    add_screenshot_placeholder(
        document,
        "Prediction history backed by SQLite",
        "Paste a browser screenshot showing a prediction row with timestamp, verdict, confidence, and model.",
        height=2.2,
    )
    document.add_heading("5.3 Reproduction and tests", level=2)
    add_code_block(
        document,
        "S:\\MLOps\\venv\\python.exe -m unittest discover -s tests -v\n"
        "From PowerShell: .\\run_flask.ps1\n"
        "Open http://127.0.0.1:5000, submit an image, then select History.",
    )
    document.add_paragraph(
        "Automated tests verify successful inference creates a permanent row with the "
        "decision, timestamp, confidence, raw probability, and model name; the result and "
        "history pages display that record; older Task 4 database schemas migrate without "
        "losing existing rows; and unsupported, corrupt, and oversized uploads are rejected."
    )

    document.save(OUTPUT_REPORT)
    print(f"Task 1-5 editable report saved to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
