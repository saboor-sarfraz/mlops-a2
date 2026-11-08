"""Append Task 8 local Flask integration evidence to the editable report."""

from __future__ import annotations

from docx import Document

import build_task5_report

ROOT = build_task5_report.ROOT
SOURCE_REPORT = ROOT / "23L-0522_VisionGuard_Tasks1-7.docx"
OUTPUT_REPORT = ROOT / "23L-0522_VisionGuard_Tasks1-8_Final.docx"
add_code_block = build_task5_report.add_code_block
add_screenshot_evidence = build_task5_report.add_screenshot_evidence
add_table = build_task5_report.add_table


def main() -> None:
    if not SOURCE_REPORT.is_file():
        raise FileNotFoundError(f"Existing Tasks 1-7 report not found: {SOURCE_REPORT}")
    document = Document(SOURCE_REPORT)
    if any(paragraph.text.strip() == "8. Flask and Hosted Model API Integration" for paragraph in document.paragraphs):
        raise RuntimeError(f"Task 8 already exists in {SOURCE_REPORT.name}.")

    document.add_page_break()
    document.add_heading("8. Flask and Hosted Model API Integration", level=1)
    document.add_paragraph(
        "The existing Flask frontend runs locally on the development machine. It forwards "
        "each uploaded image to the already-deployed, authenticated Cloud Run model API over "
        "HTTPS, renders the API's prediction in the browser, and writes prediction metadata "
        "to the local SQLite history database. Flask itself was not deployed. The API key is "
        "read from the git-ignored `.env` file and is never embedded in the application page "
        "or stored in the database."
    )

    document.add_heading("8.1 Request flow and local configuration", level=2)
    add_table(
        document,
        ["Stage", "Behavior"],
        [
            ["Browser upload", "The user submits a JPG/JPEG, PNG, BMP, or GIF image to the local Flask `/predict` route."],
            ["Flask validation", "The app checks the filename, accepted extension, and upload-size limit before forwarding the bytes."],
            ["Remote inference", "Flask sends multipart field `image` to the Cloud Run `/api/v1/predict` endpoint over HTTPS with a bearer key from environment configuration."],
            ["Result display", "The response's label, confidence, AI probability, model name, and threshold are rendered in the local result page."],
            ["History", "The local SQLite database stores prediction metadata only; uploaded image bytes are not retained."],
        ],
    )
    add_code_block(
        document,
        "From PowerShell in S:\\A2:\n"
        ".\\run_flask.ps1\n"
        "Open http://127.0.0.1:5000 in a browser.\n"
        "Upload an image, then use History to review saved predictions.\n"
        "Stop the local server with Ctrl+C.",
    )
    document.add_paragraph(
        "The ignored `.env` supplies `MODEL_API_URL` and `MODEL_API_KEY`; the key can be "
        "retrieved from Google Secret Manager for the local process and must not be copied "
        "into source code, screenshots, or this report. The API URL is "
        "`https://model-api-457009269046.europe-west1.run.app`. The web UI and its SQLite "
        "database remain local, so no Flask deployment or additional cloud service is needed."
    )

    document.add_heading("8.2 Multiple-input test results", level=2)
    document.add_paragraph(
        "Two different archive images were submitted through the running local Flask "
        "frontend. Both requests completed with HTTP 200, displayed a result page, and "
        "created a new history record. The outputs below are the model's actual decisions; "
        "they are not claims of ground-truth correctness."
    )
    add_table(
        document,
        ["Input image", "HTTP", "Prediction", "Confidence", "AI probability", "Threshold"],
        [
            [
                "RealArt: -man-sits-with-a-woman-on-her-phone-at-a-table-while-looking-at-a-computer_l.jpg",
                "200",
                "Real",
                "0.5575",
                "0.4425",
                "0.8570",
            ],
            [
                "AiArtData: -how-important-people-from-the-past-really-looked-11-Pics-5ffd5c0f4d77f__880.jpg",
                "200",
                "Real",
                "0.1657",
                "0.8343",
                "0.8570",
            ],
        ],
    )
    document.add_paragraph(
        "The AiArtData sample was classified as Real even though it came from the AI-art "
        "directory. This is a demonstrated false negative for this example and illustrates "
        "why the prediction must not be presented as certain. The threshold is the model "
        "API's validation-selected value and is displayed with each current result."
    )

    document.add_heading("8.3 History and evidence", level=2)
    document.add_paragraph(
        "The SQLite history page shows timestamps, sanitized filenames, verdicts, confidence, "
        "AI probability, decision threshold, and model identity. The history database "
        "persists locally between Flask restarts. At the time of this test, it contained eight "
        "records; the two newest entries correspond to the inputs in the results table."
    )
    add_screenshot_evidence(
        document,
        "task8_result.png",
        "Local Flask result page",
        "The result was returned through the local Flask frontend; no API key is shown.",
    )
    add_screenshot_evidence(
        document,
        "task8_history.png",
        "Multiple predictions in history",
        "The History page shows the two test inputs and their saved results.",
    )
    document.add_paragraph(
        "The HTTP status and prediction values were verified by submitting both files to "
        "`http://127.0.0.1:5000/predict`; the resulting pages redirected to their individual "
        "result records. The SQLite rows confirmed that the model metadata and threshold "
        "were saved."
    )

    document.save(OUTPUT_REPORT)
    print(f"Task 1-8 editable report saved to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
