"""Append Task 7 Cloud Run instructions and evidence to the editable report."""

from __future__ import annotations

from docx import Document

import build_task5_report
import build_task6_report

ROOT = build_task5_report.ROOT
SOURCE_REPORT = ROOT / "23L-0522_VisionGuard_Tasks1-6.docx"
OUTPUT_REPORT = ROOT / "23L-0522_VisionGuard_Tasks1-7.docx"
add_code_block = build_task5_report.add_code_block
add_screenshot_placeholder = build_task5_report.add_screenshot_placeholder
add_table = build_task5_report.add_table


def main() -> None:
    if not SOURCE_REPORT.is_file():
        build_task6_report.main()
    document = Document(SOURCE_REPORT)
    if any(paragraph.text.strip() == "7. Public Model API on Cloud Run" for paragraph in document.paragraphs):
        raise RuntimeError(f"Task 7 already exists in {SOURCE_REPORT.name}.")

    document.add_page_break()
    document.add_heading("7. Public Model API on Cloud Run", level=1)
    document.add_paragraph(
        "The trained VisionGuard model is served as a separate authenticated JSON API on "
        "Google Cloud Run. The Flask upload/history interface remains on the local machine "
        "and forwards image bytes to the remote model endpoint. A Cloud Run request returns "
        "the prediction and confidence in JSON; the local Flask app stores the response in "
        "its prediction history database."
    )
    document.add_heading("7.1 Model artifact and API contract", level=2)
    add_table(
        document,
        ["Item", "Value"],
        [
            ["Trained model", "visionguard_outputs/visionguard_comparison_champion.keras"],
            ["Model size", "10,284,071 bytes (about 9.8 MiB)"],
            ["Inference input", "multipart/form-data; image file field named image"],
            ["Prediction endpoint", "https://model-api-457009269046.europe-west1.run.app/api/v1/predict"],
            ["Authentication", "Bearer API key read from Secret Manager at runtime; key value is not included in this report"],
            ["Response", "JSON: label, confidence, ai_probability, threshold, model_name, filename, created_at"],
        ],
    )
    add_code_block(
        document,
        "POST https://model-api-457009269046.europe-west1.run.app/api/v1/predict\n"
        "Authorization: Bearer token supplied from Secret Manager (redact in screenshots)\n"
        "Body: form-data; key=image; type=File; choose a jpg/png image\n\n"
        "Expected response fields:\n"
        "{ \"label\": \"Real or AI-generated\", \"confidence\": 0.0,\n"
        "  \"ai_probability\": 0.0, \"threshold\": 0.8569827079772949,\n"
        "  \"model_name\": \"VisionGuard MobileNetV2 champion\",\n"
        "  \"filename\": \"example.png\", \"created_at\": \"UTC timestamp\" }",
    )
    document.add_paragraph(
        "The model file is baked into the dedicated Cloud Run image, so deployment does not "
        "depend on a local disk mount. The API container uses the same pinned CPU TensorFlow "
        "2.21.0 runtime, 160 x 160 RGB preprocessing, and validation-selected threshold as "
        "the trained champion. The `/health` endpoint provides a container health check; "
        "the prediction route checks a bearer key using a constant-time comparison."
    )

    document.add_heading("7.2 Google Cloud setup and deployment", level=2)
    document.add_paragraph(
        "The deployed service is named `model-api` in `europe-west1` under project "
        "`grounded-gizmo-510509-c7`. The model container was built and uploaded to Artifact "
        "Registry, and the service reads its API key from Secret Manager. Before deploying, "
        "install the Google Cloud CLI, sign in with `gcloud auth login`, select a Google "
        "Cloud project with billing enabled, and review current Cloud Run, "
        "Cloud Build, Artifact Registry, Secret Manager, and network pricing. Cloud Run is "
        "configured with zero minimum instances so idle compute can scale to zero; image "
        "storage and other Google Cloud resources may still incur charges."
    )
    add_code_block(
        document,
        "gcloud auth login\n"
        "gcloud config set project grounded-gizmo-510509-c7\n"
        "gcloud run services describe model-api --region europe-west1 "
        "--format='value(status.url)'\n"
        "gcloud secrets versions access latest "
        "--secret=visionguard-model-api-key",
    )
    document.add_paragraph(
        "The API key is read from the Secret Manager secret `visionguard-model-api-key` "
        "at runtime; the key value is never written in this report. Cloud Run exposes the "
        "service URL over HTTPS, and prediction requests use the API's bearer-key check. "
        "The user deployed the service from the Cloud Run Console; the CLI commands above "
        "verify its URL and retrieve the secret for a local test session. "
        "The Google Cloud service URL and successful prediction below were copied from the "
        "user's deployed service and `test_api.py` output."
    )
    add_screenshot_placeholder(
        document,
        "Cloud Build and Cloud Run deployment",
        "Paste the successful Cloud Build image build and Cloud Run service overview/healthy status. Do not show secrets.",
        height=2.0,
    )

    document.add_heading("7.3 API test and local Flask client", level=2)
    document.add_paragraph(
        "The user verified the deployed public API from the S:\\MLOps\\venv environment "
        "with `python test_api.py`. The script obtained the bearer key from Secret Manager "
        "into a process environment variable (without printing its value), sent an image, "
        "and received HTTP 200. This is a real API test; a Postman screenshot is still "
        "required if the assessor specifically requires Postman evidence. Redact the API "
        "key in any screenshot."
    )
    add_code_block(
        document,
        "HTTP status: 200\n"
        "label: Real\n"
        "confidence: 0.1793401837348938\n"
        "ai_probability: 0.8206598162651062\n"
        "threshold: 0.8569827079772949\n"
        "model_name: visionguard_comparison_champion\n"
        "created_at: 2026-11-08T16:57:37+00:00",
    )
    add_screenshot_placeholder(
        document,
        "Successful Postman API prediction",
        "Required for the assignment: paste a Postman screenshot with the public URL, image form-data field, HTTP 200 status and JSON output; redact the bearer token.",
        height=2.2,
    )
    document.add_paragraph(
        "To use the endpoint from the local Flask interface, add MODEL_API_URL and "
        "MODEL_API_KEY to the ignored `.env` (read the key from Secret Manager without "
        "committing or displaying it). Restart the local Flask app "
        "with `.\run_flask.ps1`, upload a test image through the browser, and verify that "
        "the result/history pages name the Cloud Run model. The remote URL is public, but "
        "prediction requests still require the API key. Do not paste the key into the report "
        "or source code."
    )
    add_code_block(
        document,
        ".\\run_flask.ps1\n"
        "Open http://127.0.0.1:5000, upload an image, then open History.\n"
        "Health check: Invoke-RestMethod https://[SERVICE URL]/health",
    )
    add_screenshot_placeholder(
        document,
        "Local Flask using Cloud Run and prediction history",
        "Paste the local Flask result and history row after a request was inferred remotely. Avoid screenshots of .env.",
        height=1.8,
    )

    document.add_heading("7.4 URL, result, and cost cleanup", level=2)
    add_table(
        document,
        ["Deployment evidence", "Record after the actual deployment"],
        [
            ["Cloud Run region", "europe-west1"],
            ["Cloud Run service URL", "https://model-api-457009269046.europe-west1.run.app"],
            ["Prediction API URL", "https://model-api-457009269046.europe-west1.run.app/api/v1/predict"],
            ["API script test", "HTTP 200; Real; confidence 0.1793401837348938"],
            ["Postman evidence", "[Still required: capture the Postman request/response; redact bearer key]"],
            ["External-network test", "[Record the network used if different from the deployment network]"],
        ],
    )
    document.add_paragraph(
        "The URL and prediction result above are the deployed service and user's actual "
        "successful API test. The external-network detail and Postman screenshot remain to "
        "be filled if available. After capturing evidence, remove the service if it is no longer "
        "needed to avoid future compute charges: "
        "`gcloud run services delete model-api --region europe-west1`. The "
        "Artifact Registry image and Secret Manager secret are separate resources and may "
        "continue to incur storage charges; remove them only after saving required evidence."
    )

    document.save(OUTPUT_REPORT)
    print(f"Task 1-7 editable report saved to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
