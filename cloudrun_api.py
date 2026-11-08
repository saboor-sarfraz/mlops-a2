"""Authenticated image prediction API for deployment to Cloud Run."""

from __future__ import annotations

import hmac
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from inference import (
    ROOT,
    InvalidImageError,
    environment_model_config,
    load_model,
    predict,
)

DEFAULT_MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif"}


def create_app(test_config: dict | None = None) -> Flask:
    load_dotenv(ROOT / ".env")
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(
        os.environ.get("MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES))
    )
    if app.config["MAX_CONTENT_LENGTH"] < 1:
        raise RuntimeError("MAX_UPLOAD_BYTES must be a positive integer.")

    api_key = os.environ.get("MODEL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("MODEL_API_KEY must be configured from a secret.")

    model_path_value = os.environ.get("MODEL_PATH", "").strip()
    threshold_value = os.environ.get("THRESHOLD", "").strip()
    if model_path_value and threshold_value:
        model_path = Path(model_path_value).expanduser()
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        try:
            threshold = float(threshold_value)
        except InvalidImageError as error:
            raise RuntimeError("THRESHOLD must be a number between 0 and 1.") from error
        if not 0 <= threshold <= 1:
            raise RuntimeError("THRESHOLD must be between 0 and 1.")
    else:
        model_path, threshold = environment_model_config()

    app.config.update(
        MODEL_API_KEY=api_key,
        VISIONGUARD_THRESHOLD=threshold,
        VISIONGUARD_MODEL_NAME=(
            os.environ.get("MODEL_NAME", "").strip() or model_path.stem
        ),
    )
    if test_config:
        app.config.update(test_config)
    model = app.config.pop("CLOUDRUN_MODEL", None)
    if model is None:
        model = load_model(model_path)
    app.extensions["visionguard_model"] = model
    app.logger.info(
        "Cloud Run prediction service loaded model %s at threshold %.6f",
        app.config["VISIONGUARD_MODEL_NAME"],
        threshold,
    )

    @app.get("/health")
    def health():
        return jsonify(status="healthy", model_loaded=True)

    @app.post("/api/v1/predict")
    def predict_image():
        authorization = request.headers.get("Authorization", "")
        bearer = authorization.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(
            bearer.encode("utf-8"), app.config["MODEL_API_KEY"].encode("utf-8")
        ):
            return jsonify(error="Unauthorized"), 401

        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return jsonify(error="Provide an image file in the 'image' field."), 400
        filename = secure_filename(uploaded.filename)
        if not filename:
            return jsonify(error="The image filename is invalid."), 400
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if extension not in ALLOWED_EXTENSIONS:
            return jsonify(error="Upload a JPG, JPEG, PNG, BMP, or GIF image."), 400
        image_bytes = uploaded.stream.read(app.config["MAX_CONTENT_LENGTH"] + 1)
        if len(image_bytes) > app.config["MAX_CONTENT_LENGTH"]:
            return jsonify(error="Image exceeds the configured request size limit."), 413
        try:
            result = predict(
                app.extensions["visionguard_model"],
                image_bytes,
                app.config["VISIONGUARD_THRESHOLD"],
            )
        except ValueError as error:
            app.logger.info("Rejected invalid image: %s", error)
            return jsonify(error="The uploaded file is not a valid supported image."), 400

        return jsonify(
            prediction=result["label"],
            label=result["label"],
            confidence=result["confidence"],
            ai_probability=result["ai_probability"],
            threshold=app.config["VISIONGUARD_THRESHOLD"],
            model_name=app.config["VISIONGUARD_MODEL_NAME"],
            filename=filename,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_error):
        return jsonify(error="Request exceeds the configured size limit."), 413

    @app.errorhandler(500)
    def internal_error(error):
        original = getattr(error, "original_exception", None)
        if original is not None:
            app.logger.error(
                "Prediction API request failed.",
                exc_info=(type(original), original, original.__traceback__),
            )
        return jsonify(error="Prediction request failed."), 500

    return app
