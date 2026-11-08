"""Flask web interface for one-image VisionGuard inference."""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote
from uuid import uuid4

from dotenv import load_dotenv
from flask import (
    Flask,
    Request,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from inference import (
    ROOT,
    InvalidImageError,
    environment_model_config,
    load_model,
    predict,
)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif"}
DEFAULT_MAX_UPLOAD_BYTES = 8 * 1024 * 1024


class MemoryOnlyRequest(Request):
    """Keep multipart file streams in memory instead of Werkzeug's disk spooling."""

    def _get_file_stream(
        self,
        total_content_length: int | None,
        content_type: str | None,
        filename: str | None = None,
        content_length: int | None = None,
    ) -> BytesIO:
        return BytesIO()


def _database_path(database_url: str) -> Path:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise RuntimeError("DATABASE_URL must be a sqlite:/// URL.")
    raw_path = unquote(database_url[len(prefix) :])
    if raw_path == ":memory:":
        return Path(":memory:")
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path


class _ConnectionAdapter:
    def __init__(self, connection, postgres: bool):
        self._connection = connection
        self._postgres = postgres

    def execute(self, statement: str, parameters: tuple = ()):
        if self._postgres:
            statement = statement.replace("?", "%s")
        return self._connection.execute(statement, parameters)


@contextmanager
def _connect(
    database_url: str,
    postgres_options: dict | None = None,
) -> Iterator[_ConnectionAdapter]:
    if postgres_options is not None:
        import psycopg
        from psycopg.rows import dict_row

        connection = psycopg.connect(**postgres_options, row_factory=dict_row)
        postgres = True
    elif database_url.startswith("sqlite:///"):
        database_path = _database_path(database_url)
        if database_path != Path(":memory:"):
            database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(database_path), timeout=10)
        connection.row_factory = sqlite3.Row
        postgres = False
    elif database_url.startswith(("postgresql://", "postgres://")):
        import psycopg
        from psycopg.rows import dict_row

        connection = psycopg.connect(database_url, row_factory=dict_row)
        postgres = True
    else:
        raise RuntimeError(
            "DATABASE_URL must be a sqlite:/// or postgresql:// URL, "
            "or configure DB_HOST/DB_NAME/DB_USER/DB_PASSWORD."
        )
    adapter = _ConnectionAdapter(connection, postgres)
    try:
        yield adapter
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _initialize_database(database_url: str, postgres_options: dict | None = None) -> None:
    with _connect(database_url, postgres_options) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                ai_probability DOUBLE PRECISION NOT NULL,
                model_name TEXT
            )
            """
        )
        if postgres_options is None and database_url.startswith("sqlite:///"):
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(predictions)"
                ).fetchall()
            }
        else:
            columns = {
                row["column_name"]
                for row in connection.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = current_schema() AND table_name = 'predictions'
                    """
                ).fetchall()
            }
        if "model_name" not in columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN model_name TEXT")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_created_at "
            "ON predictions(created_at DESC)"
        )


def create_app(test_config: dict | None = None) -> Flask:
    load_dotenv(ROOT / ".env")
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.request_class = MemoryOnlyRequest
    model_path, threshold = environment_model_config()
    database_url = os.environ.get("DATABASE_URL", "").strip()
    postgres_options = None
    if os.environ.get("DB_HOST"):
        required_db_values = {
            "host": os.environ.get("DB_HOST", "").strip(),
            "dbname": os.environ.get("DB_NAME", "").strip(),
            "user": os.environ.get("DB_USER", "").strip(),
            "password": os.environ.get("DB_PASSWORD", ""),
        }
        if not all(required_db_values.values()):
            raise RuntimeError(
                "DB_HOST, DB_NAME, DB_USER, and DB_PASSWORD must all be configured."
            )
        try:
            db_port = int(os.environ.get("DB_PORT", "5432"))
        except ValueError as error:
            raise RuntimeError("DB_PORT must be an integer.") from error
        postgres_options = {**required_db_values, "port": db_port}
        database_url = f"postgresql://{required_db_values['host']}:{db_port}/{required_db_values['dbname']}"
    elif not database_url:
        raise RuntimeError(
            "Set DATABASE_URL or configure DB_HOST/DB_NAME/DB_USER/DB_PASSWORD."
        )

    try:
        max_upload_bytes = int(
            os.environ.get("MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES))
        )
    except ValueError as error:
        raise RuntimeError("MAX_UPLOAD_BYTES must be a positive integer.") from error
    if max_upload_bytes < 1:
        raise RuntimeError("MAX_UPLOAD_BYTES must be a positive integer.")

    app.config.update(
        MAX_CONTENT_LENGTH=max_upload_bytes,
        VISIONGUARD_THRESHOLD=threshold,
        VISIONGUARD_DATABASE_URL=database_url,
        VISIONGUARD_POSTGRES_OPTIONS=postgres_options,
        VISIONGUARD_MODEL_NAME=(
            os.environ.get("MODEL_NAME", "").strip() or model_path.stem
        ),
    )
    if test_config:
        app.config.update(test_config)
    model_loader = app.config.pop("VISIONGUARD_MODEL_LOADER", load_model)
    model = app.config.pop("VISIONGUARD_MODEL", None)
    if model is None:
        model = model_loader(model_path)
    app.extensions["visionguard_model"] = model
    _initialize_database(database_url, postgres_options)
    app.logger.info(
        "VisionGuard loaded once from %s; threshold=%.6f; database_backend=%s",
        model_path,
        threshold,
        "postgresql" if postgres_options or database_url.startswith(("postgresql://", "postgres://")) else "sqlite",
    )

    @app.get("/")
    def upload_page():
        return render_template(
            "upload.html",
            max_upload_mb=app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024),
        )

    @app.post("/predict")
    def predict_image():
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return render_template(
                "error.html",
                title="No image selected",
                message="Choose an image file and try again.",
                status_code=400,
            ), 400

        filename = secure_filename(uploaded.filename)
        if not filename:
            return render_template(
                "error.html",
                title="Invalid filename",
                message="The selected file has an invalid filename.",
                status_code=400,
            ), 400
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if extension not in ALLOWED_EXTENSIONS:
            return render_template(
                "error.html",
                title="Unsupported file type",
                message="Upload a JPG, JPEG, PNG, BMP, or GIF image.",
                status_code=400,
            ), 400

        image_bytes = uploaded.stream.read(app.config["MAX_CONTENT_LENGTH"] + 1)
        if len(image_bytes) > app.config["MAX_CONTENT_LENGTH"]:
            abort(413)
        try:
            result = predict(
                app.extensions["visionguard_model"],
                image_bytes,
                app.config["VISIONGUARD_THRESHOLD"],
            )
        except InvalidImageError:
            return render_template(
                "error.html",
                title="Image could not be read",
                message="This file does not contain a valid supported image. Try another image.",
                status_code=400,
            ), 400

        prediction_id = uuid4().hex
        created_at = datetime.now(UTC).isoformat(timespec="seconds")
        with _connect(
            app.config["VISIONGUARD_DATABASE_URL"],
            app.config["VISIONGUARD_POSTGRES_OPTIONS"],
        ) as connection:
            connection.execute(
                """
                INSERT INTO predictions
                    (id, created_at, filename, label, confidence, ai_probability, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    created_at,
                    filename,
                    result["label"],
                    result["confidence"],
                    result["ai_probability"],
                    app.config["VISIONGUARD_MODEL_NAME"],
                ),
            )
        return redirect(url_for("result_page", prediction_id=prediction_id))

    @app.get("/result/<prediction_id>")
    def result_page(prediction_id: str):
        with _connect(
            app.config["VISIONGUARD_DATABASE_URL"],
            app.config["VISIONGUARD_POSTGRES_OPTIONS"],
        ) as connection:
            row = connection.execute(
                "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
            ).fetchone()
        if row is None:
            abort(404)
        return render_template(
            "result.html",
            result=dict(row),
            threshold=app.config["VISIONGUARD_THRESHOLD"],
        )

    @app.get("/history")
    def history_page():
        with _connect(
            app.config["VISIONGUARD_DATABASE_URL"],
            app.config["VISIONGUARD_POSTGRES_OPTIONS"],
        ) as connection:
            page = request.args.get("page", default=1, type=int)
            page = max(page or 1, 1)
            page_size = 50
            total = connection.execute(
                "SELECT COUNT(*) AS count FROM predictions"
            ).fetchone()["count"]
            page_count = max((total + page_size - 1) // page_size, 1)
            page = min(page, page_count)
            rows = connection.execute(
                """
                SELECT id, created_at, filename, label, confidence, ai_probability, model_name
                FROM predictions ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (page_size, (page - 1) * page_size),
            ).fetchall()
        return render_template(
            "history.html",
            predictions=[dict(row) for row in rows],
            page=page,
            page_count=page_count,
            total=total,
        )

    @app.get("/health")
    def health_check():
        try:
            with _connect(
                app.config["VISIONGUARD_DATABASE_URL"],
                app.config["VISIONGUARD_POSTGRES_OPTIONS"],
            ) as connection:
                connection.execute("SELECT 1").fetchone()
        except sqlite3.Error:
            app.logger.exception("Health check failed while accessing prediction history.")
            return jsonify(status="unhealthy", model_loaded=True, database="unavailable"), 503
        return jsonify(status="healthy", model_loaded=True, database="available"), 200

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_error):
        return render_template(
            "error.html",
            title="File is too large",
            message=(
                f"Choose a file smaller than "
                f"{app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024):g} MB."
            ),
            status_code=413,
        ), 413

    @app.errorhandler(404)
    def not_found(_error):
        return render_template(
            "error.html",
            title="Page not found",
            message="That page or prediction does not exist.",
            status_code=404,
        ), 404

    @app.errorhandler(500)
    def internal_error(error):
        original = getattr(error, "original_exception", None)
        if original is not None:
            app.logger.error(
                "Unhandled request error: %s",
                original,
                exc_info=(type(original), original, original.__traceback__),
            )
        return render_template(
            "error.html",
            title="Something went wrong",
            message="The request could not be completed. Please try again.",
            status_code=500,
        ), 500

    @app.errorhandler(HTTPException)
    def http_error(error):
        if error.code == 413:
            return upload_too_large(error)
        return error

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    host = os.environ.get("HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("PORT", "5000"))
    except ValueError as error:
        raise RuntimeError("PORT must be an integer.") from error
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
