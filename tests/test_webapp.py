from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path

from PIL import Image

from webapp import create_app


def make_png() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (4, 3), color=(100, 120, 140)).save(stream, format="PNG")
    return stream.getvalue()


PNG_BYTES = make_png()


class FakeModel:
    def predict(self, batch, verbose=0):
        if batch.shape != (1, 160, 160, 3):
            raise AssertionError(f"Unexpected inference shape: {batch.shape}")
        return [[0.9]]


class VisionGuardWebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        model_path = self.root / "dummy.keras"
        model_path.write_bytes(b"test model placeholder")
        os.environ["MODEL_PATH"] = str(model_path)
        os.environ["THRESHOLD"] = "0.85"
        os.environ["MODEL_NAME"] = "VisionGuard test champion"
        os.environ["DATABASE_URL"] = f"sqlite:///{(self.root / 'history.db').as_posix()}"
        os.environ["MAX_UPLOAD_BYTES"] = "8388608"
        self.app = create_app({"TESTING": True, "VISIONGUARD_MODEL": FakeModel()})
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()
        for key in (
            "MODEL_PATH",
            "THRESHOLD",
            "MODEL_NAME",
            "DATABASE_URL",
            "MAX_UPLOAD_BYTES",
        ):
            os.environ.pop(key, None)

    def test_pages_and_health(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/history").status_code, 200)
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "healthy")

    def test_successful_prediction_and_metadata_only_history(self):
        response = self.client.post(
            "/predict",
            data={"image": (BytesIO(PNG_BYTES), "../../private image.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 302)
        result = self.client.get(response.headers["Location"])
        self.assertEqual(result.status_code, 200)
        self.assertIn(b"AI-generated", result.data)
        self.assertIn(b"90.0%", result.data)
        self.assertIn(b"VisionGuard test champion", result.data)
        history = self.client.get("/history")
        self.assertIn(b"private_image.png", history.data)
        self.assertIn(b"VisionGuard test champion", history.data)
        with closing(sqlite3.connect(self.root / "history.db")) as connection:
            row = connection.execute(
                """
                SELECT created_at, filename, label, confidence, ai_probability, model_name
                FROM predictions
                """
            ).fetchone()
        self.assertIsNotNone(row[0])
        self.assertEqual(row[1:], (
            "private_image.png",
            "AI-generated",
            0.9,
            0.9,
            "VisionGuard test champion",
        ))
        self.assertEqual(
            sorted(path.name for path in self.root.iterdir()),
            ["dummy.keras", "history.db"],
        )

    def test_existing_database_is_migrated_without_losing_history(self):
        database_path = self.root / "legacy.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{database_path.as_posix()}"
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(
                """
                CREATE TABLE predictions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    ai_probability REAL NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
                ("old-record", "2026-01-01T00:00:00+00:00", "old.png", "Real", .8, .2),
            )
            connection.commit()
        self.app = create_app({"TESTING": True, "VISIONGUARD_MODEL": FakeModel()})
        response = self.app.test_client().get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Unknown (older record)", response.data)
        with closing(sqlite3.connect(database_path)) as connection:
            row = connection.execute(
                "SELECT id, filename, label, model_name FROM predictions"
            ).fetchone()
        self.assertEqual(row, ("old-record", "old.png", "Real", None))

    def test_history_paginates_older_records(self):
        with closing(sqlite3.connect(self.root / "history.db")) as connection:
            connection.executemany(
                """
                INSERT INTO predictions
                    (id, created_at, filename, label, confidence, ai_probability, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        f"record-{index}",
                        (
                            datetime(2026, 1, 1, tzinfo=UTC)
                            + timedelta(days=index)
                        ).isoformat(),
                        f"image-{index}.png",
                        "Real",
                        0.8,
                        0.2,
                        "VisionGuard test champion",
                    )
                    for index in range(51)
                ],
            )
            connection.commit()
        first_page = self.client.get("/history")
        self.assertIn(b"51 prediction records", first_page.data)
        self.assertIn(b"Page 1 of 2", first_page.data)
        self.assertNotIn(b"image-0.png", first_page.data)
        second_page = self.client.get("/history?page=2")
        self.assertIn(b"Page 2 of 2", second_page.data)
        self.assertIn(b"image-0.png", second_page.data)

    def test_rejects_wrong_extension(self):
        response = self.client.post(
            "/predict",
            data={"image": (BytesIO(PNG_BYTES), "notes.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Unsupported file type", response.data)

    def test_rejects_corrupt_image_bytes(self):
        response = self.client.post(
            "/predict",
            data={"image": (BytesIO(b"not an image"), "broken.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Image could not be read", response.data)

    def test_rejects_over_limit_upload(self):
        self.app.config["MAX_CONTENT_LENGTH"] = 128
        response = self.client.post(
            "/predict",
            data={"image": (BytesIO(PNG_BYTES * 100), "large.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn(b"File is too large", response.data)


if __name__ == "__main__":
    unittest.main()
