from __future__ import annotations

import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from cloudrun_api import create_app


def make_png() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (4, 3), color=(100, 120, 140)).save(stream, format="PNG")
    return stream.getvalue()


class FakeModel:
    def predict(self, batch, verbose=0):
        if batch.shape != (1, 160, 160, 3):
            raise AssertionError(f"Unexpected inference shape: {batch.shape}")
        return [[0.9]]


class CloudRunApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        model_path = Path(self.temp_dir.name) / "unused.keras"
        model_path.write_bytes(b"test model placeholder")
        os.environ["MODEL_PATH"] = str(model_path)
        os.environ["THRESHOLD"] = "0.85"
        os.environ["MODEL_NAME"] = "VisionGuard Cloud Run test model"
        os.environ["MODEL_API_KEY"] = "test-api-key"
        self.app = create_app({"TESTING": True, "CLOUDRUN_MODEL": FakeModel()})
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()
        for key in ("MODEL_PATH", "THRESHOLD", "MODEL_NAME", "MODEL_API_KEY"):
            os.environ.pop(key, None)

    def test_health_is_available(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"model_loaded": True, "status": "healthy"})

    def test_prediction_requires_bearer_key(self):
        response = self.client.post(
            "/api/v1/predict",
            data={"image": (BytesIO(make_png()), "test.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 401)

    def test_prediction_returns_json_model_result(self):
        response = self.client.post(
            "/api/v1/predict",
            headers={"Authorization": "Bearer test-api-key"},
            data={"image": (BytesIO(make_png()), "test.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["label"], "AI-generated")
        self.assertEqual(response.json["confidence"], 0.9)
        self.assertEqual(response.json["ai_probability"], 0.9)
        self.assertEqual(
            response.json["model_name"], "VisionGuard Cloud Run test model"
        )
        self.assertTrue(response.json["created_at"])

    def test_prediction_rejects_unsupported_extension(self):
        response = self.client.post(
            "/api/v1/predict",
            headers={"Authorization": "Bearer test-api-key"},
            data={"image": (BytesIO(make_png()), "notes.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
