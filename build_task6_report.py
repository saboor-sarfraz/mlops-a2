"""Build the editable Tasks 1-6 report, preserving the original Tasks 1-3 source."""

from __future__ import annotations

from docx import Document

import build_task5_report

ROOT = build_task5_report.ROOT
SOURCE_REPORT = ROOT / "23L-0522_VisionGuard_Tasks1-5.docx"
OUTPUT_REPORT = ROOT / "23L-0522_VisionGuard_Tasks1-6.docx"
add_code_block = build_task5_report.add_code_block
add_screenshot_placeholder = build_task5_report.add_screenshot_placeholder
add_table = build_task5_report.add_table


def main() -> None:
    if not SOURCE_REPORT.is_file():
        build_task5_report.main()
    document = Document(SOURCE_REPORT)
    if any(paragraph.text.strip() == "6. Containerization and CI" for paragraph in document.paragraphs):
        raise RuntimeError(f"Task 6 already exists in {SOURCE_REPORT.name}.")

    document.add_page_break()
    document.add_heading("6. Containerization and CI", level=1)
    document.add_paragraph(
        "VisionGuard is packaged as a CPU-only Flask image and a PostgreSQL-backed Compose "
        "stack. The app connects to the database through the Compose service name `db`; "
        "PostgreSQL is not published to the host. Compose waits for the database health check "
        "before starting the web service. The model file is copied into the image so every "
        "container uses the same fixed champion artifact without relying on a host mount."
    )

    document.add_heading("6.1 Docker warm-up", level=2)
    add_code_block(
        document,
        "docker pull nginx:alpine\n"
        "docker volume create vg-warmup-data-task6-20261108\n"
        "docker run -d --name vg-warmup-task6-20261108 -p 127.0.0.1:18081:80 "
        "-v vg-warmup-data-task6-20261108:/usr/share/nginx/html nginx:alpine\n"
        "docker ps -a\n"
        "docker logs vg-warmup-task6-20261108\n"
        "docker exec vg-warmup-task6-20261108 sh -c 'id; ls -la /usr/share/nginx/html'\n"
        "docker stop vg-warmup-task6-20261108\n"
        "docker rm vg-warmup-task6-20261108\n"
        "docker image ls\n"
        "docker image rm nginx:alpine\n"
        "docker volume rm vg-warmup-data-task6-20261108",
    )
    add_table(
        document,
        ["Term", "Definition"],
        [
            ["Image", "Packaged filesystem and metadata used as a template for containers."],
            ["Container", "An isolated running instance of an image."],
            ["Layer", "A cached, read-only filesystem change that later image steps build upon."],
            ["Volume", "Docker-managed persistent storage that outlives a container."],
            ["Port mapping", "A host-to-container forwarding rule that exposes a service."],
        ],
    )
    add_screenshot_placeholder(
        document,
        "Docker warm-up evidence",
        "Paste terminal screenshots for pull/run and mapped port, volume mount, logs, shell, container/image listing, and cleanup.",
        height=2.2,
    )

    document.add_heading("6.2 Dockerfile and image security", level=2)
    document.add_paragraph(
        "Dockerfile starts with the slim Python 3.11 image and installs only the inference "
        "runtime: tensorflow-cpu 2.21.0 (matching the model's training runtime), Flask, "
        "Gunicorn, Pillow, python-dotenv, and the PostgreSQL driver. CPU TensorFlow is "
        "deliberate: this inference service does not require a GPU, and using the CPU wheel "
        "avoids a CUDA runtime and its additional image size. The training libraries and "
        "tracking clients are not installed into the serving image."
    )
    document.add_paragraph(
        "The requirements file is copied and installed before application code. Docker can "
        "reuse that expensive dependency layer when only source or templates change. "
        "`.dockerignore` omits datasets, DVC/MLflow data, local databases, reports, and "
        "secrets from the build context. The model is the one intentional training output "
        "copied into the image. No `.env` file or password is copied. The process runs as "
        "UID 10001, with a read-only root filesystem in Compose, a bounded temporary "
        "filesystem, and a health check."
    )
    add_code_block(
        document,
        "docker build -t visionguard:latest .\n"
        "docker run --rm --entrypoint id visionguard:latest\n"
        "docker image inspect visionguard:latest --format '{{.Size}}'\n"
        "docker history visionguard:latest",
    )
    add_screenshot_placeholder(
        document,
        "Dockerfile and non-root identity",
        "Paste the Dockerfile and `docker compose exec web id` output showing uid=10001.",
        height=1.6,
    )

    document.add_heading("6.3 Compose services, secrets, and model", level=2)
    document.add_paragraph(
        "The `web` and `db` services share Compose's private network and the web process "
        "connects to host `db`. Postgres uses the persistent named volume "
        "`prediction-data` and its `pg_isready` health check; `depends_on: condition: "
        "service_healthy` gates app startup. Only the Flask service publishes a loopback "
        "host port (5001 by default); Postgres has no `ports` mapping. Set credentials in "
        "the ignored `.env` file (create it from `.env.example` if it does not exist); environment interpolation "
        "supplies the password to both services without baking it into the image or "
        "repository. The champion model is copied into the image for repeatable inference."
    )
    add_code_block(
        document,
        "# Create .env from .env.example if needed; otherwise add its POSTGRES_* values.\n"
        "# Replace POSTGRES_PASSWORD with a strong random alphanumeric value.\n"
        "docker compose up --build -d\n"
        "docker compose ps\n"
        "docker inspect a2-db-1 --format '{{json .NetworkSettings.Ports}}'\n"
        "Test-NetConnection 127.0.0.1 -Port 5432",
    )
    add_screenshot_placeholder(
        document,
        "Healthy Compose services and closed database port",
        "Paste `docker compose ps` showing web/db healthy, `5432/tcp: null` from inspect, and a failed host connection test.",
        height=1.8,
    )

    document.add_heading("6.4 Image size optimization", level=2)
    document.add_paragraph(
        "The optimization is using python:3.11-slim instead of the full python:3.11 base "
        "while installing only the CPU inference dependencies. The build also disables pip "
        "cache and excludes the archive, model-training outputs, and tracking artifacts from "
        "the context. The base image dropped from 1,598,274,424 bytes to 187,094,560 bytes "
        "(88.3% smaller). The complete optimized app image is 2,228,923,654 bytes. "
        "`docker history` reports the dependency-install layer at 1.65 GB, the largest "
        "layer; the copied champion model is 10.3 MB."
    )
    add_table(
        document,
        ["Build", "Image size", "Notes"],
        [
            ["Before: python:3.11 base", "1,598,274,424 bytes (1.60 GB)", "Full Python runtime base"],
            ["After: python:3.11-slim base", "187,094,560 bytes (187 MB)", "88.3% smaller base; app uses this pinned slim base"],
            ["Complete optimized VisionGuard image", "2,228,923,654 bytes (2.23 GB)", "Includes CPU TensorFlow, app, and 10.3 MB model"],
        ],
    )
    add_screenshot_placeholder(
        document,
        "Before and after image sizes",
        "Paste `docker image inspect` output for python:3.11 and python:3.11-slim plus `docker history a2-web`.",
        height=1.5,
    )

    document.add_heading("6.5 Operations and troubleshooting", level=2)
    add_code_block(
        document,
        "docker compose up --build -d\n"
        "docker compose stop\n"
        "docker compose start\n"
        "docker compose restart web\n"
        "docker compose logs --tail=100 web\n"
        "docker compose logs --tail=100 db\n"
        "docker compose exec web sh\n"
        "docker compose exec db sh",
    )
    document.add_paragraph(
        "Troubleshooting demonstration: temporarily provide a wrong DB_PASSWORD to the web "
        "service while leaving the initialized Postgres volume unchanged. The web startup "
        "log should show PostgreSQL password authentication failure; the database log should "
        "show the rejected connection. Restore the matching `.env` password and recreate "
        "the web service. Capture the actual error and recovered healthy status rather than "
        "presenting a simulated log."
    )
    add_screenshot_placeholder(
        document,
        "Failure diagnosis and recovery",
        "Paste real web/database log excerpts showing the authentication failure, diagnosis, corrected environment, and healthy recovery.",
        height=2,
    )

    document.add_heading("6.6 GitHub Actions", level=2)
    document.add_paragraph(
        "The VisionGuard CI workflow triggers on every push and pull request. A Python 3.11 "
        "and 3.12 matrix installs the serving requirements, runs Ruff, and executes the "
        "Flask unit tests. A dependent job builds the Docker image without pushing it. The "
        "repository README includes the workflow status badge. A copyable toy workflow and "
        "unit test are in `toy-ci/`; copy those files to a throwaway GitHub repository, push "
        "to see both matrix jobs, change the expected sum to produce a failing run, then "
        "restore it and push to get a green run."
    )
    document.add_paragraph(
        "Workflow secrets are stored in GitHub repository Settings → Secrets and variables "
        "→ Actions. They are injected at runtime as `${{ secrets.NAME }}` only for steps "
        "that need them (for example, a future registry push). Never place credentials in "
        "workflow YAML, source, image layers, logs, or committed `.env` files: those "
        "locations are versioned or visible to users who can read the repository/build."
    )
    add_screenshot_placeholder(
        document,
        "GitHub Actions evidence",
        "Paste the toy workflow red then green matrix runs, VisionGuard CI green matrix, Docker build job, and README badge.",
        height=2.2,
    )

    document.save(OUTPUT_REPORT)
    print(f"Task 1-6 editable report saved to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
