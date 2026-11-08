# VisionGuard

[![VisionGuard CI](https://github.com/saboor-sarfraz/mlops-a2/actions/workflows/ci.yml/badge.svg)](https://github.com/saboor-sarfraz/mlops-a2/actions/workflows/ci.yml)

VisionGuard is a Flask image classifier with a pretrained Keras model and persistent prediction history.

## Run with Docker Compose

1. If `.env` does not exist, copy `.env.example` to `.env`. Otherwise, keep the existing file and add the `POSTGRES_*` and `APP_PORT` entries from `.env.example`. Replace the password template with a long random alphanumeric password. The `.env` file is git-ignored; never commit it.
2. From the repository root, run `docker compose up --build -d`.
3. Wait until both services report `healthy` using `docker compose ps`.
4. Open `http://127.0.0.1:5001`; change `APP_PORT` in `.env` if that port is in use.
5. Stop the stack with `docker compose down`. To preserve prediction history, do not add `-v`.

The Postgres database is reachable by the app at the Compose service name `db`; it has no published host port and stores data in the named `prediction-data` volume. The selected champion `.keras` file is the only file allowed through the `visionguard_outputs/` Git ignore rule and is copied into the app image, so CI can build the same model regardless of other local outputs.

The Docker image uses `tensorflow-cpu` because this web inference workload does not require GPU acceleration. Dependencies are installed before application source is copied so source-only changes can reuse the dependency layer. `.dockerignore` keeps datasets, tracking stores, secrets, and unrelated outputs out of the build context. The app runs as UID 10001, uses a read-only root filesystem, and has an HTTP health check.

## Operations

```powershell
docker compose up --build -d
docker compose ps
docker compose logs -f web
docker compose logs -f db
docker compose exec web sh
docker compose exec db sh
docker compose restart web
docker compose stop
docker compose start
docker compose down
```

Postgres credentials are supplied through the ignored `.env` file, not embedded in the Dockerfile, image, workflow, or source. GitHub Actions secrets, when needed for a future registry push or deployment, belong in repository Settings → Secrets and variables → Actions and should be accessed through `${{ secrets.NAME }}`.

## Optional public model API on Google Cloud Run

The Cloud Run service is a separate, authenticated inference API; the browser Flask app remains local. The API returns JSON from `POST /api/v1/predict`, accepting a multipart file field named `image`. The local Flask app reads `MODEL_API_URL` and `MODEL_API_KEY` from the ignored local `.env`, forwards image bytes over HTTPS, and saves the returned decision and threshold to SQLite history.

The deployed endpoint URL and model API key have been configured in the ignored `.env` for this workspace. To use the browser frontend from PowerShell in `S:\A2`, run:

```powershell
.\run_flask.ps1
```

Open `http://127.0.0.1:5000`, submit images through the form, and choose **History** to see saved predictions. Keep the terminal open while using the app and press `Ctrl+C` to stop it. Do not copy or share the `.env` file.

To reproduce deployment after installing the Google Cloud CLI, authenticating, selecting a project with billing enabled, and reviewing current Cloud Run/Cloud Build/Artifact Registry/Secret Manager prices, run:

```powershell
gcloud auth login
.\deploy_cloudrun.ps1 -ProjectId YOUR_PROJECT_ID -Region us-central1
```

The script builds `Dockerfile.cloudrun` with Cloud Build, pushes the image to Artifact Registry, creates a Secret Manager API key, grants the Cloud Run identity access to it, and deploys with zero minimum instances and a maximum of two. It prints the actual service and prediction URLs, but not the key. Run `.\run_flask.ps1` again to make local browser predictions use the remote model. To stop ongoing service compute charges after collecting evidence:

```powershell
gcloud run services delete visionguard-model-api --region YOUR_REGION
```

Artifact Registry and Secret Manager resources are separate; review and remove them only after preserving report evidence. The report builder leaves the actual deployed URL and Postman result as placeholders until a real deployment/test is completed.

## GitHub Actions toy exercise

`toy-ci/` is a small copyable exercise. Copy its contents (including `.github/workflows/ci.yml`) into a throwaway GitHub repository, push to run the Python 3.10/3.12 matrix, temporarily change the expected result in `test_math.py` to see a red run, then restore `5` and push again for green. The live VisionGuard workflow runs lint, unit tests on Python 3.11 and 3.12, then builds the Docker image.

## Docker vocabulary

- **Image:** a packaged template of the application, runtime, and filesystem layers used to start containers.
- **Container:** an isolated running instance created from an image.
- **Layer:** a cached, read-only filesystem change in an image; later Dockerfile steps build on earlier layers.
- **Volume:** Docker-managed persistent storage that remains after a container is replaced.
- **Port mapping:** a host-to-container network forwarding rule, such as host port 5001 to app port 5000.
