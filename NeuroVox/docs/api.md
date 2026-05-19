# API Reference

## Overview

NeuroVox exposes a single REST endpoint built with **FastAPI**. The API accepts `.wav` audio file uploads and returns a JSON prediction with a label and confidence probability.

**Base URL:** `http://localhost:8000`  
**Interactive Docs (Swagger UI):** `http://localhost:8000/docs`  
**ReDoc:** `http://localhost:8000/redoc`

---

## Application Lifecycle

The FastAPI app uses a `lifespan` context manager to handle startup and shutdown cleanly:

```
App Startup
  ├── Create tmp/ directory for temporary audio files
  └── Load NeuroVoxPredictor (ONNX model) into app.state.model

App Shutdown
  └── Shutdown ThreadPoolExecutor (wait for running tasks to finish)
```

The ONNX session is initialized **once** at startup and shared across all requests, avoiding expensive reload overhead on every call.

---

## Concurrency Model

The API is fully async. Since ONNX inference is CPU-bound (blocking), it is offloaded to a `ThreadPoolExecutor` with `max_workers=4`:

```python
label, probability = await loop.run_in_executor(
    executor, app.state.model.predict, tmp_path
)
```

This keeps the async event loop non-blocking while inference runs in a background thread, allowing the server to handle multiple concurrent requests.

---

## Endpoints

### `POST /predict`

Classify a voice recording as Healthy Control or Parkinson's Disease.

#### Request

| Field   | Type               | Required | Description         |
| ------- | ------------------ | -------- | ------------------- |
| `audio` | `file` (form-data) | ✅       | A `.wav` audio file |

**Content-Type:** `multipart/form-data`

> **Note:** Only `.wav` files are accepted. Submitting any other format returns a `400` error.

#### File Handling

Uploaded files are:

1. Written to `tmp/<uuid>.wav` in chunks of 1 MB to avoid loading the entire file into memory
2. Passed to the predictor for inference
3. Deleted asynchronously via `BackgroundTasks` after the response is returned

#### Response

```json
{
  "label": "HC",
  "probability": 0.8921
}
```

| Field         | Type     | Description                                                        |
| ------------- | -------- | ------------------------------------------------------------------ |
| `label`       | `string` | `"HC"` (Healthy Control) or `"PD"` (Parkinson's Disease)           |
| `probability` | `float`  | Sigmoid confidence score ∈ [0.0, 1.0], rounded to 4 decimal places |

**Interpretation:**

| `probability` | `label` | Meaning                       |
| ------------- | ------- | ----------------------------- |
| > 0.5         | `HC`    | Predicted Healthy Control     |
| ≤ 0.5         | `PD`    | Predicted Parkinson's Disease |

#### Status Codes

| Code                        | Condition                            |
| --------------------------- | ------------------------------------ |
| `200 OK`                    | Successful prediction                |
| `400 Bad Request`           | File is not a `.wav` file            |
| `500 Internal Server Error` | Inference failed or unexpected error |

---

## Example Requests

### `curl`

```bash
curl -X POST http://localhost:8000/predict \
  -F "audio=@voice_sample.wav"
```

### Python `requests`

```python
import requests

with open("voice_sample.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"audio": ("voice_sample.wav", f, "audio/wav")},
    )

print(response.status_code)  # 200
print(response.json())       # {'label': 'HC', 'probability': 0.8921}
```

### JavaScript `fetch`

```javascript
const formData = new FormData();
formData.append("audio", fileInput.files[0]);

const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
});

const result = await response.json();
console.log(result); // { label: "HC", probability: 0.8921 }
```

### HTTPie

```bash
http POST localhost:8000/predict audio@voice_sample.wav
```

---

## Error Responses

All error responses follow FastAPI's standard structure:

```json
{
  "detail": "Human-readable error message"
}
```

**Examples:**

```json
// 400 — wrong file format
{ "detail": "Only .wav files supported" }

// 500 — inference failure
{ "detail": "Internal Server Error: <exception message>" }
```

---

## Running the Server

```bash
# Default (host=0.0.0.0, port=8000)
python -m src.api.main

# Custom host/port
uvicorn src.api.endpoint:app --host 0.0.0.0 --port 8080

# Development mode with auto-reload
uvicorn src.api.endpoint:app --reload
```

---

## Configuration

| Setting       | Value     | Description                               |
| ------------- | --------- | ----------------------------------------- |
| `TMP_DIR`     | `"tmp"`   | Directory for temporary uploaded files    |
| `max_workers` | `4`       | Thread pool size for concurrent inference |
| Host          | `0.0.0.0` | Binds to all network interfaces           |
| Port          | `8000`    | Default port                              |

The ONNX model path is read from `src/constant/constant.py`:

```python
model_path = base_dir / "checkpoint" / "best_model.onnx"
```
