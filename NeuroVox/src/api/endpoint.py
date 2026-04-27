import os
import uuid
import asyncio
from contextlib import asynccontextmanager
from src.constant.constant import model_path
from concurrent.futures import ThreadPoolExecutor
from src.inference.predictor import NeuroVoxPredictor
from fastapi import FastAPI, BackgroundTasks, File, HTTPException, UploadFile


TMP_DIR = "tmp"
executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(TMP_DIR, exist_ok=True)
    print("Loading model...")
    app.state.model = NeuroVoxPredictor(model_path)

    yield
    executor.shutdown(wait=True)
    print("Shutting down...")


app = FastAPI(title="NeuroVox API", lifespan=lifespan)


def remove_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Error deleting file {path}: {e}")


@app.post("/predict")
async def predict_audio(background_task: BackgroundTasks, audio: UploadFile = File(...)):
    if not audio.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files supported")

    tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}.wav")

    try:
        with open(tmp_path, "wb") as buffer:
            while content := await audio.read(1024 * 1024):
                buffer.write(content)

        loop = asyncio.get_running_loop()
        label, probability = await loop.run_in_executor(
            executor, app.state.model.predict, tmp_path
        )

        if label == "Error":
            remove_file(tmp_path)
            raise HTTPException(status_code=500, detail="Processing failed")

        background_task.add_task(remove_file, tmp_path)

        return {"label": label, "probability": round(float(probability), 4)}

    except Exception as e:
        remove_file(tmp_path)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
