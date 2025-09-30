"""FastAPI server exposing ESP32 human detection controls."""

import logging
import threading
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from config import Config
from esp32_human_detection_v2 import ESP32HumanDetectionSystem

Config.load_from_env()

logger = logging.getLogger("esp32_api")
logger.setLevel(logging.INFO)

app = FastAPI(
    title="ESP32 Human Detection API",
    description="REST interface to control the ESP32 human detection runtime",
    version="1.0.0",
)

_control_lock = threading.Lock()
_detector: Optional[ESP32HumanDetectionSystem] = None


class StartRequest(BaseModel):
    esp32_url: Optional[str] = None
    model_path: Optional[str] = None
    output_dir: Optional[str] = None
    force_reload: bool = False


class StatusResponse(BaseModel):
    running: bool
    is_recording: bool
    humans_detected: int
    fps: float
    last_detection_time: Optional[str]
    current_recording: Optional[str]
    esp32_url: Optional[str]
    model_path: Optional[str]
    output_dir: Optional[str]


@app.get("/")
def read_root() -> dict:
    return {
        "message": "ESP32 human detection API",
        "routes": [
            "/api/start",
            "/api/stop",
            "/api/status",
            "/api/snapshot",
            "/api/recordings",
        ],
    }


def _create_detector(esp32_url: str, model_path: str, output_dir: str) -> ESP32HumanDetectionSystem:
    logger.info(
        "Creating detection system",
        extra={
            "esp32_url": esp32_url,
            "model_path": model_path,
            "output_dir": output_dir,
        },
    )
    return ESP32HumanDetectionSystem(
        esp32_url=esp32_url,
        model_path=model_path,
        output_dir=output_dir,
    )


def _status_payload(detector: Optional[ESP32HumanDetectionSystem]) -> StatusResponse:
    if not detector:
        return StatusResponse(
            running=False,
            is_recording=False,
            humans_detected=0,
            fps=0.0,
            last_detection_time=None,
            current_recording=None,
            esp32_url=Config.ESP32_URL,
            model_path=Config.MODEL_PATH,
            output_dir=Config.OUTPUT_DIR,
        )

    data = detector.get_status()
    return StatusResponse(
        running=bool(data.get("running", detector.running)),
        is_recording=bool(data.get("is_recording", False)),
        humans_detected=int(data.get("humans_detected", 0)),
        fps=float(data.get("fps", 0.0)),
        last_detection_time=data.get("last_detection_time"),
        current_recording=data.get("current_recording"),
        esp32_url=detector.esp32_url,
        model_path=detector.model_path,
        output_dir=detector.output_dir,
    )


@app.post("/api/start", response_model=StatusResponse)
def start_detection(payload: Optional[StartRequest] = None) -> StatusResponse:
    global _detector
    payload = payload or StartRequest()

    with _control_lock:
        current = _detector
        requested_url = payload.esp32_url or (current.esp32_url if current else Config.ESP32_URL)
        requested_model = payload.model_path or (current.model_path if current else Config.MODEL_PATH)
        requested_output = payload.output_dir or (current.output_dir if current else Config.OUTPUT_DIR)

        needs_new = (
            current is None
            or payload.force_reload
            or current.esp32_url != requested_url
            or current.model_path != requested_model
            or current.output_dir != requested_output
        )

        if needs_new:
            if current and current.running:
                current.stop()
            _detector = _create_detector(requested_url, requested_model, requested_output)
            current = _detector

        if current.running:
            raise HTTPException(status_code=409, detail="Detection system already running")

        current.start()

        return _status_payload(current)


@app.post("/api/stop")
def stop_detection() -> dict:
    with _control_lock:
        if not _detector or not _detector.running:
            raise HTTPException(status_code=409, detail="Detection system is not running")
        _detector.stop()
    return {"status": "stopped"}


@app.get("/api/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    return _status_payload(_detector)


@app.get("/api/snapshot")
def get_snapshot() -> Response:
    if not _detector or not _detector.running:
        raise HTTPException(status_code=409, detail="Detection system is not running")

    frame = _detector.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available yet")

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode snapshot")

    data = buffer.tobytes()
    headers = {"Content-Length": str(len(data))}
    return Response(content=data, media_type="image/jpeg", headers=headers)


@app.get("/api/recordings")
def list_recordings(limit: int = Query(20, ge=1, le=100)) -> dict:
    if _detector:
        files = _detector.list_recordings(limit=limit)
        base_dir = _detector.output_dir
    else:
        base_dir = Config.OUTPUT_DIR
        output_path = Path(base_dir)
        if output_path.exists():
            files = sorted((p.name for p in output_path.glob("*.mp4")), reverse=True)[:limit]
        else:
            files = []

    return {
        "output_dir": str(Path(base_dir).resolve()),
        "files": list(files),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
