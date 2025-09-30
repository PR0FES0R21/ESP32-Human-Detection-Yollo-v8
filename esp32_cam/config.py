import json
import logging
import os
import urllib.parse
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class Config:
    """Central configuration for ESP32-CAM detection services."""

    # ESP32-CAM stream and control
    ESP32_URL = "http://192.168.20.152:81/stream"
    ESP32_CONTROL_PORT: Optional[int] = None  # Autodetect from stream URL when None
    ESP32_CONTROL_PATH = "/control"
    CAMERA_REQUEST_TIMEOUT = 3.0
    CAMERA_SETTINGS: Dict[str, int] = {
        "vflip": 1,
        "hmirror": 1,
        "framesize": 10,
        "gainceiling": 1,
    }

    # Detection / model
    MODEL_PATH = "yolo8n.pt"
    CONFIDENCE_THRESHOLD = 0.5
    PERSON_CLASS_ID = 0
    SKIP_FRAMES = 3

    # Performance / buffering
    MAX_QUEUE_SIZE = 5
    DISPLAY_FPS = 30
    RECORDING_FPS = 20
    PRE_DETECTION_BUFFER_SECONDS = 3.0
    ESTIMATED_FPS_FOR_BUFFER = 20.0

    # Recording
    OUTPUT_DIR = "recordings"
    RECORDING_DELAY = 10.0
    VIDEO_CODEC = "mp4v"

    # Stream stability
    STREAM_BUFFER_SIZE = 1
    RECONNECT_DELAY = 5.0

    # Display styling
    WINDOW_NAME = "ESP32-CAM Human Detection"
    BBOX_COLOR = (0, 255, 0)
    BBOX_THICKNESS = 2
    TEXT_COLOR = (0, 255, 0)
    TEXT_SCALE = 0.7
    TEXT_THICKNESS = 2

    @classmethod
    def load_from_env(cls) -> None:
        """Override configuration values from environment variables."""
        cls.ESP32_URL = os.getenv("ESP32_URL", cls.ESP32_URL)
        cls.MODEL_PATH = os.getenv("MODEL_PATH", cls.MODEL_PATH)
        cls.OUTPUT_DIR = os.getenv("OUTPUT_DIR", cls.OUTPUT_DIR)

        cls.CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", cls.CONFIDENCE_THRESHOLD))
        cls.SKIP_FRAMES = int(os.getenv("SKIP_FRAMES", cls.SKIP_FRAMES))
        cls.RECORDING_DELAY = float(os.getenv("RECORDING_DELAY", cls.RECORDING_DELAY))
        cls.RECONNECT_DELAY = float(os.getenv("RECONNECT_DELAY", cls.RECONNECT_DELAY))
        cls.RECORDING_FPS = float(os.getenv("RECORDING_FPS", cls.RECORDING_FPS))
        cls.DISPLAY_FPS = float(os.getenv("DISPLAY_FPS", cls.DISPLAY_FPS))
        cls.PRE_DETECTION_BUFFER_SECONDS = float(
            os.getenv("PRE_DETECTION_BUFFER_SECONDS", cls.PRE_DETECTION_BUFFER_SECONDS)
        )
        cls.ESTIMATED_FPS_FOR_BUFFER = float(
            os.getenv("ESTIMATED_FPS_FOR_BUFFER", cls.ESTIMATED_FPS_FOR_BUFFER)
        )

        control_port = os.getenv("ESP32_CONTROL_PORT")
        if control_port:
            try:
                cls.ESP32_CONTROL_PORT = int(control_port)
            except ValueError:
                logger.warning("Invalid ESP32_CONTROL_PORT value '%s'", control_port)

        cls.ESP32_CONTROL_PATH = os.getenv("ESP32_CONTROL_PATH", cls.ESP32_CONTROL_PATH)
        cls.CAMERA_REQUEST_TIMEOUT = float(
            os.getenv("ESP32_CAMERA_TIMEOUT", cls.CAMERA_REQUEST_TIMEOUT)
        )

        camera_settings_env = os.getenv("ESP32_CAMERA_SETTINGS")
        if camera_settings_env:
            try:
                settings = json.loads(camera_settings_env)
                if isinstance(settings, dict):
                    cls.CAMERA_SETTINGS = {str(k): int(v) for k, v in settings.items()}
                else:
                    logger.warning("ESP32_CAMERA_SETTINGS should be a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Failed to parse ESP32_CAMERA_SETTINGS: %s", exc)

    @classmethod
    def camera_settings(cls) -> Dict[str, int]:
        """Return a copy of the configured camera settings."""
        return dict(cls.CAMERA_SETTINGS)

    @classmethod
    def derive_control_base_url(cls, stream_url: str) -> Optional[str]:
        """Build the base URL used to reach the ESP32 control endpoint."""
        try:
            parsed = urllib.parse.urlparse(stream_url)
            if not parsed.scheme or not parsed.hostname:
                return None

            hostname = parsed.hostname
            scheme = parsed.scheme

            target_port = cls.ESP32_CONTROL_PORT
            if target_port is None:
                target_port = parsed.port
                if target_port is None:
                    target_port = 80 if scheme == "http" else 443
                if parsed.port == 81 and target_port == 81:
                    target_port = 80

            if (scheme == "http" and target_port == 80) or (scheme == "https" and target_port == 443):
                netloc = hostname
            else:
                netloc = f"{hostname}:{target_port}"

            return urllib.parse.urlunparse((scheme, netloc, "", "", "", "")).rstrip('/')
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Unable to derive control URL from '%s': %s", stream_url, exc)
            return None

    @classmethod
    def compose_control_url(cls, base_url: str, setting: str, value: int) -> str:
        """Compose a full control URL for the given setting/value."""
        return f"{base_url}{cls.ESP32_CONTROL_PATH}?var={setting}&val={value}"
