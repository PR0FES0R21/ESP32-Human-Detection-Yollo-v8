import os

class Config:
    # ESP32-CAM Settings
    ESP32_URL = "http://192.168.145.152:81/stream"  # Change this to your ESP32-CAM IP
    
    # YOLO Model Settings
    MODEL_PATH = "yolo8n.pt"  # Will be downloaded automatically
    CONFIDENCE_THRESHOLD = 0.5
    PERSON_CLASS_ID = 0
    
    # Performance Settings
    SKIP_FRAMES = 10  # Process every 10th frame (adjust based on your CPU)
    MAX_QUEUE_SIZE = 5
    DISPLAY_FPS = 30
    RECORDING_FPS = 20
    
    # Recording Settings
    OUTPUT_DIR = "recordings"
    RECORDING_DELAY = 4.0  # seconds
    VIDEO_CODEC = 'mp4v'
    
    # Stream Settings
    STREAM_BUFFER_SIZE = 1
    RECONNECT_DELAY = 5.0  # seconds
    
    # Display Settings
    WINDOW_NAME = "ESP32-CAM Human Detection"
    BBOX_COLOR = (0, 255, 0)  # Green
    BBOX_THICKNESS = 2
    TEXT_COLOR = (0, 255, 0)  # Green
    TEXT_SCALE = 0.7
    TEXT_THICKNESS = 2
    
    @classmethod
    def load_from_env(cls):
        """Load configuration from environment variables"""
        cls.ESP32_URL = os.getenv('ESP32_URL', cls.ESP32_URL)
        cls.MODEL_PATH = os.getenv('MODEL_PATH', cls.MODEL_PATH)
        cls.OUTPUT_DIR = os.getenv('OUTPUT_DIR', cls.OUTPUT_DIR)
        cls.CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', cls.CONFIDENCE_THRESHOLD))
        cls.SKIP_FRAMES = int(os.getenv('SKIP_FRAMES', cls.SKIP_FRAMES))
        cls.RECORDING_DELAY = float(os.getenv('RECORDING_DELAY', cls.RECORDING_DELAY))
