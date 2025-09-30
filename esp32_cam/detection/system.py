import cv2
import numpy as np
import threading
import time
import queue
from datetime import datetime
from collections import deque
import os
from ultralytics import YOLO
import logging
import urllib.request
import urllib.parse

from typing import Optional

from ..config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
__all__ = ["ESP32HumanDetectionSystem", "main"]

class ESP32HumanDetectionSystem:
    def __init__(self,
                 esp32_url: Optional[str] = None,
                 model_path: Optional[str] = None,
                 output_dir: Optional[str] = None):
        Config.load_from_env()

        # Configuration
        self.esp32_url = esp32_url or Config.ESP32_URL
        self.model_path = model_path or Config.MODEL_PATH
        self.output_dir = output_dir or Config.OUTPUT_DIR

        # Detection settings
        self.SKIP_FRAMES = Config.SKIP_FRAMES
        self.CONFIDENCE_THRESHOLD = Config.CONFIDENCE_THRESHOLD
        self.PERSON_CLASS_ID = Config.PERSON_CLASS_ID

        # Recording settings
        self.RECORDING_DELAY = Config.RECORDING_DELAY

        # --- Pre-detection Buffer ---
        self.PRE_DETECTION_BUFFER_SECONDS = Config.PRE_DETECTION_BUFFER_SECONDS
        estimated_fps = Config.ESTIMATED_FPS_FOR_BUFFER or Config.RECORDING_FPS or 20.0
        self.ESTIMATED_FPS_FOR_BUFFER = estimated_fps
        buffer_size = max(1, int(self.PRE_DETECTION_BUFFER_SECONDS * self.ESTIMATED_FPS_FOR_BUFFER))
        self.pre_detection_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        logger.info(
            "Pre-detection buffer initialized for ~%.1f seconds (size: %d frames)",
            self.PRE_DETECTION_BUFFER_SECONDS,
            buffer_size,
        )
        # --- End pre-detection buffer ---

        # Camera control defaults
        self._default_camera_settings = Config.camera_settings()
        self._control_base_url = Config.derive_control_base_url(self.esp32_url)

        # Thread control
        self.running = False
        self.threads = []

        # Queues for thread communication
        self.frame_queue = queue.Queue(maxsize=Config.MAX_QUEUE_SIZE)

        # Shared variables (thread-safe)
        self.current_frame = None
        self.current_detections = []
        self.human_count = 0
        self.fps = 0
        self.frame_lock = threading.Lock()
        self.detection_lock = threading.Lock()

        # Recording variables
        self.is_recording = False
        self.video_writer = None
        self.last_human_detection_time = None
        self.recording_stop_timer = None
        self.recording_lock = threading.Lock()
        self.current_recording_path = None

        # FPS calculation
        fps_window = max(1, int(Config.DISPLAY_FPS))
        self.fps_queue = deque(maxlen=fps_window)
        self.last_time = time.time()

        # Frame counter for skipping
        self.frame_counter = 0

        # Initialize components
        self._setup_directories()
        self._load_model()
        
    def _setup_directories(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            self.model.overrides['classes'] = [0]
            logger.info(f"YOLO model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def _connect_to_stream(self):
        """Connect to ESP32-CAM stream with retry logic"""
        while self.running:
            try:
                cap = cv2.VideoCapture(self.esp32_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.STREAM_BUFFER_SIZE)
                cap.set(cv2.CAP_PROP_FPS, Config.DISPLAY_FPS)
                
                if cap.isOpened():
                    logger.info("Connected to ESP32-CAM stream")
                    self._configure_camera()
                    return cap
                else:
                    logger.warning("Failed to connect to ESP32-CAM")
                    
            except Exception as e:
                logger.error(f"Stream connection error: {e}")
            
            time.sleep(Config.RECONNECT_DELAY)
        return None
    
    def _derive_control_base_url(self):
        """Compute the base URL for the ESP32 control endpoint."""
        base = Config.derive_control_base_url(self.esp32_url)
        if not base:
            logger.warning("Unable to determine ESP32 control URL from %s", self.esp32_url)
        return base

    def _configure_camera(self):
        """Apply ESP32-CAM settings once a new stream connection is established."""
        if not self._default_camera_settings:
            logger.info("No camera settings defined to apply")
            return

        if not self._control_base_url:
            self._control_base_url = self._derive_control_base_url()

        if not self._control_base_url:
            logger.warning("Unable to determine ESP32 control URL; skipping camera configuration")
            return

        logger.info("Applying camera settings to ESP32-CAM")
        for var, val in self._default_camera_settings.items():
            if not self._control_base_url:
                break
            control_url = Config.compose_control_url(self._control_base_url, var, val)
            try:
                with urllib.request.urlopen(control_url, timeout=Config.CAMERA_REQUEST_TIMEOUT) as response:
                    status = getattr(response, 'status', None)
                    if status is None:
                        status = response.getcode()
                    if status != 200:
                        logger.warning(f"Camera setting {var}={val} returned HTTP {status}")
            except Exception as exc:
                logger.warning(f"Failed to set camera {var}={val}: {exc}")


    def get_status(self):
        """Return a snapshot of the system state for API consumers."""
        status = {
            'running': self.running,
            'is_recording': False,
            'humans_detected': 0,
            'fps': 0.0,
            'last_detection_time': None,
            'current_recording': None,
        }

        with self.detection_lock:
            status['humans_detected'] = self.human_count
            status['fps'] = float(self.fps)

        with self.recording_lock:
            status['is_recording'] = self.is_recording
            status['current_recording'] = self.current_recording_path
            if self.last_human_detection_time:
                status['last_detection_time'] = self.last_human_detection_time.isoformat()

        return status

    def get_latest_frame(self):
        """Return a copy of the most recent frame."""
        with self.frame_lock:
            if self.current_frame is None:
                return None
            return self.current_frame.copy()

    def list_recordings(self, limit=20):
        """Return recent recording files sorted by newest first."""
        try:
            files = [f for f in os.listdir(self.output_dir) if f.lower().endswith('.mp4')]
            files.sort(reverse=True)
            return files[:limit]
        except Exception as exc:
            logger.warning(f"Failed to list recordings: {exc}")
            return []

    def _stream_thread(self):
        """Thread 1: Capture frames and continuously fill the pre-detection buffer"""
        logger.info("Stream thread started")
        cap = None
        
        while self.running:
            try:
                if cap is None or not cap.isOpened():
                    cap = self._connect_to_stream()
                    if cap is None:
                        continue
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame, reconnecting...")
                    cap.release()
                    cap = None
                    continue
                
                frame_copy = frame.copy()

                # --- MODIFIKASI: Simpan setiap frame ke dalam pre-detection buffer ---
                with self.buffer_lock:
                    self.pre_detection_buffer.append(frame_copy)
                # --- AKHIR MODIFIKASI ---

                if not self.frame_queue.full():
                    self.frame_queue.put(frame_copy)
                
                with self.frame_lock:
                    self.current_frame = frame_copy
                
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Stream thread error: {e}")
                if cap:
                    cap.release()
                cap = None
                time.sleep(1)
        
        if cap:
            cap.release()
        logger.info("Stream thread stopped")
    
    def _detection_thread(self):
        """Thread 2: Process frames with YOLO detection"""
        logger.info("Detection thread started")
        
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    self.frame_counter += 1
                    if self.frame_counter % self.SKIP_FRAMES != 0:
                        continue
                    
                    results = self.model(frame, verbose=False)
                    
                    detections = []
                    human_count = 0
                    
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                confidence = float(box.conf)
                                class_id = int(box.cls)
                                
                                if class_id == self.PERSON_CLASS_ID and confidence > self.CONFIDENCE_THRESHOLD:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    detections.append({
                                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                        'confidence': confidence
                                    })
                                    human_count += 1
                    
                    with self.detection_lock:
                        self.current_detections = detections
                        self.human_count = human_count
                    
                    self._handle_recording_logic(human_count > 0)
                    
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Detection thread error: {e}")
                time.sleep(0.1)
        
        logger.info("Detection thread stopped")
    
    def _handle_recording_logic(self, human_detected):
        """Handle smart recording logic"""
        with self.recording_lock:
            current_time = time.time()
            
            if human_detected:
                self.last_human_detection_time = current_time
                
                if not self.is_recording:
                    self._start_recording()
                
                if self.recording_stop_timer:
                    self.recording_stop_timer.cancel()
                    self.recording_stop_timer = None
            
            else:
                if self.is_recording and self.last_human_detection_time:
                    time_since_last_detection = current_time - self.last_human_detection_time
                    
                    if time_since_last_detection >= 0.5 and not self.recording_stop_timer:
                        self.recording_stop_timer = threading.Timer(
                            self.RECORDING_DELAY, self._stop_recording
                        )
                        self.recording_stop_timer.start()
                        logger.info(f"Started {self.RECORDING_DELAY}-second countdown to stop recording")
    
    def _start_recording(self):
        """Start video recording, writing pre-detection buffer first"""
        if self.is_recording:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"record_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        with self.frame_lock:
            if self.current_frame is not None:
                height, width = self.current_frame.shape[:2]
            else:
                width, height = 640, 480
        
        fourcc = cv2.VideoWriter_fourcc(*Config.VIDEO_CODEC)
        # --- MODIFIKASI: Gunakan asumsi FPS untuk VideoWriter ---
        writer_fps = self.ESTIMATED_FPS_FOR_BUFFER or Config.RECORDING_FPS or 20.0
        self.video_writer = cv2.VideoWriter(filepath, fourcc, writer_fps, (width, height))
        
        if self.video_writer.isOpened():
            self.is_recording = True
            self.current_recording_path = filepath
            logger.info(f"Started recording: {filename}")

            # --- MODIFIKASI: Tulis frame dari buffer ke file video ---
            with self.buffer_lock:
                buffered_frames = list(self.pre_detection_buffer)
            
            logger.info(f"Writing {len(buffered_frames)} frames from pre-detection buffer...")
            for frame in buffered_frames:
                # Gambar deteksi di frame buffer jika diperlukan (opsional, bisa memperlambat)
                # Atau langsung tulis frame aslinya untuk kecepatan
                self.video_writer.write(frame)
            logger.info("Finished writing buffered frames.")
            # --- AKHIR MODIFIKASI ---

        else:
            logger.error("Failed to start video recording")
            self.video_writer = None
    
    def _stop_recording(self):
        """Stop video recording"""
        with self.recording_lock:
            if self.is_recording and self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                self.is_recording = False
                self.recording_stop_timer = None
                self.current_recording_path = None
                logger.info("Recording stopped and saved")
    
    def _recording_thread(self):
        """Thread 4: Handle live video recording (after buffer is written)"""
        logger.info("Recording thread started")
        
        while self.running:
            try:
                with self.recording_lock:
                    if self.is_recording and self.video_writer:
                        with self.frame_lock:
                            if self.current_frame is not None:
                                # Tulis frame live dengan deteksi yang sudah digambar
                                display_frame = self._draw_detections(self.current_frame.copy())
                                self.video_writer.write(display_frame)
                
                # --- MODIFIKASI: Sesuaikan sleep time dengan asumsi FPS ---
                time.sleep(1 / (self.ESTIMATED_FPS_FOR_BUFFER or Config.RECORDING_FPS or 20.0))
                
            except Exception as e:
                logger.error(f"Recording thread error: {e}")
                time.sleep(0.1)
        
        with self.recording_lock:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
        
        logger.info("Recording thread stopped")
    
    def _draw_detections(self, frame):
        """Draw bounding boxes and information on frame."""
        with self.detection_lock:
            detections = self.current_detections.copy()
            human_count = self.human_count

        bbox_color = Config.BBOX_COLOR
        bbox_thickness = Config.BBOX_THICKNESS
        text_color = Config.TEXT_COLOR
        text_scale = Config.TEXT_SCALE
        text_thickness = Config.TEXT_THICKNESS

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, bbox_thickness)
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_thickness)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), bbox_color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        info_y = 30
        cv2.putText(frame, f"Humans: {human_count}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, info_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness)

        if self.is_recording:
            cv2.putText(frame, "REC", (10, info_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), text_thickness)
            cv2.circle(frame, (50, info_y + 55), 5, (0, 0, 255), -1)

        return frame
    
    def _display_thread(self):
        """Thread 3: Display video with detections"""
        # (Tidak ada perubahan signifikan di fungsi ini)
        logger.info("Display thread started")
        
        while self.running:
            try:
                with self.frame_lock:
                    if self.current_frame is not None:
                        display_frame = self._draw_detections(self.current_frame.copy())
                        
                        current_time = time.time()
                        if self.last_time > 0:
                            frame_time = current_time - self.last_time
                            if frame_time > 0:
                                self.fps_queue.append(1.0 / frame_time)
                                self.fps = sum(self.fps_queue) / len(self.fps_queue)
                        self.last_time = current_time
                        
                        cv2.imshow(Config.WINDOW_NAME, display_frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            logger.info("User requested quit")
                            self.stop()
                            break
                        elif key == ord('s'):
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"snapshot_{timestamp}.jpg"
                            filepath = os.path.join(self.output_dir, filename)
                            cv2.imwrite(filepath, display_frame)
                            logger.info(f"Snapshot saved: {filename}")
                
                time.sleep(1.0 / Config.DISPLAY_FPS if Config.DISPLAY_FPS else 0.03)
                
            except Exception as e:
                logger.error(f"Display thread error: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
        logger.info("Display thread stopped")
    
    def start(self):
        """Start the detection system"""
        # (Tidak ada perubahan di fungsi ini)
        if self.running:
            logger.warning("System is already running")
            return
        
        self.running = True
        logger.info("Starting ESP32-CAM Human Detection System")
        
        self.threads = [
            threading.Thread(target=self._stream_thread, name="StreamThread"),
            threading.Thread(target=self._detection_thread, name="DetectionThread"),
            threading.Thread(target=self._display_thread, name="DisplayThread"),
            threading.Thread(target=self._recording_thread, name="RecordingThread")
        ]
        
        for thread in self.threads:
            thread.daemon = True
            thread.start()
            logger.info(f"Started {thread.name}")
        
        logger.info("All threads started successfully")
        logger.info("Press 'q' to quit, 's' for manual snapshot")
    
    def stop(self):
        """Stop the detection system"""
        # (Tidak ada perubahan di fungsi ini)
        if not self.running:
            return
        
        logger.info("Stopping system...")
        self.running = False
        
        with self.recording_lock:
            if self.recording_stop_timer:
                self.recording_stop_timer.cancel()
            self._stop_recording()
        
        for thread in self.threads:
            thread.join(timeout=2.0)
            logger.info(f"Stopped {thread.name}")
        
        logger.info("System stopped successfully")
    
    def run(self):
        """Run the system (blocking call)"""
        # (Tidak ada perubahan di fungsi ini)
        try:
            self.start()
            
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.stop()

def main():
    """CLI entry point for running the detection system."""
    Config.load_from_env()
    esp32_url = Config.ESP32_URL
    model_path = Config.MODEL_PATH
    output_dir = Config.OUTPUT_DIR

    print("=" * 60)
    print("ESP32-CAM Human Detection System (with Pre-Detection Buffer)")
    print("=" * 60)
    print(f"ESP32-CAM URL: {esp32_url}")
    print(f"Model: {model_path}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60)
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' for manual snapshot")
    print("- Recording starts automatically when human detected")
    print(f"- Recording stops {Config.RECORDING_DELAY} seconds after last human detection")
    print(f"- Video now includes ~{Config.PRE_DETECTION_BUFFER_SECONDS} seconds BEFORE detection.")
    print("=" * 60)

    try:
        detector = ESP32HumanDetectionSystem(
            esp32_url=esp32_url,
            model_path=model_path,
            output_dir=output_dir
        )
        detector.run()

    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Check ESP32-CAM IP address and stream URL")
        print("2. Ensure YOLO model file exists")
        print("3. Check network connectivity")
        print("4. Install required packages: pip install ultralytics opencv-python")


if __name__ == "__main__":
    main()
