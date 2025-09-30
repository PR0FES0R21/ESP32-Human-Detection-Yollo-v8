import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
from ultralytics import YOLO
import logging
from datetime import datetime
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ESP32HumanDetector:
    def __init__(self, 
                 stream_url="http://192.168.145.152:81/stream",  # Ganti dengan IP ESP32-CAM Anda
                 model_path="yolov8n.pt",
                 skip_frames=1,
                 confidence_threshold=0.5,
                 enable_snapshots=False,
                 snapshot_dir="snapshots"):
        
        # Configuration
        self.stream_url = stream_url
        self.model_path = model_path
        self.skip_frames = skip_frames
        self.confidence_threshold = confidence_threshold
        self.enable_snapshots = enable_snapshots
        self.snapshot_dir = snapshot_dir
        
        # Threading components
        self.frame_queue = Queue(maxsize=5)  # Buffer untuk frame
        self.detection_queue = Queue(maxsize=5)  # Buffer untuk hasil deteksi
        self.latest_frame = None
        self.latest_detections = []
        self.human_count = 0
        
        # Control flags
        self.running = False
        self.stream_connected = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.frame_counter = 0
        
        # Locks untuk thread safety
        self.frame_lock = threading.Lock()
        self.detection_lock = threading.Lock()
        
        # Initialize model
        self.model = None
        self.init_model()
        
        # Create snapshot directory
        if self.enable_snapshots and not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)
    
    def init_model(self):
        """Initialize YOLOv8n model"""
        try:
            logger.info("Loading YOLOv8n model...")
            self.model = YOLO(self.model_path)
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def export_to_onnx(self, onnx_path="yolov8n.onnx"):
        """Export model to ONNX for better performance"""
        try:
            logger.info("Exporting model to ONNX...")
            self.model.export(format="onnx", optimize=True)
            logger.info(f"Model exported to {onnx_path}")
        except Exception as e:
            logger.error(f"Failed to export to ONNX: {e}")
    
    def stream_reader_thread(self):
        """Thread 1: Membaca stream dari ESP32-CAM"""
        cap = None
        reconnect_delay = 5
        
        while self.running:
            try:
                if cap is None or not cap.isOpened():
                    logger.info(f"Connecting to stream: {self.stream_url}")
                    cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
                    
                    if not cap.isOpened():
                        logger.warning(f"Failed to connect. Retrying in {reconnect_delay}s...")
                        time.sleep(reconnect_delay)
                        continue
                    
                    self.stream_connected = True
                    logger.info("Stream connected successfully!")
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame. Reconnecting...")
                    self.stream_connected = False
                    cap.release()
                    cap = None
                    time.sleep(reconnect_delay)
                    continue
                
                # Resize frame untuk optimasi (opsional)
                # frame = cv2.resize(frame, (640, 480))
                
                # Update latest frame
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
                # Add to queue for processing (non-blocking)
                try:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame, timeout=0.001)
                except:
                    pass  # Skip jika queue penuh
                
                time.sleep(0.01)  # Small delay to prevent CPU overload
                
            except Exception as e:
                logger.error(f"Stream reader error: {e}")
                self.stream_connected = False
                if cap:
                    cap.release()
                    cap = None
                time.sleep(reconnect_delay)
        
        if cap:
            cap.release()
    
    def detection_thread(self):
        """Thread 2: Jalankan YOLOv8n untuk deteksi manusia"""
        frame_skip_counter = 0
        
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1.0)
                frame_skip_counter += 1
                
                # Skip frames untuk optimasi
                if frame_skip_counter % self.skip_frames != 0:
                    continue
                
                # Run detection
                results = self.model(frame, classes=[0], verbose=False)  # class 0 = person
                
                detections = []
                human_count = 0
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            confidence = box.conf[0].item()
                            if confidence >= self.confidence_threshold:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                detections.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'confidence': confidence
                                })
                                human_count += 1
                
                # Update detection results
                with self.detection_lock:
                    self.latest_detections = detections
                    self.human_count = human_count
                
                # Save snapshot if human detected
                if self.enable_snapshots and human_count > 0:
                    self.save_snapshot(frame)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Detection thread error: {e}")
                time.sleep(0.1)
    
    def display_thread(self):
        """Thread 3: Tampilkan hasil ke layar"""
        window_name = "ESP32-CAM Human Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        while self.running:
            try:
                with self.frame_lock:
                    if self.latest_frame is None:
                        time.sleep(0.01)
                        continue
                    display_frame = self.latest_frame.copy()
                
                # Draw detections
                with self.detection_lock:
                    detections = self.latest_detections.copy()
                    human_count = self.human_count
                
                # Draw bounding boxes
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    confidence = detection['confidence']
                    
                    # Green bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Confidence label
                    label = f"Person: {confidence:.2f}"
                    cv2.putText(display_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw info overlay
                self.draw_info_overlay(display_frame, human_count)
                
                # Show frame
                cv2.imshow(window_name, display_frame)
                
                # Update FPS counter
                self.update_fps()
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop()
                    break
                elif key == ord('s'):
                    # Save current frame
                    cv2.imwrite(f"manual_snapshot_{int(time.time())}.jpg", display_frame)
                    logger.info("Manual snapshot saved")
                elif key == ord('+'):
                    # Decrease skip frames (more frequent detection)
                    self.skip_frames = max(1, self.skip_frames - 1)
                    logger.info(f"Skip frames: {self.skip_frames}")
                elif key == ord('-'):
                    # Increase skip frames (less frequent detection)
                    self.skip_frames = min(30, self.skip_frames + 1)
                    logger.info(f"Skip frames: {self.skip_frames}")
                
            except Exception as e:
                logger.error(f"Display thread error: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
    
    def draw_info_overlay(self, frame, human_count):
        """Draw information overlay on frame"""
        h, w = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Connection status
        status_color = (0, 255, 0) if self.stream_connected else (0, 0, 255)
        status_text = "CONNECTED" if self.stream_connected else "DISCONNECTED"
        cv2.putText(frame, f"Status: {status_text}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Human count
        cv2.putText(frame, f"Humans: {human_count}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Skip frames info
        cv2.putText(frame, f"Skip: {self.skip_frames}", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Controls info
        controls_text = [
            "Controls:",
            "Q: Quit",
            "S: Save snapshot",
            "+/-: Adjust skip frames"
        ]
        
        for i, text in enumerate(controls_text):
            cv2.putText(frame, text, (w-200, 30 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def save_snapshot(self, frame):
        """Save snapshot when human detected"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.snapshot_dir}/human_detected_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        logger.info(f"Snapshot saved: {filename}")
    
    def start(self):
        """Start the detection system"""
        logger.info("Starting ESP32-CAM Human Detection System...")
        self.running = True
        
        # Start threads
        self.stream_thread = threading.Thread(target=self.stream_reader_thread, daemon=True)
        self.detect_thread = threading.Thread(target=self.detection_thread, daemon=True)
        self.display_thread_obj = threading.Thread(target=self.display_thread, daemon=True)
        
        self.stream_thread.start()
        self.detect_thread.start()
        self.display_thread_obj.start()
        
        logger.info("All threads started successfully!")
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.stop()
    
    def stop(self):
        """Stop the detection system"""
        logger.info("Stopping system...")
        self.running = False
        
        # Wait for threads to finish
        if hasattr(self, 'stream_thread'):
            self.stream_thread.join(timeout=2)
        if hasattr(self, 'detect_thread'):
            self.detect_thread.join(timeout=2)
        if hasattr(self, 'display_thread_obj'):
            self.display_thread_obj.join(timeout=2)
        
        logger.info("System stopped")

def main():
    """Main function"""
    # Konfigurasi sistem
    config = {
        'stream_url': "http://192.168.145.152:81/stream",  # Ganti dengan IP ESP32-CAM Anda
        'model_path': "yolov8n.pt",  # Model akan didownload otomatis jika belum ada
        'skip_frames': 1,  # Proses setiap 10 frame
        'confidence_threshold': 0.5,  # Threshold confidence
        'enable_snapshots': True,  # Aktifkan snapshot otomatis
        'snapshot_dir': "snapshots"  # Direktori untuk menyimpan snapshot
    }
    
    try:
        # Initialize detector
        detector = ESP32HumanDetector(**config)
        
        # Optional: Export to ONNX for better performance
        # detector.export_to_onnx()
        
        # Start detection system
        detector.start()
        
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Program error: {e}")
    finally:
        logger.info("Program finished")

if __name__ == "__main__":
    main()