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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ESP32HumanDetectionSystem:
    def __init__(self, esp32_url="http://192.168.145.152:81/stream", 
                 model_path="yolo8n.pt", output_dir="recordings"):
        # Configuration
        self.esp32_url = esp32_url
        self.model_path = model_path
        self.output_dir = output_dir
        
        # Detection settings
        self.SKIP_FRAMES = 3
        self.CONFIDENCE_THRESHOLD = 0.5
        self.PERSON_CLASS_ID = 0
        
        # Recording settings
        self.RECORDING_DELAY = 4.0  # 4 seconds delay before stopping recording
        
        # Thread control
        self.running = False
        self.threads = []
        
        # Queues for thread communication
        self.frame_queue = queue.Queue(maxsize=5)
        self.detection_queue = queue.Queue(maxsize=5)
        self.display_queue = queue.Queue(maxsize=5)
        
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
        
        # FPS calculation
        self.fps_queue = deque(maxlen=30)
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
            # Filter to only detect persons (class 0)
            self.model.overrides['classes'] = [0]  # Only person class
            logger.info(f"YOLO model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def _connect_to_stream(self):
        """Connect to ESP32-CAM stream with retry logic"""
        while self.running:
            try:
                cap = cv2.VideoCapture(self.esp32_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                if cap.isOpened():
                    logger.info("Connected to ESP32-CAM stream")
                    return cap
                else:
                    logger.warning("Failed to connect to ESP32-CAM")
                    
            except Exception as e:
                logger.error(f"Stream connection error: {e}")
            
            time.sleep(5)  # Wait before retry
        return None
    
    def _stream_thread(self):
        """Thread 1: Capture frames from ESP32-CAM stream"""
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
                
                # Put frame in queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                
                # Update current frame for display
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                time.sleep(0.01)  # Small delay to prevent CPU overload
                
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
                    
                    # Frame skipping logic
                    self.frame_counter += 1
                    if self.frame_counter % self.SKIP_FRAMES != 0:
                        continue
                    
                    # Run YOLO detection
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
                    
                    # Update detection results
                    with self.detection_lock:
                        self.current_detections = detections
                        self.human_count = human_count
                    
                    # Handle recording logic
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
                
                # Start recording if not already recording
                if not self.is_recording:
                    self._start_recording()
                
                # Cancel stop timer if it exists
                if self.recording_stop_timer:
                    self.recording_stop_timer.cancel()
                    self.recording_stop_timer = None
            
            else:  # No human detected
                if self.is_recording and self.last_human_detection_time:
                    # Check if we should start the stop timer
                    time_since_last_detection = current_time - self.last_human_detection_time
                    
                    if time_since_last_detection >= 0.5 and not self.recording_stop_timer:
                        # Start the 4-second countdown
                        self.recording_stop_timer = threading.Timer(
                            self.RECORDING_DELAY, self._stop_recording
                        )
                        self.recording_stop_timer.start()
                        logger.info("Started 4-second countdown to stop recording")
    
    def _start_recording(self):
        """Start video recording"""
        if self.is_recording:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"record_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        # Get frame dimensions
        with self.frame_lock:
            if self.current_frame is not None:
                height, width = self.current_frame.shape[:2]
            else:
                width, height = 640, 480  # Default
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filepath, fourcc, 20.0, (width, height))
        
        if self.video_writer.isOpened():
            self.is_recording = True
            logger.info(f"Started recording: {filename}")
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
                logger.info("Recording stopped and saved")
    
    def _recording_thread(self):
        """Thread 4: Handle video recording"""
        logger.info("Recording thread started")
        
        while self.running:
            try:
                with self.recording_lock:
                    if self.is_recording and self.video_writer:
                        with self.frame_lock:
                            if self.current_frame is not None:
                                # Write frame with detections drawn
                                display_frame = self._draw_detections(self.current_frame.copy())
                                self.video_writer.write(display_frame)
                
                time.sleep(0.05)  # 20 FPS for recording
                
            except Exception as e:
                logger.error(f"Recording thread error: {e}")
                time.sleep(0.1)
        
        # Cleanup recording on exit
        with self.recording_lock:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
        
        logger.info("Recording thread stopped")
    
    def _draw_detections(self, frame):
        """Draw bounding boxes and information on frame"""
        with self.detection_lock:
            detections = self.current_detections.copy()
            human_count = self.human_count
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw green rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence text
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw info overlay
        info_y = 30
        cv2.putText(frame, f"Humans: {human_count}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, info_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Recording indicator
        if self.is_recording:
            cv2.putText(frame, "REC", (10, info_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.circle(frame, (50, info_y + 55), 5, (0, 0, 255), -1)
        
        return frame
    
    def _display_thread(self):
        """Thread 3: Display video with detections"""
        logger.info("Display thread started")
        
        while self.running:
            try:
                with self.frame_lock:
                    if self.current_frame is not None:
                        display_frame = self._draw_detections(self.current_frame.copy())
                        
                        # Calculate FPS
                        current_time = time.time()
                        if self.last_time > 0:
                            frame_time = current_time - self.last_time
                            if frame_time > 0:
                                self.fps_queue.append(1.0 / frame_time)
                                self.fps = sum(self.fps_queue) / len(self.fps_queue)
                        self.last_time = current_time
                        
                        # Display frame
                        cv2.imshow('ESP32-CAM Human Detection', display_frame)
                        
                        # Handle key press
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            logger.info("User requested quit")
                            self.stop()
                            break
                        elif key == ord('s'):
                            # Manual snapshot
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"snapshot_{timestamp}.jpg"
                            filepath = os.path.join(self.output_dir, filename)
                            cv2.imwrite(filepath, display_frame)
                            logger.info(f"Snapshot saved: {filename}")
                
                time.sleep(0.03)  # ~30 FPS display
                
            except Exception as e:
                logger.error(f"Display thread error: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
        logger.info("Display thread stopped")
    
    def start(self):
        """Start the detection system"""
        if self.running:
            logger.warning("System is already running")
            return
        
        self.running = True
        logger.info("Starting ESP32-CAM Human Detection System")
        
        # Start all threads
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
        if not self.running:
            return
        
        logger.info("Stopping system...")
        self.running = False
        
        # Cancel recording timer if active
        with self.recording_lock:
            if self.recording_stop_timer:
                self.recording_stop_timer.cancel()
            self._stop_recording()
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2.0)
            logger.info(f"Stopped {thread.name}")
        
        logger.info("System stopped successfully")
    
    def run(self):
        """Run the system (blocking call)"""
        try:
            self.start()
            
            # Keep main thread alive
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.stop()

def main():
    # Configuration
    ESP32_URL = "http://192.168.145.152:81/stream"  # Change to your ESP32-CAM IP
    MODEL_PATH = "yolo8n.pt"  # YOLO model path
    OUTPUT_DIR = "recordings"
    
    print("=" * 60)
    print("ESP32-CAM Human Detection System")
    print("=" * 60)
    print(f"ESP32-CAM URL: {ESP32_URL}")
    print(f"Model: {MODEL_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 60)
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' for manual snapshot")
    print("- Recording starts automatically when human detected")
    print("- Recording stops 4 seconds after last human detection")
    print("=" * 60)
    
    try:
        # Create and run the detection system
        detector = ESP32HumanDetectionSystem(
            esp32_url=ESP32_URL,
            model_path=MODEL_PATH,
            output_dir=OUTPUT_DIR
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