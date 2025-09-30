# optimized_detector.py - Enhanced version with ONNX support and better performance

import cv2
import numpy as np
import threading
import time
import queue
from datetime import datetime
from collections import deque
import os
import logging
from pathlib import Path

# Try to import ONNX runtime for optimized inference
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Import YOLO
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
__all__ = ["OptimizedHumanDetector", "main"]

class OptimizedHumanDetector:
    def __init__(self, esp32_url="http://192.168.20.152:81/stream", 
                 model_path="yolov8n.pt", output_dir="recordings",
                 use_onnx=True, input_size=(640, 640)):
        
        # Core settings
        self.esp32_url = esp32_url
        self.model_path = model_path
        self.output_dir = output_dir
        self.use_onnx = use_onnx and ONNX_AVAILABLE
        self.input_size = input_size
        
        # Performance settings
        self.SKIP_FRAMES = 8  # Optimized for better balance
        self.CONFIDENCE_THRESHOLD = 0.4  # Slightly lower for better detection
        self.NMS_THRESHOLD = 0.5
        self.PERSON_CLASS_ID = 0
        
        # Recording settings
        self.RECORDING_DELAY = 4.0
        self.RECORDING_BUFFER_SIZE = 60  # frames
        
        # Thread control
        self.running = False
        self.threads = []
        
        # Thread-safe queues
        self.frame_queue = queue.Queue(maxsize=3)  # Smaller queue for lower latency
        self.detection_queue = queue.Queue(maxsize=3)
        
        # Shared data with locks
        self.current_frame = None
        self.current_detections = []
        self.human_count = 0
        self.fps = 0.0
        self.inference_time = 0.0
        
        # Thread locks
        self.frame_lock = threading.Lock()
        self.detection_lock = threading.Lock()
        self.recording_lock = threading.Lock()
        
        # Recording state
        self.is_recording = False
        self.video_writer = None
        self.last_human_detection_time = None
        self.recording_stop_timer = None
        self.recording_buffer = deque(maxlen=self.RECORDING_BUFFER_SIZE)
        
        # Performance monitoring
        self.fps_queue = deque(maxlen=30)
        self.inference_times = deque(maxlen=30)
        self.last_time = time.time()
        self.frame_counter = 0
        
        # Initialize
        self._setup_directories()
        self._load_model()
        
    def _setup_directories(self):
        """Setup output directories"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.output_dir, "snapshots")).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def _load_model(self):
        """Load and optimize YOLO model"""
        try:
            # Load YOLO model
            self.yolo_model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded: {self.model_path}")
            
            # Try to export to ONNX for optimization
            if self.use_onnx:
                onnx_path = self.model_path.replace('.pt', '.onnx')
                
                if not os.path.exists(onnx_path):
                    logger.info("Exporting model to ONNX format...")
                    self.yolo_model.export(format='onnx', imgsz=self.input_size)
                
                if os.path.exists(onnx_path):
                    # Load ONNX model
                    providers = ['CPUExecutionProvider']
                    if ort.get_device() == 'GPU':
                        providers.insert(0, 'CUDAExecutionProvider')
                    
                    self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
                    self.model_input_name = self.onnx_session.get_inputs()[0].name
                    self.model_outputs = [output.name for output in self.onnx_session.get_outputs()]
                    
                    logger.info(f"ONNX model loaded with providers: {providers}")
                    self.use_onnx = True
                else:
                    logger.warning("ONNX export failed, using PyTorch model")
                    self.use_onnx = False
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            self.use_onnx = False
    
    def _preprocess_frame(self, frame):
        """Preprocess frame for ONNX inference"""
        # Resize frame
        input_frame = cv2.resize(frame, self.input_size)
        
        # Convert BGR to RGB
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        
        # Normalize
        input_frame = input_frame.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose
        input_frame = np.transpose(input_frame, (2, 0, 1))  # HWC to CHW
        input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension
        
        return input_frame
    
    def _postprocess_detections(self, outputs, frame_shape):
        """Post-process ONNX model outputs"""
        detections = []
        
        if len(outputs) > 0:
            predictions = outputs[0]  # Get first output
            
            # Handle different output shapes
            if len(predictions.shape) == 3:
                predictions = predictions[0]  # Remove batch dimension
            
            # Transpose if needed (some models output different shapes)
            if predictions.shape[0] == 84:  # 84 = 4 (bbox) + 80 (classes)
                predictions = predictions.T
            
            # Extract relevant information
            boxes = predictions[:, :4]  # x, y, w, h
            scores = predictions[:, 4]  # confidence scores
            class_probs = predictions[:, 5:]  # class probabilities
            
            # Get class predictions
            class_ids = np.argmax(class_probs, axis=1)
            class_scores = np.max(class_probs, axis=1)
            
            # Calculate final confidence scores
            confidences = scores * class_scores
            
            # Filter for person class and confidence threshold
            person_mask = (class_ids == self.PERSON_CLASS_ID) & (confidences > self.CONFIDENCE_THRESHOLD)
            
            if np.any(person_mask):
                # Get filtered detections
                filtered_boxes = boxes[person_mask]
                filtered_confidences = confidences[person_mask]
                
                # Convert from center format to corner format and scale
                h_orig, w_orig = frame_shape[:2]
                h_input, w_input = self.input_size
                
                scale_x = w_orig / w_input
                scale_y = h_orig / h_input
                
                for i, (box, conf) in enumerate(zip(filtered_boxes, filtered_confidences)):
                    cx, cy, w, h = box
                    
                    # Convert to corner coordinates and scale
                    x1 = int((cx - w/2) * scale_x)
                    y1 = int((cy - h/2) * scale_y)
                    x2 = int((cx + w/2) * scale_x)
                    y2 = int((cy + h/2) * scale_y)
                    
                    # Clamp to frame boundaries
                    x1 = max(0, min(x1, w_orig))
                    y1 = max(0, min(y1, h_orig))
                    x2 = max(0, min(x2, w_orig))
                    y2 = max(0, min(y2, h_orig))
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(conf)
                    })
        
        return detections
    
    def _run_inference(self, frame):
        """Run inference using ONNX or YOLO"""
        start_time = time.time()
        
        try:
            if self.use_onnx:
                # ONNX inference
                input_tensor = self._preprocess_frame(frame)
                outputs = self.onnx_session.run(self.model_outputs, {self.model_input_name: input_tensor})
                detections = self._postprocess_detections(outputs, frame.shape)
            else:
                # YOLO inference
                results = self.yolo_model(frame, verbose=False, classes=[self.PERSON_CLASS_ID])
                detections = []
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            confidence = float(box.conf)
                            if confidence > self.CONFIDENCE_THRESHOLD:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                detections.append({
                                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                    'confidence': confidence
                                })
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.inference_time = np.mean(self.inference_times)
            
            return detections
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return []
    
    def _connect_to_stream(self):
        """Enhanced stream connection with better error handling"""
        retry_count = 0
        max_retries = 10
        
        while self.running and retry_count < max_retries:
            try:
                # Try different backends for better compatibility
                backends = [cv2.CAP_ANY, cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER]
                
                for backend in backends:
                    cap = cv2.VideoCapture(self.esp32_url, backend)
                    
                    # Optimize capture settings
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    if cap.isOpened():
                        # Test if we can actually read a frame
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            logger.info(f"Connected to ESP32-CAM using backend: {backend}")
                            return cap
                        else:
                            cap.release()
                
                retry_count += 1
                logger.warning(f"Connection attempt {retry_count} failed, retrying in 3 seconds...")
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"Stream connection error: {e}")
                retry_count += 1
                time.sleep(3)
        
        logger.error("Failed to connect to ESP32-CAM after multiple attempts")
        return None
    
    def _stream_thread(self):
        """Enhanced stream capture thread"""
        logger.info("Stream thread started")
        cap = None
        consecutive_failures = 0
        
        while self.running:
            try:
                if cap is None or not cap.isOpened():
                    cap = self._connect_to_stream()
                    if cap is None:
                        time.sleep(5)
                        continue
                    consecutive_failures = 0
                
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        logger.warning("Too many consecutive failures, reconnecting...")
                        cap.release()
                        cap = None
                        consecutive_failures = 0
                    continue
                
                consecutive_failures = 0
                
                # Update current frame
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Add to recording buffer
                with self.recording_lock:
                    self.recording_buffer.append(frame.copy())
                
                # Add to processing queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass  # Skip frame if queue is full
                
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
        """Enhanced detection thread with better performance"""
        logger.info("Detection thread started")
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                
                # Frame skipping logic
                self.frame_counter += 1
                if self.frame_counter % self.SKIP_FRAMES != 0:
                    continue
                
                # Run inference
                detections = self._run_inference(frame)
                human_count = len(detections)
                
                # Update detection results
                with self.detection_lock:
                    self.current_detections = detections
                    self.human_count = human_count
                
                # Handle recording logic
                self._handle_recording_logic(human_count > 0)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Detection thread error: {e}")
                time.sleep(0.1)
        
        logger.info("Detection thread stopped")
    
    def _handle_recording_logic(self, human_detected):
        """Enhanced recording logic with buffering"""
        with self.recording_lock:
            current_time = time.time()
            
            if human_detected:
                self.last_human_detection_time = current_time
                
                if not self.is_recording:
                    self._start_recording()
                
                # Cancel stop timer
                if self.recording_stop_timer:
                    self.recording_stop_timer.cancel()
                    self.recording_stop_timer = None
            
            else:  # No human detected
                if self.is_recording and self.last_human_detection_time:
                    time_since_last = current_time - self.last_human_detection_time
                    
                    if time_since_last >= 0.5 and not self.recording_stop_timer:
                        self.recording_stop_timer = threading.Timer(
                            self.RECORDING_DELAY, self._stop_recording
                        )
                        self.recording_stop_timer.start()
                        logger.info(f"Started {self.RECORDING_DELAY}s countdown to stop recording")
    
    def _start_recording(self):
        """Enhanced recording start with pre-buffering"""
        if self.is_recording:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"human_detected_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        # Get frame dimensions from buffer
        if self.recording_buffer:
            height, width = self.recording_buffer[-1].shape[:2]
        else:
            width, height = 640, 480
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filepath, fourcc, 20.0, (width, height))
        
        if self.video_writer.isOpened():
            # Write buffered frames (pre-roll)
            for buffered_frame in self.recording_buffer:
                annotated_frame = self._draw_detections(buffered_frame.copy())
                self.video_writer.write(annotated_frame)
            
            self.is_recording = True
            logger.info(f"Started recording with {len(self.recording_buffer)} pre-buffered frames: {filename}")
        else:
            logger.error("Failed to start video recording")
            self.video_writer = None
    
    def _stop_recording(self):
        """Enhanced recording stop"""
        with self.recording_lock:
            if self.is_recording and self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                self.is_recording = False
                self.recording_stop_timer = None
                logger.info("Recording stopped and saved")
    
    def _recording_thread(self):
        """Enhanced recording thread"""
        logger.info("Recording thread started")
        
        while self.running:
            try:
                with self.recording_lock:
                    if self.is_recording and self.video_writer:
                        with self.frame_lock:
                            if self.current_frame is not None:
                                annotated_frame = self._draw_detections(self.current_frame.copy())
                                self.video_writer.write(annotated_frame)
                
                time.sleep(0.05)  # 20 FPS recording
                
            except Exception as e:
                logger.error(f"Recording thread error: {e}")
                time.sleep(0.1)
        
        # Cleanup
        with self.recording_lock:
            if self.video_writer:
                self.video_writer.release()
        
        logger.info("Recording thread stopped")
    
    def _draw_detections(self, frame):
        """Enhanced visualization with better info display"""
        with self.detection_lock:
            detections = self.current_detections.copy()
            human_count = self.human_count
        
        # Draw bounding boxes
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence and person ID
            label = f"Person {i+1}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Enhanced info overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        info_lines = [
            f"Humans Detected: {human_count}",
            f"FPS: {self.fps:.1f}",
            f"Inference Time: {self.inference_time*1000:.1f}ms",
            f"Model: {'ONNX' if self.use_onnx else 'PyTorch'}",
            f"Recording: {'ON' if self.is_recording else 'OFF'}"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(frame, line, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Recording indicator
        if self.is_recording:
            cv2.circle(frame, (370, 30), 8, (0, 0, 255), -1)
        
        return frame
    
    def _display_thread(self):
        """Enhanced display thread with better FPS calculation"""
        logger.info("Display thread started")
        
        while self.running:
            try:
                with self.frame_lock:
                    if self.current_frame is not None:
                        display_frame = self._draw_detections(self.current_frame.copy())
                        
                        # Calculate and update FPS
                        current_time = time.time()
                        if self.last_time > 0:
                            frame_time = current_time - self.last_time
                            if frame_time > 0:
                                self.fps_queue.append(1.0 / frame_time)
                                self.fps = np.mean(self.fps_queue)
                        self.last_time = current_time
                        
                        # Display frame
                        cv2.imshow('ESP32-CAM Human Detection (Optimized)', display_frame)
                        
                        # Handle keyboard input
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            logger.info("User requested quit")
                            self.stop()
                            break
                        elif key == ord('s'):
                            self._save_snapshot(display_frame)
                        elif key == ord('r'):
                            self._toggle_recording()
                        elif key == ord('c'):
                            self._clear_recordings()
                
                time.sleep(0.03)  # ~30 FPS display
                
            except Exception as e:
                logger.error(f"Display thread error: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
        logger.info("Display thread stopped")
    
    def _save_snapshot(self, frame):
        """Save manual snapshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, "snapshots", filename)
        cv2.imwrite(filepath, frame)
        logger.info(f"Snapshot saved: {filename}")
    
    def _toggle_recording(self):
        """Manually toggle recording"""
        with self.recording_lock:
            if self.is_recording:
                self._stop_recording()
                logger.info("Recording stopped manually")
            else:
                self._start_recording()
                logger.info("Recording started manually")
    
    def _clear_recordings(self):
        """Clear old recordings (keep last 10)"""
        try:
            recordings = sorted([f for f in os.listdir(self.output_dir) 
                               if f.endswith('.mp4')], reverse=True)
            
            if len(recordings) > 10:
                for old_file in recordings[10:]:
                    os.remove(os.path.join(self.output_dir, old_file))
                    logger.info(f"Deleted old recording: {old_file}")
                
                logger.info(f"Cleared {len(recordings) - 10} old recordings")
            else:
                logger.info("No old recordings to clear")
                
        except Exception as e:
            logger.error(f"Error clearing recordings: {e}")
    
    def start(self):
        """Start the optimized detection system"""
        if self.running:
            logger.warning("System is already running")
            return
        
        self.running = True
        logger.info("Starting Optimized ESP32-CAM Human Detection System")
        logger.info(f"Using {'ONNX' if self.use_onnx else 'PyTorch'} inference")
        
        # Start all threads
        self.threads = [
            threading.Thread(target=self._stream_thread, name="StreamThread", daemon=True),
            threading.Thread(target=self._detection_thread, name="DetectionThread", daemon=True),
            threading.Thread(target=self._display_thread, name="DisplayThread", daemon=True),
            threading.Thread(target=self._recording_thread, name="RecordingThread", daemon=True)
        ]
        
        for thread in self.threads:
            thread.start()
            logger.info(f"Started {thread.name}")
        
        logger.info("All threads started successfully")
    
    def stop(self):
        """Stop the detection system"""
        if not self.running:
            return
        
        logger.info("Stopping system...")
        self.running = False
        
        # Cancel timers
        with self.recording_lock:
            if self.recording_stop_timer:
                self.recording_stop_timer.cancel()
            self._stop_recording()
        
        # Wait for threads
        for thread in self.threads:
            thread.join(timeout=3.0)
        
        logger.info("System stopped successfully")
    
    def run(self):
        """Run the system"""
        try:
            self.start()
            
            print("\n" + "="*60)
            print("ESP32-CAM Human Detection System (Optimized)")
            print("="*60)
            print("Controls:")
            print("  'q' - Quit")
            print("  's' - Save snapshot")
            print("  'r' - Toggle recording")
            print("  'c' - Clear old recordings")
            print("="*60)
            
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ESP32-CAM Human Detection System')
    parser.add_argument('--url', default='http://192.168.20.152:81/stream',
                       help='ESP32-CAM stream URL')
    parser.add_argument('--model', default='yolov8n.pt',
                       help='YOLO model path')
    parser.add_argument('--output', default='recordings',
                       help='Output directory')
    parser.add_argument('--no-onnx', action='store_true',
                       help='Disable ONNX optimization')
    parser.add_argument('--skip-frames', type=int, default=8,
                       help='Number of frames to skip between detections')
    
    args = parser.parse_args()
    
    try:
        detector = OptimizedHumanDetector(
            esp32_url=args.url,
            model_path=args.model,
            output_dir=args.output,
            use_onnx=not args.no_onnx
        )
        
        detector.SKIP_FRAMES = args.skip_frames
        detector.run()
        
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()