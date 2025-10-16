import cv2
import time
import pyttsx3
from ultralytics import YOLO
from threading import Thread
import threading
from queue import Queue
import torch



# ============================================================================
# GPU AVAILABILITY CHECK
# ============================================================================
def check_gpu_availability():
    """Check if GPU is available and print device information"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        return True, 'cuda'
    else:
        return False, 'cpu'



GPU_AVAILABLE, DEVICE = check_gpu_availability()



# ============================================================================
# MODEL INITIALIZATION - PYTORCH ONLY (DUAL MODEL SUPPORT)
# ============================================================================
def load_models():
    """Load both regular object detection and pothole detection models"""
    # Load regular object detection model
    regular_model_name = "weights/yolov8n.pt"
    regular_model = YOLO(regular_model_name).to(DEVICE)

    # Load pothole detection model
    pothole_model_name = "weights/yolov8n_pothole.pt"
    pothole_model = YOLO(pothole_model_name).to(DEVICE)

    return regular_model, pothole_model


# Load both models
regular_model, pothole_model = load_models()



# ============================================================================
# PARAMETERS - OPTIMIZED FOR REAL-TIME PERFORMANCE
# ============================================================================
# Distance estimation parameters
FOCAL_LENGTH = 360
KNOWN_WIDTH = 60


# Performance parameters
DETECTION_INTERVAL = 0.033  # ~30 FPS detection rate (real-time)
SKIP_FRAMES = 2  # Process every 2nd frame for display smoothness
CONFIDENCE_THRESHOLD = 0.5  # Lowered for faster NMS
IOU_THRESHOLD = 0.7  # Higher IOU = less NMS work


# Frame preprocessing - optimized for speed
INPUT_WIDTH = 320  # Reduced from 640 for 4x speedup
INPUT_HEIGHT = 320
MAX_DETECTIONS = 10  # Limit detections for faster processing


# Speech parameters
SPEECH_INTERVAL = 2  # Speak every 2 seconds


# Load class names for regular objects
my_file = open("utils/coco.txt", "r")
data = my_file.read()
regular_class_list = data.split("\n")
my_file.close()

# Create combined class list (regular objects + pothole)
class_list = regular_class_list + ['pothole']

# Detection colors - special color for potholes (red)
detection_colors = [(0, 255, 0)] * len(regular_class_list) + [(0, 0, 255)]  # Green for regular, Red for pothole



# ============================================================================
# OPTIMIZED VIDEO CAPTURE CLASS
# ============================================================================
class VideoCapture:
    """Threaded video capture with optimized settings for real-time processing"""
    
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        
        # Optimize VideoCapture settings for minimal latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
        
        # For webcam sources, set FPS
        if isinstance(src, int):
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        
        # Larger queue for smoother processing
        self.queue = Queue(maxsize=8)
        
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        """Continuously read frames in background thread"""
        while not self.stopped:
            ret, frame = self.cap.read()
            
            if not ret:
                self.stop()
                return
            
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except:
                    pass
            
            self.queue.put((ret, frame))
                
    def read(self):
        """Get latest frame from queue"""
        if not self.queue.empty():
            return self.queue.get()
        return False, None
        
    def stop(self):
        """Stop the capture thread"""
        self.stopped = True
        self.cap.release()
        
    def is_stopped(self):
        """Check if capture is stopped"""
        return self.stopped



# ============================================================================
# ASYNC SPEECH CLASS (UNCHANGED - ALREADY OPTIMIZED)
# ============================================================================
class AsyncSpeech:
    """Non-blocking text-to-speech using threading"""
    
    def __init__(self):
        self.engine = pyttsx3.init()
        self.speech_queue = Queue()
        self.stopped = False
        
    def start(self):
        Thread(target=self._speak_worker, daemon=True).start()
        return self
        
    def _speak_worker(self):
        """Background worker for speech synthesis"""
        while not self.stopped:
            if not self.speech_queue.empty():
                text = self.speech_queue.get()
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except:
                    pass
            time.sleep(0.1)
                
    def speak(self, text):
        """Queue speech text (clears old speech)"""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except:
                break
        self.speech_queue.put(text)
        
    def stop(self):
        """Stop the speech worker"""
        self.stopped = True



# ============================================================================
# DUAL-MODEL YOLO DETECTOR (REGULAR OBJECTS + POTHOLES)
# ============================================================================
class DualYOLODetector:
    """
    Dual-model YOLOv8 detector supporting both regular objects and pothole detection
    - Runs both models simultaneously for comprehensive detection
    - Prioritizes pothole detection for safety
    - Optimized for real-time processing
    """

    def __init__(self, regular_model=None, pothole_model=None):
        self.regular_model = regular_model if regular_model is not None else regular_model
        self.pothole_model = pothole_model if pothole_model is not None else pothole_model

        # Get class names from both models
        self.regular_class_names = self.regular_model.names if hasattr(self.regular_model, 'names') else {}
        self.pothole_class_names = self.pothole_model.names if hasattr(self.pothole_model, 'names') else {}


    def detect(self, frame, confidence_threshold=0.5):
        """
        Dual-model detection: runs both regular object detection and pothole detection
        Returns ALL detections with potholes prioritized for safety
        """
        resized_frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))

        original_height, original_width = frame.shape[:2]
        scale_x = original_width / INPUT_WIDTH
        scale_y = original_height / INPUT_HEIGHT

        all_detections = []

        # Run regular object detection
        regular_results = self.regular_model(
            resized_frame,
            stream=True,
            verbose=False,
            conf=confidence_threshold,
            iou=IOU_THRESHOLD,
            max_det=MAX_DETECTIONS,
            device=DEVICE
        )

        for res in regular_results:
            if not res.boxes:
                continue

            for bbox in res.boxes:
                if bbox.conf >= confidence_threshold:
                    x1, y1, x2, y2 = bbox.xyxy[0].cpu().numpy()

                    scaled_x1 = int(x1 * scale_x)
                    scaled_y1 = int(y1 * scale_y)
                    scaled_x2 = int(x2 * scale_x)
                    scaled_y2 = int(y2 * scale_y)

                    scaled_area = (scaled_x2 - scaled_x1) * (scaled_y2 - scaled_y1)
                    class_id = int(bbox.cls)

                    detection = {
                        'bbox': [scaled_x1, scaled_y1, scaled_x2, scaled_y2],
                        'class_id': class_id,  # Regular object class ID
                        'class_name': res.names[class_id] if hasattr(res, 'names') else f"class_{class_id}",
                        'confidence': float(bbox.conf),
                        'area': scaled_area,
                        'is_largest': False,
                        'is_pothole': False,  # Mark as regular object
                        'detection_type': 'regular'
                    }

                    all_detections.append(detection)

        # Run pothole detection
        pothole_results = self.pothole_model(
            resized_frame,
            stream=True,
            verbose=False,
            conf=confidence_threshold,
            iou=IOU_THRESHOLD,
            max_det=MAX_DETECTIONS,
            device=DEVICE
        )

        for res in pothole_results:
            if not res.boxes:
                continue

            for bbox in res.boxes:
                if bbox.conf >= confidence_threshold:
                    x1, y1, x2, y2 = bbox.xyxy[0].cpu().numpy()

                    scaled_x1 = int(x1 * scale_x)
                    scaled_y1 = int(y1 * scale_y)
                    scaled_x2 = int(x2 * scale_x)
                    scaled_y2 = int(y2 * scale_y)

                    scaled_area = (scaled_x2 - scaled_x1) * (scaled_y2 - scaled_y1)
                    class_id = int(bbox.cls)

                    # Map pothole class ID to combined class list position
                    pothole_mapped_class_id = len(regular_class_list)  # Pothole is last in combined list

                    detection = {
                        'bbox': [scaled_x1, scaled_y1, scaled_x2, scaled_y2],
                        'class_id': pothole_mapped_class_id,
                        'class_name': 'pothole',
                        'confidence': float(bbox.conf),
                        'area': scaled_area,
                        'is_largest': False,
                        'is_pothole': True,  # Mark as pothole
                        'detection_type': 'pothole'
                    }

                    all_detections.append(detection)

        # Find largest detection across all detections
        max_area = 0
        largest_index = -1

        for i, detection in enumerate(all_detections):
            if detection['area'] > max_area:
                max_area = detection['area']
                largest_index = i

        # Mark the largest detection
        if largest_index >= 0 and len(all_detections) > 0:
            all_detections[largest_index]['is_largest'] = True

        return all_detections  # Return ALL detections



# ============================================================================
# FRAME ANNOTATION FUNCTION (UPDATED FOR DUAL DETECTION)
# ============================================================================
def annotate_frame(frame, detections):
    """
    Enhanced frame annotation with special pothole handling
    - RED for potholes (safety priority)
    - BLUE for largest regular object
    - Default colors for other objects
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class_name']
        class_id = det['class_id']
        confidence = det['confidence']
        is_largest = det.get('is_largest', False)
        is_pothole = det.get('is_pothole', False)

        # Calculate distance
        width_pixels = x2 - x1
        distance_cm = (KNOWN_WIDTH * FOCAL_LENGTH) / width_pixels if width_pixels > 0 else 0

        # Special color logic for potholes and largest objects
        if is_pothole:
            color = (0, 0, 255)  # RED for potholes (safety priority)
        elif is_largest:
            color = (255, 0, 0)  # BLUE for largest regular object
        else:
            color = detection_colors[class_id] if class_id < len(detection_colors) else (0, 255, 0)

        # Thicker border for potholes (safety emphasis)
        border_thickness = 4 if is_pothole else 3

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, border_thickness)

        # Enhanced label for potholes
        if is_pothole:
            # Add warning symbol for potholes
            label = f"⚠ {class_name} {distance_cm:.1f}cm ({confidence:.2f})"
        else:
            label = f"{class_name} {distance_cm:.1f}cm ({confidence:.2f})"

        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )

        # Draw background for text
        cv2.rectangle(
            frame,
            (int(x1), int(y1) - text_height - 10),
            (int(x1) + text_width, int(y1)),
            color,
            -1
        )

        # Draw text
        cv2.putText(
            frame,
            label,
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

    return frame



# ============================================================================
# MAIN FUNCTION - REAL-TIME DETECTION PIPELINE
# ============================================================================
def main():
    """
    Main real-time detection pipeline with all optimizations:
    - PyTorch GPU acceleration (CUDA if available)
    - Threaded video capture
    - Frame skipping for display smoothness
    - Streaming mode inference
    - Async speech synthesis
    - Performance metrics display
    """

    video_source = r"D:\D Drive\Object-Detection-For-Blind-People-Using-Yolov8-With-Distance-Estimation-of-Multiple-Objects-\pothole_dataset\sample_video.mp4"
    video_capture = VideoCapture(video_source).start()
    detector = DualYOLODetector(regular_model, pothole_model)
    speech = AsyncSpeech().start()

    # Timing variables
    prev_detect_time = 0
    prev_speech_time = 0

    # Detection state
    all_detections = None
    last_detection = None
    last_actual_detection = None  # Stores most recent real detection
    last_detection_time = 0  # Timestamp of last actual detection
    speech_inference_time = 0  # Track speech detection time

    # Performance metrics
    frame_count = 0
    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0
    inference_time = 0
    
    # Main processing loop
    while not video_capture.is_stopped():
        ret, frame = video_capture.read()
        
        if frame is None:
            time.sleep(0.01)
            continue
        
        current_time = time.time()
        frame_count += 1
        fps_counter += 1
        
        # Calculate FPS every second
        if current_time - fps_start_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_start_time = current_time
        
        # Run detection every SKIP_FRAMES frames
        should_detect = (frame_count % SKIP_FRAMES == 0) and \
                       (current_time - prev_detect_time >= DETECTION_INTERVAL)
        
        if should_detect:
            # Run detection with timing
            detect_start = time.time()
            all_detections = detector.detect(frame, CONFIDENCE_THRESHOLD)  # Now returns ALL
            inference_time = time.time() - detect_start

            # Update last detection
            if all_detections:
                last_detection = all_detections
                # Store largest detection for speech
                last_actual_detection = next((d for d in all_detections if d.get('is_largest')), None)
                last_detection_time = current_time

            prev_detect_time = current_time

        else:
            # Reuse last detection for smooth display
            all_detections = last_detection
        
        # ====================================================================
        # FIXED SPEECH OUTPUT - Only updates timer when speech actually occurs
        # ====================================================================
        if current_time - prev_speech_time >= SPEECH_INTERVAL:
            # Force a fresh detection for speech accuracy
            detect_start = time.time()
            fresh_detection = detector.detect(frame, CONFIDENCE_THRESHOLD)
            speech_inference_time = time.time() - detect_start
            
            speech_triggered = False  # Track if speech actually happened
            
            if fresh_detection and len(fresh_detection) > 0:
                # Use largest detection for speech
                largest_for_speech = next((d for d in fresh_detection if d.get('is_largest')), fresh_detection[0])
                bbox = largest_for_speech['bbox']
                width_pixels = bbox[2] - bbox[0]
                distance_cm = (KNOWN_WIDTH * FOCAL_LENGTH) / width_pixels if width_pixels > 0 else 0
                
                speech_text = f"Detected {largest_for_speech['class_name']} at {distance_cm:.0f} centimeters"
                speech.speak(speech_text)
                speech_triggered = True
                
                print(f"🔊 Speech: {largest_for_speech['class_name']} at {distance_cm:.0f}cm | "
                      f"Fresh detection: {speech_inference_time*1000:.1f}ms")
            
            elif last_actual_detection:
                # No detection in current frame, use last known if available
                bbox = last_actual_detection['bbox']
                width_pixels = bbox[2] - bbox[0]
                distance_cm = (KNOWN_WIDTH * FOCAL_LENGTH) / width_pixels if width_pixels > 0 else 0
                
                age = current_time - last_detection_time
                speech_text = f"Last detected {last_actual_detection['class_name']} at {distance_cm:.0f} centimeters"
                speech.speak(speech_text)
                speech_triggered = True
                
                print(f"🔊 Speech (cached): {last_actual_detection['class_name']} at {distance_cm:.0f}cm | "
                      f"Age: {age:.1f}s")
            
            # Only update timer if speech was actually triggered
            if speech_triggered:
                prev_speech_time = current_time
        
        # Annotate frame with ALL detections
        if all_detections:
            frame = annotate_frame(frame, all_detections)
        
        # Display system info on frame
        info_y = 30
        info_x = 10

        # System title
        system_text = "Object Detection For blind People With Speech Output"
        cv2.putText(frame, system_text, (info_x, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Real-Time Object Detection and Distance Estimation", frame)
        
        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break
    
    # Cleanup
    video_capture.stop()
    speech.stop()
    cv2.destroyAllWindows()



# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    except Exception as e:
        cv2.destroyAllWindows()
