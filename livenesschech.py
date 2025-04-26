import concurrent.futures
import logging

import mediapipe as mp
import cv2
from deepface import DeepFace
import numpy as np
import os
# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("attendance_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
detector_backend = "retinaface"  # Can be: opencv, ssd, mtcnn, dlib, retinaface, mediapipe or yolov8
# Environment configuration
class Config:
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    FACE_MATCH_THRESHOLD = 0.4  # Lower is stricter
    LIVENESS_THRESHOLD = 0.92  # Higher is stricter
    ALLOWED_LOCATION_RADIUS = 100  # meters
    TEMP_FOLDER = 'temp'
    JWT_SECRET = os.getenv('JWT_SECRET', 'development_secret_key')
    JWT_EXPIRATION = 3600  # 1 hour
    # Firebase configuration
    FIREBASE_CRED_PATH = os.getenv('FIREBASE_CREDENTIALS', 'path/to/credentials.json')
    # Authorized locations - would come from a database in production
    AUTHORIZED_LOCATIONS = {
        'office1': (40.7128, -74.0060),  # New York
        'office2': (34.0522, -118.2437),  # Los Angeles
    }

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)
def check_liveness(face_image):
    """
    Multi-factor liveness detection and anti-spoofing check
    Returns: (is_live, confidence_score)
    """
    try:
        logger.info("Starting liveness detection")
        
        # Input validation
        if face_image is None or face_image.size == 0:
            logger.error("Invalid input: face_image is None or empty")
            return False, 0.0
            
        # Check if required dependencies are available
        if 'face_mesh' not in globals():
            logger.error("Required dependency 'face_mesh' not initialized")
            return False, 0.0
            
        # Resize image for better performance if it's too large
        max_dimension = 640
        h, w = face_image.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            face_image = cv2.resize(face_image, (int(w * scale), int(h * scale)))
            logger.debug(f"Resized image from {w}x{h} to {int(w * scale)}x{int(h * scale)}")
            
        # RGB conversion (used by multiple tasks)
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Define individual analysis functions to run in parallel
        def analyze_emotion():
            try:
                emotion_analysis = DeepFace.analyze(face_image, actions=['emotion'], 
                                                   enforce_detection=False, 
                                                   detector_backend=detector_backend)
                dominant_emotion = emotion_analysis[0]['dominant_emotion']
                emotion_score = emotion_analysis[0]['emotion'][dominant_emotion] / 100
                normalized_score = min(emotion_score, 0.95)
                logger.debug(f"Emotion analysis score: {normalized_score:.4f}")
                return normalized_score
            except Exception as e:
                logger.warning(f"Emotion analysis failed: {e}")
                return 0.5
                
        def analyze_landmarks():
            try:
                result = face_mesh.process(rgb_image)
                
                if not result.multi_face_landmarks:
                    logger.debug("No face landmarks detected")
                    return 0.5, 0.5  # Return default scores for ear and symmetry
                    
                landmarks = result.multi_face_landmarks[0].landmark
                
                # Calculate Eye Aspect Ratio (EAR)
                def euclidean_dist(p1, p2):
                    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5
                
                def calculate_ear(eye_pts):
                    v1 = euclidean_dist(eye_pts[1], eye_pts[5])
                    v2 = euclidean_dist(eye_pts[2], eye_pts[4])
                    h = euclidean_dist(eye_pts[0], eye_pts[3])
                    return (v1 + v2) / (2.0 * h)
                
                # Eye landmark indices
                left_eye_pts = [landmarks[362], landmarks[385], landmarks[387], 
                               landmarks[263], landmarks[373], landmarks[380]]
                right_eye_pts = [landmarks[33], landmarks[160], landmarks[158], 
                                landmarks[133], landmarks[153], landmarks[144]]
                
                left_ear = calculate_ear(left_eye_pts)
                right_ear = calculate_ear(right_eye_pts)
                avg_ear = (left_ear + right_ear) / 2.0
                
                ear_score = min(1.0, max(0.0, (avg_ear - 0.15) / 0.15))
                logger.debug(f"Eye aspect ratio score: {ear_score:.4f}")
                
                # Calculate facial symmetry
                symmetry_points = [
                    (landmarks[234], landmarks[454]),  # Face edges
                    (landmarks[93], landmarks[323]),   # Mouth corners
                    (landmarks[70], landmarks[300])    # Eyebrows
                ]
                
                symmetry_score = 0
                for left, right in symmetry_points:
                    left_dist = euclidean_dist(landmarks[1], left)
                    right_dist = euclidean_dist(landmarks[1], right)
                    point_symmetry = 1.0 - min(left_dist, right_dist) / max(left_dist, right_dist)
                    symmetry_score += point_symmetry
                
                symmetry_score = 1.0 - (symmetry_score / len(symmetry_points))
                logger.debug(f"Facial symmetry score: {symmetry_score:.4f}")
                
                return ear_score, symmetry_score
            except Exception as e:
                logger.warning(f"Landmark analysis failed: {e}")
                return 0.5, 0.5
                
        def analyze_demographics():
            try:
                demographics = DeepFace.analyze(face_image, actions=['age', 'gender'], 
                                              enforce_detection=False, 
                                              detector_backend=detector_backend)
                if demographics and len(demographics) > 0:
                    # Non-integer age values are more natural for real faces
                    age = demographics[0]['age']
                    age_confidence = 0.8 if (age % 1 != 0) else 0.6
                    logger.debug(f"Age analysis confidence: {age_confidence:.4f}")
                    return age_confidence
                return 0.6
            except Exception as e:
                logger.warning(f"Demographics analysis failed: {e}")
                return 0.6
                
        def analyze_texture():
            try:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                # Simple edge detection to find texture patterns
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                
                if edge_density < 0.01:  # Too smooth
                    score = 0.4
                elif edge_density > 0.3:  # Too noisy
                    score = 0.5
                else:
                    # Normalize between 0.7-0.95 for reasonable edge density
                    score = 0.7 + min(0.25, edge_density)
                    
                logger.debug(f"Texture analysis score: {score:.4f}")
                return score
            except Exception as e:
                logger.warning(f"Texture analysis failed: {e}")
                return 0.6
        
        # Execute analyses in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start all tasks
            emotion_future = executor.submit(analyze_emotion)
            landmarks_future = executor.submit(analyze_landmarks)
            demographics_future = executor.submit(analyze_demographics)
            texture_future = executor.submit(analyze_texture)
            
            # Get results with timeout to prevent hanging
            emotion_score = emotion_future.result(timeout=3000)
            ear_score, symmetry_score = landmarks_future.result(timeout=3000)
            age_confidence = demographics_future.result(timeout=3000)
            texture_score = texture_future.result(timeout=3000)
            
        # Compile scores
        scores = [emotion_score, ear_score, symmetry_score, age_confidence, texture_score]
        
        # Calculate final liveness score as weighted average
        weights = [0.25, 0.2, 0.2, 0.15, 0.2]  # Adjust weights based on reliability of each factor
        final_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        # Decision threshold - adjustable based on security requirements
        is_live = final_score >= Config.LIVENESS_THRESHOLD
        
        logger.info(f"Liveness check completed: Score={final_score:.4f}, Result={'PASS' if is_live else 'FAIL'}")
        return is_live, final_score
        
    except concurrent.futures.TimeoutError as e:
        logger.error(f"Liveness check timed out: {str(e)}")
        return False, 0.0
    except Exception as e:
        logger.error(f"Liveness check error: {str(e)}")
        return False, 0.0