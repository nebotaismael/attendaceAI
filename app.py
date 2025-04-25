import os
import logging
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import uuid
# Web framework
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
# Image processing & ML
import cv2
from deepface import DeepFace
import flask_cors
# Global variable for detector
face_detector = None
detector_backend = "retinaface"  # Can be: opencv, ssd, mtcnn, dlib, retinaface, mediapipe or yolov8
# Anti-spoofing libraries
import dlib
import mediapipe as mp
# Storage
import firebase_admin
from firebase_admin import credentials, firestore, storage
# Geolocation
from geopy.distance import geodesic
# Authentication & security
import jwt
from cryptography.fernet import Fernet
import bcrypt
from livenesschech import Config, check_liveness


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


# Initialize application
app = Flask(__name__)
# Add CORS support for production
flask_cors.CORS(app)
os.makedirs(Config.TEMP_FOLDER, exist_ok=True)

# Initialize security
encryption_key = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())
cipher_suite = Fernet(encryption_key)

# Initialize Firebase
try:
    cred = credentials.Certificate(Config.FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred, {
        'storageBucket': "airecognition-63fac.firebasestorage.app"
    })
    db = firestore.client()
    bucket = storage.bucket()
    logger.info("Firebase services initialized successfully")
except Exception as e:
    logger.critical(f"Failed to initialize Firebase: {e}")
    raise


# Initialize face detection tools
try:
    face_detector = dlib.get_frontal_face_detector()  # type: dlib.fhog_object_detector
except AttributeError:
    # Fallback if type hint is incorrect for your dlib version
    face_detector = dlib.get_frontal_face_detector()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)



# Security middleware
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            logger.warning("Missing authentication token")
            return jsonify({'error': 'Authentication token is required'}), 401
            
        try:
            token = token.split("Bearer ")[1]
            data = jwt.decode(token, Config.JWT_SECRET, algorithms=["HS256"])
            request.user = data  # Add user data to request
        except Exception as e:
            logger.warning(f"Invalid token: {e}")
            return jsonify({'error': 'Invalid or expired token'}), 401
            
        return f(*args, **kwargs)
    return decorated

# Helper functions
def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_IMAGE_EXTENSIONS

def secure_save_file(file):
    """Securely save uploaded file with sanitized filename"""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add random component to prevent filename collisions
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(Config.TEMP_FOLDER, unique_filename)
        file.save(file_path)
        return file_path
    return None

def read_image(file_path):
    """Read image from file path using OpenCV"""
    image = cv2.imread(file_path)
    return image

def detect_faces(image):
    """Detect faces in an image using dlib"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    return faces, gray

def extract_face_features(image, face_rect):
    """Extract face area and compute features"""
    x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
    # Add padding to the face region
    padding = int((x2 - x1) * 0.3)
    face_img = image[max(0, y1-padding):min(image.shape[0], y2+padding),
                     max(0, x1-padding):min(image.shape[1], x2+padding)]
    return face_img


def verify_location(lat, lng, location_id=None):
    """Verify if the user is at an authorized location"""
    if not lat or not lng:
        return False, "Missing location data"
        
    if location_id and location_id in Config.AUTHORIZED_LOCATIONS:
        authorized_coords = Config.AUTHORIZED_LOCATIONS[location_id]
        current_coords = (lat, lng)
        distance = geodesic(authorized_coords, current_coords).meters
        
        if distance <= Config.ALLOWED_LOCATION_RADIUS:
            return True, f"Within authorized radius ({distance:.1f}m)"
        else:
            return False, f"Outside authorized radius ({distance:.1f}m)"
    
    # Check against all authorized locations if no specific one provided
    for loc_name, loc_coords in Config.AUTHORIZED_LOCATIONS.items():
        distance = geodesic(loc_coords, (lat, lng)).meters
        if distance <= Config.ALLOWED_LOCATION_RADIUS:
            return True, f"Within authorized radius of {loc_name} ({distance:.1f}m)"
    
    return False, "Not near any authorized location"

def generate_attendance_id(user_id, timestamp):
    """Generate a unique attendance ID based on user and time"""
    str_to_hash = f"{user_id}-{timestamp.isoformat()}"
    return hashlib.sha256(str_to_hash.encode()).hexdigest()[:20]

def cleanup_files(file_paths):
    """Remove temporary files after processing"""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                logger.debug(f"Removed temporary file: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {path}: {e}")

# API Routes

@app.route('/login', methods=['POST'])
def login():
    """User login to get authentication token. Creates user if not exists (for testing)."""
    auth = request.authorization
    
    if not auth or not auth.username or not auth.password:
        return jsonify({'error': 'Missing credentials'}), 401
    
    try:
        # Get user from Firestore
        user_ref = db.collection('users').document(auth.username)
        user = user_ref.get()
        user_data = {} # Initialize user_data

        if not user.exists:
            # --- START: Auto-register user if not found (for testing) ---
            logger.info(f"User '{auth.username}' not found. Creating new user for testing.")
            # Hash the provided password
            hashed_password = bcrypt.hashpw(auth.password.encode(), bcrypt.gensalt())
            
            # Create user data
            user_data = {
                'username': auth.username,
                'password_hash': hashed_password.decode(), # Store hash as string
                'name': auth.username, # Default name to username
                'created_at': datetime.utcnow()
            }
            
            # Save the new user to Firestore
            user_ref.set(user_data)
            logger.info(f"User '{auth.username}' created successfully during login.")
            # --- END: Auto-register user ---
        else:
            user_data = user.to_dict()
            # Check password (assuming passwords are stored as bcrypt hashes)
            if 'password_hash' not in user_data or not bcrypt.checkpw(auth.password.encode(), user_data['password_hash'].encode()):
                logger.warning(f"Failed login attempt for user: {auth.username}")
                return jsonify({'error': 'Invalid credentials'}), 401
            
        # Generate token
        token_expiry = datetime.utcnow() + timedelta(seconds=Config.JWT_EXPIRATION) # Corrected datetime usage
        token = jwt.encode(
            {
                'user_id': auth.username,
                'name': user_data.get('name', ''),
                'exp': token_expiry
            }, 
            Config.JWT_SECRET, 
            algorithm="HS256"
        )
        
        return jsonify({
            'token': token,
            'expires_at': token_expiry.isoformat(),
            'user': {
                'id': auth.username,
                'name': user_data.get('name', '')
            }
        })
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Authentication failed'}), 500


@app.route('/attendance/register', methods=['POST'])
@token_required
def register_face():
    """Register a user's face for future attendance verification"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    user_id = request.user['user_id']
    file = request.files['image']
    print(user_id, file.filename)
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
        
    try:
        # Save and process the image
        file_path = secure_save_file(file)
        if not file_path:
            return jsonify({'error': 'Failed to save image'}), 500
            
        image = read_image(file_path)
        if image is None:
            return jsonify({'error': 'Failed to read image'}), 400
            
        # Detect face
        faces, _ = detect_faces(image)
        if len(faces) == 0:
            return jsonify({'error': 'No face detected in image'}), 400
        if len(faces) > 1:
            return jsonify({'error': 'Multiple faces detected, please provide an image with only your face'}), 400
            
        # Check liveness
        face_img = extract_face_features(image, faces[0])
        is_live, liveness_score = check_liveness(face_img)
        
        if not is_live:
            logger.warning(f"Liveness check failed for user {user_id}: score {liveness_score:.4f}")
            return jsonify({'error': 'Liveness check failed. Please ensure you are using a real face.'}), 400
            
        # Store the reference image in Firebase Storage
        timestamp = datetime.utcnow()
        image_path = f"reference_faces/{user_id}/{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        blob = bucket.blob(image_path)
        
        # Upload the image
        with open(file_path, 'rb') as img_file:
            blob.upload_from_file(img_file)
            
        # Create/update user face profile in Firestore
        db.collection('users').document(user_id).set({
            'reference_face': image_path,
            'reference_face_updated': timestamp,
            'liveness_score': liveness_score
        }, merge=True)
        
        # Cleanup
        cleanup_files([file_path])
        
        return jsonify({
            'status': 'success',
            'message': 'Face registered successfully',
            'timestamp': timestamp.isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in face registration: {e}")
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/attendance/verify', methods=['POST'])
@token_required
def verify_attendance():
    """Complete attendance verification with multi-factor authentication"""
    user_id = request.user['user_id']
    
    # Required fields
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    # Get location information
    latitude = request.form.get('latitude', type=float)
    longitude = request.form.get('longitude', type=float)
    location_id = request.form.get('location_id')
    
    # Get additional validation factors
    pin_code = request.form.get('pin_code')
    device_id = request.form.get('device_id')
    
    # Processing starts
    temp_files = []
    try:
        # 1. Process and validate the verification image
        file = request.files['image']
        file_path = secure_save_file(file)
        temp_files.append(file_path)
        
        if not file_path:
            return jsonify({'error': 'Failed to save image'}), 500
            
        verification_image = read_image(file_path)
        if verification_image is None:
            return jsonify({'error': 'Failed to read image'}), 400
            
        # 2. Face detection
        faces, _ = detect_faces(verification_image)
        if len(faces) == 0:
            return jsonify({'error': 'No face detected in verification image'}), 400
        if len(faces) > 1:
            return jsonify({'error': 'Multiple faces detected, please provide a clear image with only your face'}), 400
            
        # 3. Liveness detection (anti-spoofing)
        face_img = extract_face_features(verification_image, faces[0])
        is_live, liveness_score = check_liveness(face_img)
        # Convert to native Python types
        is_live = bool(is_live)
        liveness_score = float(liveness_score)
        
        if not is_live:
            logger.warning(f"Liveness check failed during verification for user {user_id}: score {liveness_score:.4f}")
            # Log the attempt as potentially fraudulent
            db.collection('security_events').add({
                'user_id': user_id,
                'event_type': 'liveness_check_failed',
                'timestamp': datetime.utcnow(),
                'liveness_score': liveness_score,
                'device_id': device_id,
                'latitude': float(latitude) if latitude else None,
                'longitude': float(longitude) if longitude else None
            })
            return jsonify({
                'error': 'Liveness check failed. Please ensure you are using a real face.',
                'verified': False,
            }), 400
            
        # 4. Get the user's reference face from Firestore
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return jsonify({'error': 'User profile not found'}), 404
            
        user_data = user_doc.to_dict()
        if 'reference_face' not in user_data:
            return jsonify({'error': 'No reference face registered for this user'}), 400
            
        # 5. Download reference image from Firebase Storage
        reference_blob = bucket.blob(user_data['reference_face'])
        reference_path = os.path.join(Config.TEMP_FOLDER, f"ref_{user_id}_{uuid.uuid4().hex}.jpg")
        reference_blob.download_to_filename(reference_path)
        temp_files.append(reference_path)
        
        reference_image = read_image(reference_path)
        if reference_image is None:
            return jsonify({'error': 'Failed to read reference image'}), 500
                    
        # 6. Face comparison using DeepFace
        try:
            result = DeepFace.verify(
                face_img, 
                reference_image,
                model_name="VGG-Face",
                enforce_detection=False,
                detector_backend="dlib"
            )
            # Convert NumPy types to Python native types
            face_match = bool(result.get("verified", False))
            face_distance = float(result.get("distance", 1.0))
            face_match_confidence = float(max(0, min(100, 100 * (1 - face_distance / 2))))
                    
        except Exception as e:
            logger.error(f"Face verification error: {e}")
            return jsonify({'error': f'Face verification failed: {str(e)}'}), 500
            
        # 7. Location verification
        location_verified, location_message = verify_location(latitude, longitude, location_id)
        location_verified = bool(location_verified)  # Ensure Python native boolean
        
        # 8. Optional PIN verification
        pin_verified = False
        if pin_code:
            stored_pin_hash = user_data.get('pin_hash')
            if stored_pin_hash:
                pin_verified = bool(bcrypt.checkpw(pin_code.encode(), stored_pin_hash.encode()))
        
        # 9. Compile verification results
        timestamp = datetime.utcnow()
        attendance_id = generate_attendance_id(user_id, timestamp)
        
        # Determine overall verification status with native Python types
        verification_factors = [
            {
                "factor": "face_recognition", 
                "verified": bool(face_match), 
                "confidence": float(face_match_confidence)
            },
            {
                "factor": "liveness", 
                "verified": bool(is_live), 
                "confidence": float(liveness_score * 100)
            },
            {
                "factor": "location", 
                "verified": bool(location_verified), 
                "message": location_message
            }
        ]
        
        if pin_code:
            verification_factors.append({
                "factor": "pin_code", 
                "verified": bool(pin_verified)
            })
            
        # Calculate overall verification status
        verified = bool(face_match and is_live and location_verified)
        if pin_code:
            verified = bool(verified and pin_verified)
        
        # 10. Store attendance record with native Python types
        attendance_record = {
            'attendance_id': attendance_id,
            'user_id': user_id,
            'timestamp': timestamp,
            'verified': bool(verified),
            'verification_factors': verification_factors,
            'face_distance': float(face_distance),
            'device_id': device_id,
            'location': {
                'latitude': float(latitude) if latitude else None,
                'longitude': float(longitude) if longitude else None,
                'location_id': location_id,
                'verified': bool(location_verified),
                'message': location_message
            }
        }
        
        # Store record in Firestore
        db.collection('attendance').document(attendance_id).set(attendance_record)
        
        # 11. Update user's attendance history
        user_ref.collection('attendance_history').add({
            'attendance_id': attendance_id,
            'timestamp': timestamp,
            'verified': bool(verified),
            'location_verified': bool(location_verified)
        })
        
        # 12. Create appropriate response
        response = {
            'attendance_id': attendance_id,
            'timestamp': timestamp.isoformat(),
            'verified': bool(verified),
            'verification_details': verification_factors
        }
        
        # 13. Cleanup temporary files
        cleanup_files(temp_files)
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Attendance verification error: {e}")
        # Ensure cleanup happens even on error
        cleanup_files(temp_files)
        return jsonify({'error': f'Attendance verification failed: {str(e)}'}), 500
@app.route('/attendance/status', methods=['GET'])
@token_required
def attendance_status():
    """Get user's attendance status and history"""
    user_id = request.user['user_id']
    days = request.args.get('days', default=7, type=int)
    
    try:
        # Get today's attendance
        today = datetime.utcnow().date()
        today_start = datetime.combine(today, datetime.min.time())
        today_end = datetime.combine(today, datetime.max.time())
        
        today_query = db.collection('attendance')\
            .where('user_id', '==', user_id)\
            .where('timestamp', '>=', today_start)\
            .where('timestamp', '<=', today_end)\
            .order_by('timestamp', direction=firestore.Query.DESCENDING)\
            .stream()
        
        today_records = [doc.to_dict() for doc in today_query]
        
        # Get attendance history for specified number of days
        history_start = datetime.combine(today - datetime.timedelta(days=days), datetime.min.time())
        
        history_query = db.collection('attendance')\
            .where('user_id', '==', user_id)\
            .where('timestamp', '>=', history_start)\
            .where('timestamp', '<', today_start)\
            .order_by('timestamp', direction=firestore.Query.DESCENDING)\
            .stream()
        
        history_records = [doc.to_dict() for doc in history_query]
        
        # Format response
        response = {
            'user_id': user_id,
            'today': {
                'date': today.isoformat(),
                'records': today_records,
                'present': any(record.get('verified', False) for record in today_records)
            },
            'history': [
                {
                    'date': record['timestamp'].date().isoformat(),
                    'verified': record.get('verified', False),
                    'timestamp': record['timestamp'].isoformat()
                }
                for record in history_records
            ]
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error retrieving attendance status: {e}")
        return jsonify({'error': f'Failed to retrieve attendance status: {str(e)}'}), 500

if __name__ == '__main__':
    # Use PORT environment variable provided by Heroku
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)