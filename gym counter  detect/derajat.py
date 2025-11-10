import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def hitung_sudut(a, b, c):
    a = np.array(a)
    b = np.array(b) 
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

# Membuat warna custom untuk landmark dan koneksi
custom_landmark_spec = mp_drawing.DrawingSpec(
    color=(0, 255, 0),  # Warna hijau untuk landmark
    thickness=4,
    circle_radius=4
)

custom_connection_spec = mp_drawing.DrawingSpec(
    color=(255, 255, 0),  # Warna kuning untuk koneksi
    thickness=2
)

# Capture video
cap = cv2.VideoCapture(r"D:\02 area belajar\2025\00 Main Goals\AI & Computer Vision\gym counter  detect\tes.mp4")

# Untuk efek glow
def add_glow(image, landmarks, radius=10, intensity=0.5):
    overlay = image.copy()
    for landmark in landmarks.landmark:
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        cv2.circle(overlay, (x, y), radius, (0, 255, 0), -1)
    
    return cv2.addWeighted(overlay, intensity, image, 1 - intensity, 0)

counter = 0  # Counter untuk push-up
stage = None  # Status posisi (up/down)
    
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
        
    # Resize frame untuk performa lebih baik
    frame = cv2.resize(frame, (640, 480))
    
    # Convert BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Deteksi pose
    results = pose.process(image)
    
    # Convert kembali ke BGR untuk OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        if results.pose_landmarks:
            # Tambahkan efek glow
            image = add_glow(image, results.pose_landmarks)
            
            # Gambar landmark dan koneksi
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                custom_landmark_spec,
                custom_connection_spec)
            
            landmarks = results.pose_landmarks.landmark
            
            # Hitung sudut untuk siku kiri
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Hitung sudut
            angle = hitung_sudut(shoulder, elbow, wrist)
            
            # Visualisasi sudut
            cv2.putText(image, f"{int(angle)}deg", 
                       tuple(np.multiply(elbow, [640, 480]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                       cv2.LINE_AA)
            
            # Logic untuk menghitung push-up
            if angle < 90:
                stage = "DOWN"
            if angle > 160 and stage == "DOWN":
                stage = "UP"
                counter += 1

            # Tambahkan informasi counter dan stage
            cv2.putText(image, f'PUSH-UPS: {counter}', 
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                       cv2.LINE_AA)
            
            cv2.putText(image, f'STAGE: {stage}', 
                       (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                       cv2.LINE_AA)
            
            # Tambahkan gauge bar untuk visualisasi sudut
            bar_max = 180
            bar_value = int((angle/bar_max) * 100)
            bar_color = (0, 255, 0) if stage == "UP" else (0, 0, 255)
            cv2.rectangle(image, (500, 50), (530, 400), (255, 255, 255), 3)
            cv2.rectangle(image, (500, int(400 - bar_value * 3.5)), 
                         (530, 400), bar_color, cv2.FILLED)
            
    except Exception as e:
        print(e)
        pass

    # Tampilkan frame
    cv2.imshow('Push-up Counter', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()