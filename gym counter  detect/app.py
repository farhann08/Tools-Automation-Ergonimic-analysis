
from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
import time

app = Flask(__name__)

class PushUpDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils 
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.counter = 0
        self.stage = None
        
        self.custom_landmark_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0), 
            thickness=4,
            circle_radius=4
        )
        self.custom_connection_spec = self.mp_drawing.DrawingSpec(
            color=(255, 255, 0),  
            thickness=2
        )

    def reset_counter(self):
        self.counter = 0
        self.stage = None
        return {"status": "success", "counter": self.counter}

    def hitung_sudut(self, a, b, c):
        a = np.array(a)
        b = np.array(b) 
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
        return angle

    def add_glow(self, image, landmarks, radius=10, intensity=0.5):
        overlay = image.copy()
        for landmark in landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(overlay, (x, y), radius, (0, 255, 0), -1)
        
        return cv2.addWeighted(overlay, intensity, image, 1 - intensity, 0)

    def process_frame(self, frame):
        if frame is None:
            return None
            
        # Resize frame
        frame = cv2.resize(frame, (640, 480))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Deteksi pose
        results = self.pose.process(image)
        
        # Convert kembali ke BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            if results.pose_landmarks:
                # Tambahkan efek glow
                image = self.add_glow(image, results.pose_landmarks)
                
                # Gambar landmark dan koneksi
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.custom_landmark_spec,
                    self.custom_connection_spec)
                
                landmarks = results.pose_landmarks.landmark
                
                # Hitung sudut untuk siku kiri
                shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Hitung sudut
                angle = self.hitung_sudut(shoulder, elbow, wrist)
                
                # Visualisasi sudut
                cv2.putText(image, f"{int(angle)}deg", 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                           cv2.LINE_AA)
                
                # Logic untuk push-up
                if angle < 90:
                    self.stage = "DOWN"
                if angle > 160 and self.stage == "DOWN":
                    self.stage = "UP"
                    self.counter += 1

                # Tambahkan informasi counter dan stage
                cv2.putText(image, f'REPETITION: {self.counter}', 
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                           cv2.LINE_AA)
                
                cv2.putText(image, f'STAGE: {self.stage if self.stage else "NONE"}', 
                           (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                           cv2.LINE_AA)
                
                # Gauge bar
                bar_max = 180
                bar_value = int((angle/bar_max) * 100)
                bar_color = (0, 255, 0) if self.stage == "UP" else (0, 0, 255)
                cv2.rectangle(image, (500, 50), (530, 400), (255, 255, 255), 3)
                cv2.rectangle(image, (500, int(400 - bar_value * 3.5)), 
                             (530, 400), bar_color, cv2.FILLED)
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            pass
            
        return image

# Inisialisasi detector
detector = PushUpDetector()
video_source = None  # Akan diinisialisasi di gen_frames
video_source_type = "none"  # Default ke none (tidak ada sumber)
is_streaming = False  # Status streaming

# Video paths (dapat disesuaikan)
video_path = r"D:\02 area belajar\2025\00 Main Goals\AI & Computer Vision\gym counter  detect\tes.mp4"

def initialize_video_source():
    global video_source, video_source_type
    
    try:
        # Tutup video_source yang ada jika ada
        if video_source is not None:
            video_source.release()
            video_source = None
            
        # Buka sumber video baru berdasarkan tipe
        if video_source_type == "webcam":
            video_source = cv2.VideoCapture(0)
            print("Menggunakan webcam")
        elif video_source_type == "video":
            if os.path.exists(video_path):
                video_source = cv2.VideoCapture(video_path)
                print(f"Menggunakan file video: {video_path}")
            else:
                # Fallback ke webcam jika video tidak ditemukan
                video_source = cv2.VideoCapture(0)
                video_source_type = "webcam"
                print("Video tidak ditemukan, menggunakan webcam")
        else:
            # Jika tipe adalah "none" atau lainnya
            return False
                
        return video_source is not None and video_source.isOpened()
    except Exception as e:
        print(f"Error initializing video source: {e}")
        return False

def create_blank_frame(message="No video source selected"):
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank_frame, message, 
               (int(640/2 - 150), int(480/2)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return blank_frame

def gen_frames():
    global video_source, video_source_type, is_streaming
    
    # Jika tidak streaming, kembalikan frame kosong
    if not is_streaming:
        blank_frame = create_blank_frame()
        ret, buffer = cv2.imencode('.jpg', blank_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return
    
    # Coba inisialisasi video source
    if not initialize_video_source():
        print("Gagal menginisialisasi sumber video")
        blank_frame = create_blank_frame("Failed to initialize video source")
        ret, buffer = cv2.imencode('.jpg', blank_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return
    
    try:
        while is_streaming:
            if video_source is None or not video_source.isOpened():
                if not initialize_video_source():
                    is_streaming = False
                    break
                    
            success, frame = video_source.read()
            if not success:
                print("Tidak dapat membaca frame.")
                # Untuk video file, reset ke awal
                if video_source_type == "video":
                    video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # Coba reinisialisasi webcam
                    if not initialize_video_source():
                        is_streaming = False
                        break
                    continue
                
            processed_frame = detector.process_frame(frame)
            if processed_frame is None:
                continue
                
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Delay kecil untuk mengurangi beban CPU
            time.sleep(0.01)
    
    except Exception as e:
        print(f"Error in gen_frames: {e}")
    finally:
        if video_source is not None:
            video_source.release()
            video_source = None
        print("Video source dilepaskan")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset_counter', methods=['POST'])
def reset_counter():
    result = detector.reset_counter()
    return jsonify(result)

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global video_source_type, is_streaming
    
    # Dapatkan tipe sumber dari request
    data = request.json
    source_type = data.get('source', 'video')
    
    # Ubah sumber video
    if source_type in ['video', 'webcam']:
        video_source_type = source_type
        is_streaming = True
        result = {"status": "success", "source": video_source_type}
    else:
        result = {"status": "error", "message": "Sumber tidak valid"}
    
    return jsonify(result)

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global video_source_type, is_streaming, video_source
    
    # Hentikan streaming
    is_streaming = False
    
    # Lepaskan video source
    if video_source is not None:
        video_source.release()
        video_source = None
    
    # Tetapkan tipe ke none
    video_source_type = "none"
    
    return jsonify({"status": "success", "message": "Stream stopped"})

@app.route('/get_status', methods=['GET'])
def get_status():
    global is_streaming, video_source_type
    
    return jsonify({
        "counter": detector.counter, 
        "stage": detector.stage,
        "source": video_source_type,
        "streaming": is_streaming
    })

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    finally:
        # Bersihkan resource
        if video_source is not None:
            video_source.release()

