import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk

class PushUpAnalyzer:
    def __init__(self):
        # Inisialisasi MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Custom drawing specs
        self.custom_landmark_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Warna hijau untuk landmark
            thickness=4,
            circle_radius=4
        )

        self.custom_connection_spec = self.mp_drawing.DrawingSpec(
            color=(255, 255, 0),  # Warna kuning untuk koneksi
            thickness=2
        )

        # Variabel tracking
        self.counter = 0
        self.stage = None
        self.is_paused = False

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
                if not self.is_paused:
                    if angle < 90:
                        self.stage = "DOWN"
                    if angle > 160 and self.stage == "DOWN":
                        self.stage = "UP"
                        self.counter += 1

                # Tampilkan informasi
                cv2.putText(image, f'PUSH-UPS: {self.counter}', 
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                           cv2.LINE_AA)
                
                cv2.putText(image, f'STAGE: {self.stage}', 
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
            print(e)
            pass

        return image

def create_control_window():
    control_window = tk.Tk()
    control_window.title("Push-up Controls")
    control_window.geometry("200x150")

    # Buat frame untuk tombol
    button_frame = ttk.Frame(control_window)
    button_frame.pack(pady=10)

    # Tombol kontrol
    ttk.Button(button_frame, text="Reset Counter", 
               command=lambda: reset_counter()).pack(pady=5)
    ttk.Button(button_frame, text="Pause/Resume", 
               command=lambda: toggle_pause()).pack(pady=5)
    ttk.Button(button_frame, text="Quit", 
               command=lambda: quit_program()).pack(pady=5)

    return control_window

def reset_counter():
    analyzer.counter = 0
    analyzer.stage = None

def toggle_pause():
    analyzer.is_paused = not analyzer.is_paused

def quit_program():
    cap.release()
    cv2.destroyAllWindows()
    control_window.quit()

# Main program
if __name__ == "__main__":
    # Inisialisasi
    analyzer = PushUpAnalyzer()
    cap = cv2.VideoCapture(r"D:\02 area belajar\2025\00 Main Goals\AI & Computer Vision\gym counter  detect\tes.mp4")
    
    # Buat window kontrol
    control_window = create_control_window()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Proses frame
        output_frame = analyzer.process_frame(frame)
        
        # Tampilkan frame
        cv2.imshow('Push-up Counter', output_frame)
        
        # Update GUI
        control_window.update()
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    control_window.destroy()