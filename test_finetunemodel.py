import os
import cv2
import csv
import json
import time
import random
import requests
import keyboard 
import threading
import config_gui
import numpy as np
import pandas as pd
import tkinter as tk
import add_camera_gui
import mediapipe as mp
import tensorflow as tf
from collections import deque
from dotenv import load_dotenv
from PIL import Image, ImageTk
from flask import Flask, request, jsonify
from tkinter import messagebox, simpledialog
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tkinter import simpledialog, messagebox, Toplevel
from urllib.parse import urlparse, urlunparse, quote

# ==================================================  set env  =======================================

ENV_PATH = ".env"

# ================================================== Flask Setup ======================================

app = Flask(__name__)
stop_flag = False

# ==================================================== Config ==========================================
MAX_CAMERAS = 9
CONFIG_FILE = "stream_config.json"

capture_threads = {}
capture_stop_flags = {}

# ==================================================== Train Data =======================================
TRAINING_DATA_FILE = "training_data.csv"

# ============================================== ป้องกันโมเดลใช้งานพร้อมกัน ==================================

model_lock = threading.Lock () 

# ============================================ ประกาศตัวแปร Media Pipe.  ==================================

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose



# ========================================== โหลด username / password จาก .env ทุกครั้งที่เรียก  ==============
def get_credentials():
    load_dotenv(ENV_PATH, override=True)
    username = os.getenv("ADMIN_USERNAME")
    password = os.getenv("ADMIN_PASSWORD")
    return username, password

# แก้ไขรหัสผ่านในไฟล์ .env แล้วโหลดใหม่ทันที
def update_env_password(new_password):
    with open(ENV_PATH, "r") as file:
        lines = file.readlines()

    with open(ENV_PATH, "w") as file:
        for line in lines:
            if line.startswith("ADMIN_PASSWORD="):
                file.write(f"ADMIN_PASSWORD={new_password}\n")
            else:
                file.write(line)

    load_dotenv(ENV_PATH, override=True)  # โหลดใหม่หลังแก้



# ================================== ประกาศ Global history  ของตำแหน่งข้อมือ ===============================


wrist_history = {
    'left_x': deque(maxlen=15),
    'left_y': deque(maxlen=15),
    'right_x': deque(maxlen=15),
    'right_y': deque(maxlen=15)
}


# ============================================ Train Model จากข้อมูลใหม่ ====================================


def retrain_model_thread() :
    global model
    while True :
        time.sleep(3600)

        try :
            df = pd.read_csv ("training_data.csv")

            X= df.drop("label", axis=1).values
            y= df["label"].values
            X_seq, y_seq = [], []
            for i in range (0, len(X) - 30) :
                X_seq.append(X[i:i+30])
                y_seq.append(y[i+29])
            X_seq, y_seq = np.array(X_seq), np.array(y_seq)


            #New Model
            new_model = tf.keras.Sequential ([
                tf.keras.layers.LSTM(64, return_sequences = True, input_shape=(30, X.shape[1])),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation = 'sigmoid')
            ])

            new_model.compile(optimizer=Adam(1e-4), loss = 'binary_crossentropy', metrics= ['accuracy'])

            new_model.fit(X_seq, y_seq, epochs=5, batch_size=32, verbose=0)

            #สำรองโมเดลเดิม 
            with model_lock: 
                model.save("backup_model.h5")
                model =  new_model

                print("Retrain and Upload Model Successfully")

        except Exception as e : 
            print(f"Retrain Model Error : {e}")

def load_stream_sources():
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w') as f:
            json.dump({"streams": []}, f)
        return []

    with open(CONFIG_FILE, 'r') as f:
        data = json.load(f)

    # รองรับทั้งแบบ list และ dictionary
    if isinstance(data, dict) and "streams" in data:
        sources = data["streams"]
    elif isinstance(data, list):
        sources = data
    else:
        print("[ERROR] stream_config.json รูปแบบไม่ถูกต้อง")
        return []

    if len(sources) > MAX_CAMERAS:
        print(f"[WARN] กล้องเกิน {MAX_CAMERAS} ตัว ระบบจะใช้แค่ {MAX_CAMERAS} ตัวแรก")
        sources = sources[:MAX_CAMERAS]

    return sources

def save_stream_sources(sources):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(sources, f, indent=2)

# ==================================================== GPU config ========================================

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"✅ ใช้ GPU ได้: {physical_devices}")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except Exception as e:
        print(f"⚠️ Warning setting memory growth: {e}")
else:
    print("❌ ไม่พบ GPU")

# ==================================================== Load Model =======================================

model = tf.keras.models.load_model("/mnt/c/ElderlyCareSystem/models/falldetect_bi_lstm_testmodel.h5")

# ==================================================== Global variables =================================

stream_sources = load_stream_sources()

frames = [np.zeros((240, 320, 3), dtype=np.uint8)] * len(stream_sources)
sequence_list = [[] for _ in stream_sources]
lock = threading.Lock()

fall_counters = [0] * len(stream_sources)
FALL_CONFIRM_FRAMES = 5
last_log_time = [0] * len(stream_sources)
last_person_detected = [False] * len(stream_sources)
LOG_INTERVAL = 2.0
last_debug_log_time = [0] * len(stream_sources)

# สำหรับเก็บเวลาที่เริ่มล้มจริงๆ (นิ่ง)
fall_start_time = [None] * len(stream_sources)
FALL_ALERT_DELAY = 20  # วินาทีที่ต้องนิ่งก่อนแจ้งเตือน


# ==================================================== Login GUI ==========================================

def login_and_start():
    def verify_login():
        username = username_entry.get()
        password = password_entry.get()
        
        username_env, password_env = get_credentials()
        
        if username == username_env and password == password_env:
            login_win.destroy()
            main()
        else:
            messagebox.showerror("Login Failed", "Username or Password is incorrect.")

 
    def reset_password() :
        confirm = messagebox.askyesno("Reset Password", "Do you want to reset your password.")
        if confirm :
            new_pass = simpledialog.askstring("Reset Password", "Enter your new password.", show="*")
            if new_pass :
                update_env+password(new_pass)
                messagebox.showinfo("Success", "Reset password successfully")


    login_win = tk.Tk()
    login_win.title("Login")
    
    w, h = 500, 320  
    screen_w = login_win.winfo_screenwidth()
    screen_h = login_win.winfo_screenheight()
    x = (screen_w // 2) - (w // 2)
    y = (screen_h // 2) - (h // 2)
    login_win.geometry(f"{w}x{h}+{x}+{y}")
    login_win.resizable(False, False)  

    # ฟอนต์
    label_font = ("Arial", 14)
    entry_font = ("Arial", 14)

    # ช่องว่างด้านบน
    tk.Label(login_win, text="Admin Login", font=("Arial", 18, "bold")).pack(pady=15)

    # Username
    tk.Label(login_win, text="Username :", font=label_font).pack(pady=(5, 2))
    username_entry = tk.Entry(login_win, font=entry_font, width=30)
    username_entry.pack()

    # Password
    tk.Label(login_win, text="Password :", font=label_font).pack(pady=(10, 2))
    password_entry = tk.Entry(login_win, show="*", font=entry_font, width=30)
    password_entry.pack()

    # Login button
    tk.Button(login_win, text="Login", command=verify_login, font=("Arial", 14), fg="black", width=15).pack(pady=20)

    tk.Button(login_win, text="forget Password", command=reset_password, font=("Arial", 12), fg="red", bd=0).pack()

    
    login_win.mainloop()

# ==================================================== เก็บข้อมูลเพื่อไปทำ Online Training =================================


def save_training_data(keypoints, label) :
    if not os.path.exists(TRAINING_DATA_FILE) : 
        with open(TRAINING_DATA_FILE, "w", newline = '') as f : 
            writer = csv.writer(f)
            writer.writerow([f"f{i}" for i in range (len(keypoints))] + ["label"])
    with open (TRAINING_DATA_FILE, "a", newline = '') as f : 
        writer =  csv.writer(f)
        writer.writerow(keypoints + [label])


# ====================================================== Class Camera App ===============================================

class CameraApp:
    def __init__(self, master, frames, lock):
        self.master = master
        self.frames = frames
        self.lock = lock
        self.video_labels = []
        self.update_interval = 30  # ms

        self.master.title("Fall Detection - Real-time")
        self.master.geometry("1080x720")

        self.fall_detected_flags = [False] * len (self.frames)
        self.popup_shown_flags = [False] * len (self.frames)


        # ============================================== Scrollable canvas ================================================ 
        
        self.canvas = tk.Canvas(master)
        self.scrollbar = tk.Scrollbar(master, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # สร้าง frame สำหรับปุ่ม
        self.button_frame = tk.Frame(self.scrollable_frame)
        self.button_frame.grid(row=0, column=0, columnspan=2, sticky="w", pady=10, padx=10)

        # ปุ่ม Add
        self.btn_add_camera = tk.Button(self.button_frame, text="Add New Camera", command=self.add_camera_dialog)
        self.btn_add_camera.pack(side="left", padx=(0, 5))

        # ปุ่ม Remove
        self.btn_remove_camera = tk.Button(self.button_frame, text="Remove Camera", command=self.remove_camera_dialog)
        self.btn_remove_camera.pack(side="left", padx=(5, 0))


        self.create_video_labels()
        self.update_videos()


    def create_video_labels(self):
        for lbl in self.video_labels:
            lbl.destroy()
        self.video_labels.clear()

        with self.lock:
            num_cams = len(self.frames)

        for i in range(num_cams):
            lbl = tk.Label(self.scrollable_frame)
            lbl.grid(row=(i // 3) + 1, column=i % 3, padx=5, pady=5)  # +1 เพราะแถว 0 คือปุ่ม
            self.video_labels.append(lbl)


    def update_videos(self):
        with self.lock:
            frames_copy = self.frames.copy()

        for i, frame in enumerate(frames_copy):
            if isinstance(frame, np.ndarray) and frame.size != 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((320, 240))

                imgtk = ImageTk.PhotoImage(image=img)
                if i < len(self.video_labels):
                    self.video_labels[i].imgtk = imgtk
                    self.video_labels[i].config(image=imgtk)


            if i < len(self.fall_detected_flags) :
                if self.fall_detected_flags[i] :
                    if not self.popup_shown_flags[i] :
                        self.popup_shown_flags[i] = True
                        self.show_fall_popup(i)
                
        if len(frames_copy) != len(self.video_labels):
            self.create_video_labels()

        self.master.after(self.update_interval, self.update_videos)


    def add_camera_dialog(self):
        ip = simpledialog.askstring("Add Camera", "Enter new camera IP or RTSP URL:")
        if ip:
            self.add_camera_api(ip)


    def add_camera_api(self, ip):
        try:
            url = "http://localhost:5001/add_camera"
            resp = requests.post(url, json={"ip": ip})
            if resp.status_code == 200:
                messagebox.showinfo("Success", f"Camera added: {ip}")
            else:
                error = resp.json().get("error", "Unknown error")
                messagebox.showerror("Error", f"Failed to add camera:\n{error}")
        except Exception as e:
            messagebox.showerror("Error", f"Error connecting to API:\n{e}")

        self.fall_detected_flags.append(False)
        self.popup_shown_flags.append(False)


    def remove_camera_dialog(self):
        with self.lock:
            num_cams = len(self.frames)

        if num_cams == 0:
            messagebox.showwarning("No Camera", "No cameras to remove.")
            return

        cam_order = simpledialog.askinteger("Remove Camera", "Enter camera number:")  
        print("User input:", cam_order)  

        if cam_order is not None and 1 <= cam_order <= num_cams:
            index = cam_order - 1
            self.remove_camera_api(index)
        else:
            messagebox.showerror("Error", "Invalid camera number.")


    def remove_camera_api(self, index):
        try:
            url = "http://localhost:5001/remove_camera"
            resp = requests.post(url, json={"index": index})
            if resp.status_code == 200:
                messagebox.showinfo("Success", f"Camera removed successfuly")
            else:
                error = resp.json().get("error", "Unknown error")
                messagebox.showerror("Error", f"Failed to remove camera:\n{error}")
        except Exception as e:
            messagebox.showerror("Error", f"Error connecting to API:\n{e}")
        
        self.fall_detected_flags.pop(index)
        self.popup_shown_flags.pop(index)

        
    def show_fall_popup (self, index):
        popup = Toplevel(self.master)
        popup.title("Notifications")
        popup.geometry("300x100")
        popup.configure(bg="white")

        label = tk.Label(popup, text=f" Camera {index+1} FAll DETECTED!", fg="red", font=("Arial", 12, "bold"), bg="white")
        label.pack(pady=20)

        def destroy_popup():
            popup.destroy()
            self.fall_detected_flags[index] = False  

        popup.after(5000, destroy_popup)


# ======================================================== Helper Function ==================================================

def calculate_body_angle_3d(keypoints):
    try:
        shoulder = np.array([keypoints[11*3], keypoints[11*3+1], keypoints[11*3+2]])
        hip = np.array([keypoints[23*3], keypoints[23*3+1], keypoints[23*3+2]])
        knee = np.array([keypoints[25*3], keypoints[25*3+1], keypoints[25*3+2]])
        vec1 = shoulder - hip
        vec2 = knee - hip
        angle = np.degrees(np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0)))
        return angle
    except Exception as e:
        print(f"[ERROR] calculate_body_angle_3d: {e}")
        return None

def calculate_knee_angle_3d(keypoints):
    try:
        hip = np.array([keypoints[23*3], keypoints[23*3+1], keypoints[23*3+2]])
        knee = np.array([keypoints[25*3], keypoints[25*3+1], keypoints[25*3+2]])
        ankle = np.array([keypoints[27*3], keypoints[27*3+1], keypoints[27*3+2]])
        vec1 = hip - knee
        vec2 = ankle - knee
        angle = np.degrees(np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0)))
        return angle
    except Exception as e:
        print(f"[ERROR] calculate_knee_angle_3d: {e}")
        return None

def is_fallen_by_locked_z(keypoints, lock_point_index=23, z_threshold=0.1):
    lock_z = keypoints[lock_point_index * 3 + 2]
    check_points = [0, 11, 12, 23, 24, 25, 26]
    return all(abs(keypoints[idx * 3 + 2] - lock_z) <= z_threshold for idx in check_points)


def is_squat_pose(keypoints):
    try:
        hip_y = keypoints[23*3 + 1]
        ankle_y = keypoints[27*3 + 1]
        return abs(hip_y - ankle_y) < 0.15
    except Exception as e:
        print(f"[ERROR] is_squat_pose: {e}")
        return False


def calculate_body_angle_y_axis(keypoints):
    try:
        shoulder = np.array([keypoints[11*3], keypoints[11*3+1], keypoints[11*3+2]])
        hip = np.array([keypoints[23*3], keypoints[23*3+1], keypoints[23*3+2]])
        vec = shoulder - hip
        vertical = np.array([0, -1, 0])  
        cos_theta = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
        return angle
    except Exception as e:
        print(f"[ERROR] calculate_body_angle_y_axis: {e}")
        return None


def is_standing_pose(keypoints, height_threshold=0.15, angle_threshold=40):
    try:
        hip_y = keypoints[23*3 + 1]
        ankle_y = keypoints[27*3 + 1]
        body_angle_y = calculate_body_angle_y_axis(keypoints)

        hip_above_ankle = (ankle_y - hip_y) > height_threshold
        is_straight = (body_angle_y is not None and body_angle_y < angle_threshold)

        return hip_above_ankle and is_straight
    except Exception as e:
        print(f"[ERROR] is_standing_pose: {e}")
        return False


def detect_gesture_ok(landmarks, handedness_label):
    try:
        thumb = np.array([landmarks[4].x, landmarks[4].y])
        index = np.array([landmarks[8].x, landmarks[8].y])
        dist = np.linalg.norm(thumb - index)

        return dist < 0.05
    except Exception as e:
        print(f"[ERROR] detect_gesture_ok: {e}")
        return False


def detect_hand_raised(pose_results):
    try:
        if not pose_results or not pose_results.pose_landmarks:
            return False

        landmarks = pose_results.pose_landmarks.landmark

        # ตรวจสอบว่าจุดสำคัญมี visibility เพียงพอ
        required_points = [
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]

        for point in required_points:
            if landmarks[point.value].visibility < 0.7:
                return False

        # ดึงตำแหน่ง
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        # เงื่อนไข: มือยกสูงกว่าหัวไหล่ และศอกอยู่ต่ำกว่าไหล่
        left_hand_raised = (
            left_wrist.y < left_shoulder.y and
            abs(left_shoulder.y - left_wrist.y) > 0.2 and
            left_elbow.y > left_shoulder.y
        )

        right_hand_raised = (
            right_wrist.y < right_shoulder.y and
            abs(right_shoulder.y - right_wrist.y) > 0.2 and
            right_elbow.y > right_shoulder.y
        )

        return left_hand_raised or right_hand_raised

    except Exception as e:
        print(f"[ERROR] detect_hand_raised : {e}")
        return False


def detect_waving_hand(pose_results):
    try:
        if not pose_results or not pose_results.pose_landmarks:
            return False

        landmark = pose_results.pose_landmarks.landmark
        left_wrist_x = landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x
        left_wrist_y = landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        right_wrist_x = landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
        right_wrist_y = landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y

        left_shoulder_y = landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_shoulder_y = landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y

        # ✅ ถ้ามือทั้งสองข้างอยู่ต่ำกว่าหัวไหล่ ไม่ถือว่าโบกมือ
        if (left_wrist_y > left_shoulder_y) and (right_wrist_y > right_shoulder_y):
            return False

        # เก็บ history ตำแหน่งมือ
        wrist_history['left_x'].append(left_wrist_x)
        wrist_history['left_y'].append(left_wrist_y)
        wrist_history['right_x'].append(right_wrist_x)
        wrist_history['right_y'].append(right_wrist_y)

        def is_waving(history):
            if len(history) < 10:
                return False
            movement = max(history) - min(history)
            if movement < 0.15:
                return False

            # ตรวจจับจำนวนครั้งที่ "เปลี่ยนทิศทาง"
            direction_changes = 0
            for i in range(2, len(history)):
                diff1 = history[i-1] - history[i-2]      
                diff2 = history[i] - history[i-1]
                if diff1 * diff2 < 0:
                    direction_changes += 1
            return direction_changes >= 4

        waving_left = (
            is_waving(wrist_history['left_x']) or
            is_waving(wrist_history['left_y'])
        )
        waving_right = (
            is_waving(wrist_history['right_x']) or
            is_waving(wrist_history['right_y'])
        )

        return waving_left or waving_right

    except Exception as e:
        print(f"[ERROR] detect_waving_hand: {e}")
        return False


def is_patient_ok(pose_results, hands_results):
    try:
        gesture_ok = False

        if hands_results and hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                label = handedness.classification[0].label if handedness.classification else "Unknown"
                if detect_gesture_ok(hand_landmarks.landmark, label):
                    gesture_ok = True
                    break

        hand_raised = detect_hand_raised(pose_results)
        waving_hand = detect_waving_hand(pose_results)  

        # Debug log
        print(f"[DEBUG] is_patient_ok: gesture_ok={gesture_ok}, hand_raised={hand_raised}, waving_hand={waving_hand}")

        return gesture_ok or hand_raised or waving_hand  

    except Exception as e:
        print(f"[ERROR] is_patient_ok: {e}")
        return False


# ================================================ Core Detection ===============================================================


def detect_fall(sequence, pose_results, hands_results, index):
    global last_log_time, last_person_detected, fall_counters, last_debug_log_time, fall_start_time

    current_time = time.time()

    if pose_results.pose_landmarks:
        keypoints = []
        visibility = []
        for lm in pose_results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
            visibility.append(lm.visibility)
    else:
        keypoints = [0.0] * 99
        visibility = [0.0] * 33

    if sum(keypoints) == 0.0:
        fall_counters[index] = 0
        fall_start_time[index] = None
        if last_person_detected[index]:
            print(f"[LOG] Cam {index+1} | No people detected")
            last_person_detected[index] = False
        return "no_people"

    if not last_person_detected[index]:
        print(f"[LOG] Cam {index+1} | Person detected")
        last_person_detected[index] = True

    gesture_ok_flag = is_patient_ok(pose_results=pose_results, hands_results=hands_results)
    if gesture_ok_flag:
        print(f"[INFO] Cam {index+1} | Patient OK (OK gesture or hand raised)")

    important_indices = [23, 24, 25, 26]
    avg_visibility = np.mean([visibility[i] for i in important_indices])
    if avg_visibility < 0.3:
        print(f"[SKIP] Cam {index+1} | ความมั่นใจต่ำ (avg_visibility={avg_visibility:.2f}) → ข้าม")
        fall_counters[index] = 0
        fall_start_time[index] = None
        return False

    body_angle = calculate_body_angle_3d(keypoints)
    body_angle_y = calculate_body_angle_y_axis(keypoints)
    knee_angle = calculate_knee_angle_3d(keypoints)
    z_values = [keypoints[i * 3 + 2] for i in important_indices]
    z_variance = max(z_values) - min(z_values)

    hip_y = keypoints[23*3 + 1]
    ankle_y = keypoints[27*3 + 1]
    height_diff = ankle_y - hip_y

    standing = is_standing_pose(keypoints)

    if is_squat_pose(keypoints):
        print("NO FALLEN (นั่ง/นั่งยอง)")
        fall_counters[index] = 0
        fall_start_time[index] = None
        if random.random() < 0.05:
            save_training_data(keypoints, 0)
        return False

    ANGLE_Y_THRESHOLD = 45
    HEIGHT_DIFF_THRESHOLD = 0.3

    is_laying_down = body_angle_y is not None and body_angle_y > ANGLE_Y_THRESHOLD and height_diff > HEIGHT_DIFF_THRESHOLD

    if standing:
        print("NO FALLEN (กำลังยืน)")
        fall_counters[index] = 0
        fall_start_time[index] = None
        if random.random() < 0.05:
            save_training_data(keypoints, 0)
        return False

    if len(sequence) < 29:
        sequence.append(keypoints)
        return False

    input_seq = np.expand_dims(np.array(sequence + [keypoints]), axis=0)
    prediction = model.predict(input_seq, verbose=0)[0][0]
    sequence.append(keypoints)
    if len(sequence) > 29:
        sequence.pop(0)

    fallen_by_z_lock = is_fallen_by_locked_z (keypoints)
    is_flat = z_variance < 0.2

    if current_time - last_debug_log_time[index] > 2:
        print(f"[DEBUG] Cam {index+1} | pred={prediction:.3f} | body_angle={body_angle:.1f} | body_angle_y={body_angle_y:.1f} | knee_angle={knee_angle:.1f} | z_var={z_variance:.3f} | height_diff={height_diff:.3f}")
        last_debug_log_time[index] = current_time

    if fall_counters[index] >= FALL_CONFIRM_FRAMES:
        if gesture_ok_flag:
            print(f"[INFO] Cam {index+1} | OK Gesture detected after fall. Reset fall counter.")
            fall_counters[index] = 0
            fall_start_time[index] = None
            return False
        else:
            if fall_start_time[index] is None:
                fall_start_time[index] = current_time
            else:
                elapsed = current_time - fall_start_time[index]
                if elapsed > FALL_ALERT_DELAY:
                    print(f"[ALERT] Cam {index+1} | FALL DETECTED and no response for {FALL_ALERT_DELAY} seconds!")
                    fall_start_time[index] = current_time

            save_training_data(keypoints, 1)
            return True

    if prediction > 0.9 or (is_laying_down and prediction > 0.6):
        fall_counters[index] += 1
        print(f"FALL DETECTING... ({fall_counters[index]}/{FALL_CONFIRM_FRAMES})")
        if fall_counters[index] >= FALL_CONFIRM_FRAMES:
            print("FALL DETECTED ✅")
            save_training_data(keypoints, 1)
        return False

    if prediction > 0.7 and body_angle is not None and body_angle < 45 and is_flat:
        fall_counters[index] += 1
        print(f"FALL DETECTING (combined factors)... ({fall_counters[index]}/{FALL_CONFIRM_FRAMES})")
        if fall_counters[index] >= FALL_CONFIRM_FRAMES:
            print("FALL DETECTED ✅ (from body angle + flat Z)")
            save_training_data(keypoints, 1)
        return False

    print("NO FALLEN (ไม่เข้าเกณฑ์)")
    fall_counters[index] = 0
    fall_start_time[index] = None

    if random.random() < 0.05:
        save_training_data(keypoints, 0)

    return False

# ============================================================= Visualization ====================================================

def draw_landmarks(frame, pose_landmarks):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_drawing.draw_landmarks(
        frame, pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
    )

    keypoints = pose_landmarks.landmark
    important_indices = [23, 24, 25, 26]
    avg_visibility = np.mean([keypoints[i].visibility for i in important_indices])
    if avg_visibility < 0.5:
        return

    z_values = [keypoints[i].z for i in important_indices]
    z_variance = max(z_values) - min(z_values)

    try:
        shoulder = np.array([keypoints[11].x, keypoints[11].y, keypoints[11].z])
        hip = np.array([keypoints[23].x, keypoints[23].y, keypoints[23].z])
        knee = np.array([keypoints[25].x, keypoints[25].y, keypoints[25].z])
        vec1 = shoulder - hip
        vec2 = knee - hip
        angle = np.degrees(np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0)))
    except:
        angle = 90

    if z_variance < 0.6 and angle < 45:
        for i in important_indices:
            cx = int(keypoints[i].x * frame.shape[1])
            cy = int(keypoints[i].y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)
        cv2.putText(frame, "Z-FLAT!", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


# ======================================================= Threaded Stream Capture ===================================================


def capture_stream(index, source, stop_event, app_gui):
    global frames, sequence_list, fall_counters
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    hands_detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = None

    while not stop_event.is_set():
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"[WARN] กล้อง {index + 1} ไม่เชื่อมต่อ")
                with lock:
                    temp = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(temp, "Reconnecting...", (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if index < len(frames):
                        frames[index] = temp
                if cap:
                    cap.release()
                time.sleep(3)
                continue
            else:
                print(f"[INFO] กล้อง {index + 1} เชื่อมต่อแล้ว")

        try:
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] กล้อง {index + 1} หลุดการเชื่อมต่อ")
                cap.release()
                cap = None
                time.sleep(2)
                continue

            frame = cv2.resize(frame, (320, 240))
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ประมวลผลด้วย Mediapipe
            pose_results = pose.process(img_rgb)
            hands_results = hands_detector.process(img_rgb)

            # วาด pose landmarks
            if pose_results.pose_landmarks:
                draw_landmarks(frame, pose_results.pose_landmarks)

            # ตรวจจับท่าทาง OK
            ok_gesture = False
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    if detect_gesture_ok(hand_landmarks.landmark):
                        ok_gesture = True
                        break

            # ตรวจจับการล้ม
            if index < len(sequence_list):
                result = detect_fall(sequence_list[index], pose_results, hands_results, index)
            else:
                result = False

            if result is True:
                fall_counters[index] = 0  # Reset counter after confirmed
                fall_start_time[index] = None
                app_gui.fall_detected_flags[index] = True  # ✅ บอก GUI ให้ popup
                app_gui.popup_shown_flags[index] = False   # ✅ Reset popup flag
                cv2.putText(frame, " FALL DETECTED ", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                threading.Thread(
                    target=fall_response_process,
                    args=(source,),
                    daemon=True
                ).start()


            if ok_gesture:
                cv2.putText(frame, " OK Gesture Detected ", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # แสดงชื่อกล้อง
            cv2.rectangle(frame, (0, 0), (320, 25), (0, 0, 0), -1)
            cv2.putText(frame, f"Camera {index + 1}", (10, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # อัปเดตเฟรม
            with lock:
                if index < len(frames):
                    frames[index] = frame

            time.sleep(0.01)

        except Exception as e:
            print(f"[ERROR] กล้อง {index + 1} processing error: {e}")
            if cap:
                cap.release()
            cap = None
            time.sleep(3)


# ============================================  ตอบสนองเมื่อเกิดการล้ม =====================================================


def fall_response_process(source):
    print(f"[ALERT] FALL response triggered for source: {source}")
    # เพิ่มการแจ้งเตือนเข้าไปยังพวกแอปได้ทีหลัง 


# ========================================= Flask API: เพิ่มกล้อง ===================================================================


@app.route("/add_camera", methods=["POST"])
def add_camera():
    global stream_sources, frames, sequence_list, fall_counters
    global last_log_time, last_person_detected, last_debug_log_time, fall_start_time
    global capture_threads, capture_stop_flags
    global app_gui

    data = request.json
    new_ip = data.get('ip') if data else None
    if not new_ip:
        return jsonify({"error": "Missing 'ip' field"}), 400

    with lock:
        if len(stream_sources) >= MAX_CAMERAS:
            return jsonify({"error": f"Maximum number of cameras ({MAX_CAMERAS}) reached"}), 400
        if new_ip in stream_sources:
            return jsonify({"error": "IP camera already exists"}), 400

        stream_sources.append(new_ip)
        save_stream_sources(stream_sources)

        # เตรียม list
        frames.append(np.zeros((240, 320, 3), dtype=np.uint8))
        sequence_list.append([])
        fall_counters.append(0)
        last_log_time.append(0)
        last_person_detected.append(False)
        last_debug_log_time.append(0)
        fall_start_time.append(None)

        index = len(stream_sources) - 1

        # สร้าง flag หยุด thread และ thread ใหม่
        stop_flag_thread = threading.Event()
        capture_stop_flags[index] = stop_flag_thread
        t = threading.Thread(target=capture_stream, args=(index, new_ip, stop_flag_thread))
        t.daemon = True
        t.start()
        capture_threads[index] = t

    return jsonify({"message": f"Camera added at index {index}", "ip": new_ip}), 200

# ========================================= Flask API: Remove Camera ====================================================================


@app.route("/remove_camera", methods=["POST"])
def remove_camera():
    global stream_sources, frames, sequence_list, fall_counters
    global last_log_time, last_person_detected, last_debug_log_time, fall_start_time
    global capture_threads, capture_stop_flags

    data = request.json
    index = data.get("index")

    if index is None or not isinstance(index, int):
        return jsonify({"error": "Missing or invalid 'index' field"}), 400

    with lock:
        if index < 0 or index >= len(stream_sources):
            return jsonify({"error": f"Camera index {index} is out of range"}), 400

        # หยุด thread กล้องตัวนั้น
        if index in capture_stop_flags:
            capture_stop_flags[index].set()
            capture_threads[index].join(timeout=5)
            del capture_stop_flags[index]
            del capture_threads[index]

        # ลบข้อมูลกล้อง
        removed_ip = stream_sources.pop(index)
        frames.pop(index)
        sequence_list.pop(index)
        fall_counters.pop(index)
        last_log_time.pop(index)
        last_person_detected.pop(index)
        last_debug_log_time.pop(index)
        fall_start_time.pop(index)

        new_capture_threads = {}
        new_capture_stop_flags = {}
        for i, ip in enumerate(stream_sources):

            old_i = i if i < index else i + 1
            if old_i in capture_threads:
                new_capture_threads[i] = capture_threads[old_i]
                new_capture_stop_flags[i] = capture_stop_flags[old_i]
        capture_threads = new_capture_threads
        capture_stop_flags = new_capture_stop_flags

        save_stream_sources(stream_sources)

    return jsonify({"message": f"Camera at index {index} removed", "ip": removed_ip}), 200


# ======================================================== Main ===============================================================

def main():
    global stream_sources
    config_gui.open_config_gui()
    stream_sources = load_stream_sources()

    with lock:
        frames.clear()
        sequence_list.clear()
        fall_counters.clear()
        last_log_time.clear()
        last_person_detected.clear()
        last_debug_log_time.clear()
        fall_start_time.clear()

        for _ in range(len(stream_sources)):
            frames.append(np.zeros((240, 320, 3), dtype=np.uint8))
            sequence_list.append([])
            fall_counters.append(0)
            last_log_time.append(0)
            last_person_detected.append(False)
            last_debug_log_time.append(0)
            fall_start_time.append(None)

    root = tk.Tk()
    app_gui = CameraApp(root, frames, lock)  # สร้างก่อน

    threads = []
    stop_events = []

    for i, src in enumerate(stream_sources):
        stop_event = threading.Event()
        stop_events.append(stop_event)
        t = threading.Thread(target=capture_stream, args=(i, src, stop_event, app_gui))  # ส่ง app_gui เข้าไป
        t.daemon = True
        t.start()
        threads.append(t)

    def run_flask():
        app.run(host='0.0.0.0', port=5001)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    threading.Thread(target=retrain_model_thread, daemon=True).start()

    root.mainloop()

    for event in stop_events:
        event.set()

    for t in threads:
        t.join(timeout=3)

    print("[INFO] Program Stop Successfully")


if __name__ == "__main__":
    login_and_start()