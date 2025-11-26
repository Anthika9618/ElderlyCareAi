
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import os
import socket
import uuid
import xml.etree.ElementTree as ET
import traceback
import mysql.connector
import requests
import time
import cv2
import urllib.parse
import subprocess
import numpy as np
import tempfile
from urllib.parse import quote
from functools import wraps
import random
import mediapipe as mp
from tensorflow.keras.models import load_model
import threading
import queue

app = Flask(__name__)

selected_devices_global = []

active_readers = {} 
monitor_lock = threading.Lock()

# =================== BASIC AUTH ===================

USERNAME = "anthika_Forth"
PASSWORD = "Forth0263"


def check_auth(username, password):
    return username == USERNAME and password == PASSWORD


def authenticate():
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# =============================== #
#           Connect DB.           #
# =============================== #

def get_db_connection():
    conn = mysql.connector.connect(
        host="192.168.1.64",
        user="root",
        password="W3r4dm1n",
        database="blackbox"
    )
    return conn

# =============================== #
#      DB. Logging Functions      #
# =============================== #

def save_log_worker(cam_index, event_type, message):
    """ ฟังก์ชันทำงานเบื้องหลัง เพื่อบันทึกข้อมูลลง Database """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        sql = "INSERT INTO system_logs (camera_index, event_type, message) VALUES (%s, %s, %s)"
        cursor.execute(sql, (cam_index + 1, event_type, message))

        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ [DB. SAVED] Cam {cam_index + 1} : {event_type} - {message}")
    except Exception as e:
        print (f"❌ [DB. ERROR] {e}")

def log_event_to_db(cam_index, event_type, message):
    """ เรียกฟังก์ชันนี้เพื่อสั่งบันทึก (ไม่ต้องรอ response) """
    t = threading.Thread(target=save_log_worker, args=(cam_index, event_type, message))
    t.start()

# =========== Helper stubs (so code runs) ===========
# If you have your real implementations, replace these.
def save_training_data(keypoints, label):
    # stub: store training examples if you want
    pass

# =============================== #
#      class VideoFileReader      #
# =============================== #

class VideoFileReader(threading.Thread):
    def __init__(self, rtsp_url):
        threading.Thread.__init__(self)
        self.rtsp_url = rtsp_url
        self.frame_queue = queue.Queue(maxsize=1) 
        self.stop_event = threading.Event()
        self.cap = None

    def run(self):
        
        if "transport=tcp" not in self.rtsp_url:
            if '?' in self.rtsp_url:
                rtsp_url_tcp = self.rtsp_url + "&transport=tcp"
            else:
                rtsp_url_tcp = self.rtsp_url + "?transport=tcp"
        else:
            rtsp_url_tcp = self.rtsp_url
            
        rtsp_url_decoded = urllib.parse.unquote(rtsp_url_tcp)

        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                # เพิ่ม cv2.CAP_FFMPEG เพื่อความเสถียรของ RTSP
                self.cap = cv2.VideoCapture(rtsp_url_decoded, cv2.CAP_FFMPEG) 
                if not self.cap.isOpened():
                    time.sleep(3)
                    continue
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

            ret, frame = self.cap.read()
            if ret:
                
                try:
                    self.frame_queue.put_nowait(frame) 
                except queue.Full:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
            else:
                if self.cap:
                    self.cap.release()
                    self.cap = None 
                time.sleep(1)
        
        # ปล่อยทรัพยากรเมื่อ Thread หยุด
        if self.cap:
            self.cap.release()
            self.cap = None

    def stop(self):
        self.stop_event.set()
            
    def get_latest_frame(self):
        try:
            return True, self.frame_queue.get_nowait()
        except queue.Empty:
            return False, None

# =============================== #
#       Class PatientMonitor      #
# =============================== #

class PatientMonitor:
    def __init__(self, model_path, fall_confirm_frame=5, fall_alert_delay=3):
        # load model (will raise if path wrong)
        self.model = load_model(model_path)

        self.FALL_CONFIRM_FRAMES = fall_confirm_frame
        self.FALL_ALERT_DELAY = fall_alert_delay

        # state (per-camera indexes stored in dicts)
        self.sequence = {} 
        self.fall_counters = {}
        self.fall_start_time = {}
        self.last_debug_log_time = {}
        self.last_person_detected = {}
        self.help_counter_hand_raised = {}
        self.wrist_history = {} 
        self.wrist_help_wave_history = {}
        self.last_db_save_time = {}

        # mediapipe references
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        

        self.pose_processor = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands_processor = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def __del__(self):

        if hasattr(self, 'pose_processor'):
            self.pose_processor.close()
        if hasattr(self, 'hands_processor'):
            self.hands_processor.close()


    # ------------------ Helper Functions ------------------

    @staticmethod
    def vector_angle(vec1, vec2):
        # safe vector angle
        denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        if denom == 0:
            return None
        cos_theta = np.dot(vec1, vec2) / denom
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    @staticmethod
    def get_keypoint(keypoints, idx):
        return np.array([keypoints[idx * 3], keypoints[idx * 3 + 1], keypoints[idx * 3 + 2]])

    # ------------------ Pose & Angle Calculations ------------------
    def calculate_body_angle_3d(self, keypoints):
        try:
            shoulder = self.get_keypoint(keypoints, 11)
            hip = self.get_keypoint(keypoints, 23)
            knee = self.get_keypoint(keypoints, 25)
            return self.vector_angle(shoulder - hip, knee - hip)
        except Exception as e:
            print(f"[ERROR] calculate_body_angle_3d: {e}")
            return None

    def calculate_knee_angle_3d(self, keypoints):
        try:
            hip = self.get_keypoint(keypoints, 23)
            knee = self.get_keypoint(keypoints, 25)
            ankle = self.get_keypoint(keypoints, 27)
            return self.vector_angle(hip - knee, ankle - knee)
        except Exception as e:
            print(f"[ERROR] calculate_knee_angle_3d: {e}")
            return None

    def calculate_body_angle_y_axis(self, keypoints):
        try:
            shoulder = self.get_keypoint(keypoints, 11)
            hip = self.get_keypoint(keypoints, 23)
            vec = shoulder - hip
            vertical = np.array([0, -1, 0])
            return self.vector_angle(vec, vertical)
        except Exception as e:
            print(f"[ERROR] calculate_body_angle_y_axis: {e}")
            return None

    def is_standing_pose(self, keypoints, height_threshold=0.15, angle_threshold=40):
        try:
            hip_y = keypoints[23 * 3 + 1]
            ankle_y = keypoints[27 * 3 + 1]
            body_angle_y = self.calculate_body_angle_y_axis(keypoints)
            hip_above_ankle = (ankle_y - hip_y) > height_threshold
            is_straight = (body_angle_y is not None and body_angle_y < angle_threshold)
            return hip_above_ankle and is_straight
        except Exception as e:
            print(f"[ERROR] is_standing_pose: {e}")
            return False

    def is_squat_pose(self, keypoints):
        try:
            hip_y = keypoints[23 * 3 + 1]
            ankle_y = keypoints[27 * 3 + 1]
            return abs(hip_y - ankle_y) < 0.15
        except Exception as e:
            print(f"[ERROR] is_squat_pose:{e}")
            return False

    def is_fallen_by_locked_z(self, keypoints, lock_point_index=23, z_threshold=0.1):
        try:
            lock_z = keypoints[lock_point_index * 3 + 2]
            check_points = [0, 11, 12, 23, 24, 25, 26]
            return all(abs(keypoints[idx * 3 + 2] - lock_z) <= z_threshold for idx in check_points)
        except Exception as e:
            print(f"[ERROR] is_fallen_by_locked_z: {e}")
            return False

    # ------------------ Gesture / Hand Detection ------------------
    
    @staticmethod
    def detect_gesture_ok(landmarks):
        try:
            thumb = np.array([landmarks[4].x, landmarks[4].y])
            index = np.array([landmarks[8].x, landmarks[8].y])
            return np.linalg.norm(thumb - index) < 0.05
        except Exception as e:
            print(f"[ERROR] detect_gesture_ok: {e}")
            return False

    def detect_hand_raised(self, pose_results):
        try:
            if not pose_results or not getattr(pose_results, "pose_landmarks", None):
                return False
            lm = pose_results.pose_landmarks.landmark
            left_up = lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y < lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            right_up = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y < lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            return left_up or right_up
        except Exception as e:
            print(f"[ERROR] detect_hand_raised: {e}")
            return False

    def detect_waving_hand(self, pose_results, index=0):
        try:
            if not pose_results or not getattr(pose_results, "pose_landmarks", None):
                return False

            lm = pose_results.pose_landmarks.landmark

            # 🛡️ กฎใหม่: เช็คข้อมือเทียบไหล่ (ถ้ามือต่ำกว่าไหล่ ไม่นับ)
            right_wrist = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            if right_wrist.y > right_shoulder.y:
                self.wrist_history[index] = {'left_x': [], 'left_y': [], 'right_x': [], 'right_y': []}
                return False

            # ensure wrist_history exists
            wh = self.wrist_history.setdefault(index, {'left_x': [], 'left_y': [], 'right_x': [], 'right_y': []})

            # เก็บประวัติ (ใช้แค่แกน X ของมือขวาเป็นหลักสำหรับ Wave)
            wh['right_x'].append(right_wrist.x)
            if len(wh['right_x']) > 20: wh['right_x'].pop(0) # เพิ่ม Window Size
            
            history = wh['right_x']
            if len(history) < 20: return False

            # 🛡️ กฎใหม่: ต้องโบกกว้างๆ (> 0.2)
            movement = max(history) - min(history)
            if movement < 0.2: return False

            direction_changes = 0
            for i in range(2, len(history)):
                diff1 = history[i - 1] - history[i - 2]
                diff2 = history[i] - history[i - 1]
                if diff1 * diff2 < 0:
                    direction_changes += 1
            
            return direction_changes >= 5 # ต้องโบก 5 ครั้งขึ้นไป
        except Exception as e:
            print(f"[ERROR] detect_waving_hand: {e}")
            return False

    def is_patient_ok(self, pose_results, hands_results, index=0):
        try:
            # 1. มือจีบ 👌 (ท่านี้ยอมรับทุกกรณี เพราะตั้งใจทำ)
            gesture_ok = False
            if hands_results and getattr(hands_results, "multi_hand_landmarks", None):
                for hl in hands_results.multi_hand_landmarks:
                    if self.detect_gesture_ok(hl.landmark):
                        print(f"[INFO-OK] Cam {index + 1} | Patient OK (Hand Sign)")
                        
                        # ✅✅✅ [เพิ่มจุดที่ 1] บันทึกกรณีทำมือจีบ ✅✅✅
                        current_time = time.time()
                        # เช็คว่าผ่านไป 5 วิหรือยัง (Cooldown)
                        if current_time - self.last_db_save_time.get(f"{index}_OK", 0) > 5:
                            log_event_to_db(index, "OK", "Patient signaled OK (Hand Sign)")
                            self.last_db_save_time[f"{index}_OK"] = current_time
                        # ✅✅✅
                        
                        return True

            # 2. เช็คสรีระ (Body Check) - จุดสำคัญที่เพิ่มมา!
            if not pose_results or not getattr(pose_results, "pose_landmarks", None):
                return False

            landmarks = pose_results.pose_landmarks.landmark
            
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

            mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            mid_hip_y = (left_hip.y + right_hip.y) / 2
            mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            mid_hip_x = (left_hip.x + right_hip.x) / 2

            body_height = abs(mid_shoulder_y - mid_hip_y)
            body_width = abs(mid_shoulder_x - mid_hip_x)

            # ✅✅✅ จุดกรองคนนั่ง/ยืนทิ้ง ✅✅✅
            # ถ้าความสูง > 60% ของความกว้าง = ตัวตั้ง (นั่ง/ยืน/ยอง) -> ไม่รับ Wave/Raise
            if body_height > (body_width * 0.6):
                return False

            # 3. ถ้านอนราบแล้ว -> ถึงจะยอมรับการโบกมือ/ยกมือ
            hand_raised = self.detect_hand_raised(pose_results)
            waving = self.detect_waving_hand(pose_results, index)

            if hand_raised or waving:
                print(f"[INFO-OK] Cam {index + 1} | Patient OK from FLOOR")
                
                # ✅✅✅ [เพิ่มจุดที่ 2] บันทึกกรณีโบกมือตอนนอน ✅✅✅
                current_time = time.time()
                if current_time - self.last_db_save_time.get(f"{index}_OK", 0) > 5:
                    log_event_to_db(index, "OK", "Patient signaled OK (Wave from Floor)")
                    self.last_db_save_time[f"{index}_OK"] = current_time
                # ✅✅✅

                return True

            return False
        except Exception as e:
            print(f"[ERROR] is_patient_ok: {e}")
            return False

    # ------------------ Core Fall Detection ------------------

    def detect_fall(self, pose_results, hands_results, index):
        current_time = time.time()

        # 1. Init
        self.sequence.setdefault(index, [])
        self.fall_counters.setdefault(index, 0)
        self.fall_start_time.setdefault(index, None)
        self.last_debug_log_time.setdefault(index, 0)
        self.last_person_detected.setdefault(index, False)
        self.help_counter_hand_raised.setdefault(index, 0)
        self.wrist_history.setdefault(index, {'left_x': [], 'left_y': [], 'right_x': [], 'right_y': []})
        
        if not hasattr(self, 'last_db_save_time'): self.last_db_save_time = {}

        # 2. Extract Keypoints
        if pose_results and getattr(pose_results, "pose_landmarks", None):
            keypoints = []
            visibility = []
            for lm in pose_results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
                visibility.append(getattr(lm, 'visibility', 0.0))
        else:
            self.fall_counters[index] = 0
            self.fall_start_time[index] = None
            if self.last_person_detected[index]:
                print(f"[LOG-CAM] Cam {index + 1} | No people detected")
                self.last_person_detected[index] = False
            return False, False

        self.last_person_detected[index] = True

        # 3. Check OK
        if self.is_patient_ok(pose_results, hands_results, index):
            self.fall_counters[index] = 0
            self.fall_start_time[index] = None
            return False, False

        # 4. Check Visibility
        important_indices = [23, 24, 25, 26]
        vis_list = [visibility[i] for i in important_indices if i < len(visibility)]
        avg_visibility = np.mean(vis_list) if vis_list else 0.0
        if avg_visibility < 0.3:
            self.fall_counters[index] = 0
            self.fall_start_time[index] = None
            return False, False

        # 5. สรีระ: นอนราบไหม?
        lm = pose_results.pose_landmarks.landmark
        h = abs(((lm[11].y + lm[12].y)/2) - ((lm[23].y + lm[24].y)/2))
        w = abs(((lm[11].x + lm[12].x)/2) - ((lm[23].x + lm[24].x)/2))
        is_laying_down = (w > h)

        # =========================================================
        # ✅ ส่วนที่แก้ไข: เปลี่ยนชื่อ Log เป็น "NO FALLEN" ✅
        # =========================================================

        # 6.1 เช็ค Standing (ยืน)
        if not is_laying_down and self.is_standing_pose(keypoints):
            print("NO FALLEN (standing)")
            
            # บันทึกเป็น "NO FALLEN" (แต่ใช้ Key จับเวลา _STANDING เพื่อไม่ให้ตีกับ Squat)
            if current_time - self.last_db_save_time.get(f"{index}_STANDING", 0) > 10:
                log_event_to_db(index, "NO FALLEN", "Person is Standing/Walking")
                self.last_db_save_time[f"{index}_STANDING"] = current_time
            
            self.fall_counters[index] = 0
            return False, self.detect_help_request(pose_results, index)

        # 6.2 เช็ค Squat (นั่งยอง)
        if not is_laying_down and self.is_squat_pose(keypoints):
            print("NO FALLEN (squat)")
            
            # บันทึกเป็น "NO FALLEN" (แต่ใช้ Key จับเวลา _SQUATTING เพื่อให้บันทึกทันทีที่เปลี่ยนท่า)
            if current_time - self.last_db_save_time.get(f"{index}_SQUATTING", 0) > 10:
                log_event_to_db(index, "NO FALLEN", "Person is Squatting")
                self.last_db_save_time[f"{index}_SQUATTING"] = current_time

            self.fall_counters[index] = 0
            return False, self.detect_help_request(pose_results, index)

        # =========================================================

        # 7. เตรียมข้อมูลเข้า AI
        seq_buf = self.sequence[index]
        if len(seq_buf) < 29:
            seq_buf.append(keypoints)
            self.sequence[index] = seq_buf
            return False, self.detect_help_request(pose_results, index)

        input_seq = np.expand_dims(np.array(seq_buf + [keypoints]), axis=0)
        try:
            prediction = float(self.model.predict(input_seq, verbose=0)[0][0])
        except:
            prediction = 0.0

        seq_buf.append(keypoints)
        if len(seq_buf) > 29: seq_buf.pop(0)
        self.sequence[index] = seq_buf

        # 8. Fall Detection Logic
        z_values = [keypoints[i * 3 + 2] for i in important_indices]
        z_variance = max(z_values) - min(z_values) if z_values else 0.0
        is_flat = z_variance < 0.2
        
        fall_detected = False

        # --- เงื่อนไขล้ม (FALL) ---
        if prediction > 0.9 or (is_laying_down and prediction > 0.6):
            self.fall_counters[index] += 1
            print(f"FALL DETECTING... ({self.fall_counters[index]}/{self.FALL_CONFIRM_FRAMES})")
            
            if self.fall_counters[index] >= self.FALL_CONFIRM_FRAMES:
                print("FALL DETECTED ✅")
                
                # ✅ [DB] บันทึก FALL
                if current_time - self.last_db_save_time.get(f"{index}_FALL", 0) > 5:
                    log_event_to_db(index, "FALL", "Fall detected via AI Model")
                    self.last_db_save_time[f"{index}_FALL"] = current_time

        elif prediction > 0.7 and is_laying_down and is_flat:
            self.fall_counters[index] += 1
            print(f"FALL DETECTING (Angle)... ({self.fall_counters[index]}/{self.FALL_CONFIRM_FRAMES})")
            
            if self.fall_counters[index] >= self.FALL_CONFIRM_FRAMES:
                print("FALL DETECTED ✅ (Angle+Flat)")
                
                # ✅ [DB] บันทึก FALL
                if current_time - self.last_db_save_time.get(f"{index}_FALL", 0) > 5:
                    log_event_to_db(index, "FALL", "Fall detected via Angle+Flat")
                    self.last_db_save_time[f"{index}_FALL"] = current_time

        else:
            self.fall_counters[index] = 0
            self.fall_start_time[index] = None

        # 9. Alert Logic
        if self.fall_counters[index] >= self.FALL_CONFIRM_FRAMES:
            if self.fall_start_time[index] is None:
                self.fall_start_time[index] = current_time
            else:
                if current_time - self.fall_start_time[index] > self.FALL_ALERT_DELAY:
                    print(f"[ALERT] Cam {index + 1} | FALL CONFIRMED")
                    fall_detected = True

        help_requested = self.detect_help_request(pose_results, index)
        return fall_detected, help_requested

    # help gestures (copied/converted from your functions)
    def detect_fast_hand_wave(self, pose_results, index, window_size=15, min_swings=2):
        if not pose_results or not getattr(pose_results, "pose_landmarks", None):
            self.wrist_help_wave_history.setdefault(index, [])
            self.wrist_help_wave_history[index] = []
            return False

        landmarks = pose_results.pose_landmarks.landmark
        right_wrist_x = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x

        lst = self.wrist_help_wave_history.setdefault(index, [])
        lst.append(right_wrist_x)
        if len(lst) > window_size:
            lst.pop(0)
        self.wrist_help_wave_history[index] = lst

        if len(lst) < window_size:
            return False

        swings = 0
        direction = 0
        for i in range(1, len(lst)):
            diff = lst[i] - lst[i - 1]
            if diff > 0 and direction != 1:
                direction = 1
                swings += 1
            elif diff < 0 and direction != -1:
                direction = -1
                swings += 1
        return swings >= min_swings

    def is_near_shoulder(self, wrist, shoulder, max_dist=0.12):
        dist = ((wrist.x - shoulder.x) ** 2 + (wrist.y - shoulder.y) ** 2) ** 0.5
        return dist < max_dist


    def detect_help_request(self, pose_results, index):
        try:
            if not pose_results or not getattr(pose_results, "pose_landmarks", None):
                self.help_counter_hand_raised.setdefault(index, 0)
                self.help_counter_hand_raised[index] = 0
                return False

            landmarks = pose_results.pose_landmarks.landmark

            # =======================================================
            # 🛡️ กฎเหล็ก: เช็คสรีระก่อน (Body Orientation Check)
            # =======================================================
            # ใช้ไหล่ และ สะโพก เพื่อดูแนวแกนกระดูกสันหลัง
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

            # หาจุดกึ่งกลางไหล่ และ กึ่งกลางสะโพก
            mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            mid_hip_x = (left_hip.x + right_hip.x) / 2
            mid_hip_y = (left_hip.y + right_hip.y) / 2

            # คำนวณระยะห่างแนวตั้ง (Y) และ แนวนอน (X) ของลำตัว
            body_height = abs(mid_shoulder_y - mid_hip_y)
            body_width = abs(mid_shoulder_x - mid_hip_x)

            # ถ้า "ความสูง" มากกว่า "ความกว้าง" = ตัวตั้ง (ยืน/นั่ง/นั่งยอง)
            # ให้ถือว่ายังไม่ล้ม -> ไม่รับคำสั่งขอความช่วยเหลือ
            if body_height > body_width:
                # เคลียร์ค่าการโบกมือทิ้ง เพื่อไม่ให้จำค่าสะสมตอนยืน
                self.wrist_help_wave_history[index] = [] 
                self.help_counter_hand_raised[index] = 0
                # print(f"[IGNORE] Cam {index+1}: Person is upright (Sitting/Standing), ignoring gestures.")
                return False
            
            # =======================================================
            # ถ้าผ่านจุดบนมาได้ แปลว่า "ตัวนอนราบ" (Horizontal) แล้ว
            # เริ่มกระบวนการตรวจจับท่าทางขอความช่วยเหลือ
            # =======================================================

            def is_valid(lm, min_vis=0.6):
                return (getattr(lm, 'visibility', 0.0) > min_vis and 0 <= lm.x <= 1 and 0 <= lm.y <= 1)

            # 1. ตรวจจับการโบกมือ (Wave)
            # ใช้ฟังก์ชัน detect_fast_hand_wave (ที่คุณแก้ไปก่อนหน้านี้)
            fast_wave = self.detect_fast_hand_wave(pose_results, index)

            # 2. ตรวจจับท่าเจ็บหน้าอก / กุมหัว
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            # Logic จับอก/กุมหัว (ปรับให้เหมาะกับท่านอน)
            # ใช้ระยะห่าง (Distance) แทนการดูความสูงต่ำ (Y-axis) เพราะตอนนอนแกน Y มันเพี้ยน
            chest_help = False
            if is_valid(left_wrist) and is_valid(right_wrist) and is_valid(left_shoulder) and is_valid(right_shoulder):
                # เช็คว่ามืออยู่ใกล้ไหล่/อก หรือไม่ (ระยะห่าง < 0.15)
                chest_help = (
                    (self.is_near_shoulder(left_wrist, left_shoulder, 0.15) or self.is_near_shoulder(left_wrist, right_shoulder, 0.15)) or
                    (self.is_near_shoulder(right_wrist, right_shoulder, 0.15) or self.is_near_shoulder(right_wrist, left_shoulder, 0.15))
                )

            nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
            head_hold = False
            if is_valid(nose):
                # ถ้ามือกุมแถวๆ จมูก/ตา (บนพื้น)
                dist_l_nose = ((left_wrist.x - nose.x)**2 + (left_wrist.y - nose.y)**2)**0.5
                dist_r_nose = ((right_wrist.x - nose.x)**2 + (right_wrist.y - nose.y)**2)**0.5
                head_hold = (dist_l_nose < 0.15) or (dist_r_nose < 0.15)

            # สรุปผล
            if fast_wave or chest_help or head_hold:
                print(f"[INFO-HELP] Cam {index + 1} : 🆘 Help gesture detected on FLOOR! (Wave={fast_wave}, Chest={chest_help}, Head={head_hold})")
                
                # ✅✅✅ [ส่วนที่เพิ่ม] บันทึกลง Database ✅✅✅
                current_time = time.time()
                # เช็คว่าผ่านไป 5 วินาทีหรือยัง (เพื่อไม่ให้บันทึกซ้ำรัวๆ)
                if current_time - self.last_db_save_time.get(f"{index}_HELP", 0) > 5:
                    # สร้างข้อความ Log ตาม Logic เดิมของคุณ
                    msg = f"Help gesture detected on FLOOR! (Wave={fast_wave}, Chest={chest_help}, Head={head_hold})"
                    log_event_to_db(index, "HELP", msg)
                    self.last_db_save_time[f"{index}_HELP"] = current_time
                # ✅✅✅ [จบส่วนที่เพิ่ม] ✅✅✅

                self.help_counter_hand_raised[index] = 0
                return True

            self.help_counter_hand_raised[index] = 0
            return False
        except Exception as e:
            print(f"[ERROR] detect_help_request: {e}")
            return False

    # ------------------ Visualization ------------------

    def draw_landmarks(self, frame, pose_landmarks):
        try:
            self.mp_drawing.draw_landmarks(
                frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            keypoints = pose_landmarks.landmark
            important_indices = [23, 24, 25, 26]
            avg_visibility = np.mean([getattr(keypoints[i], 'visibility', 0.0) for i in important_indices])
            if avg_visibility < 0.5:
                return

            z_values = [getattr(keypoints[i], 'z', 0.0) for i in important_indices]
            z_variance = max(z_values) - min(z_values)

            try:
                shoulder = np.array([keypoints[11].x, keypoints[11].y, keypoints[11].z])
                hip = np.array([keypoints[23].x, keypoints[23].y, keypoints[23].z])
                knee = np.array([keypoints[25].x, keypoints[25].y, keypoints[25].z])
                vec1 = shoulder - hip
                vec2 = knee - hip
                denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                if denom == 0:
                    angle = 90
                else:
                    angle = np.degrees(np.arccos(np.clip(np.dot(vec1, vec2) / denom, -1.0, 1.0)))
            except Exception:
                angle = 90

            if z_variance < 0.6 and angle < 45:
                for i in important_indices:
                    cx = int(keypoints[i].x * frame.shape[1])
                    cy = int(keypoints[i].y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)
                cv2.putText(frame, "Z-FLAT!", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        except Exception as e:
            # don't crash visualization
            print(f"[ERROR] draw_landmarks: {e}")


# =============================== #
#             Model init          #
# =============================== #

model_path = r"/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/models/falldetect_bi_lstm_testmodel.h5"
monitor = PatientMonitor(model_path)

# =============================== #
#        ONVIF Scan Functions     #
# (kept your implementation but minor cleanups)
# =============================== #
def build_probe(types):
    message_id = f"uuid:{uuid.uuid4()}"
    types_str = " ".join(types)
    probe_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
            xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <e:Header>
    <w:MessageID>{message_id}</w:MessageID>
    <w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
    <w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
  </e:Header>
  <e:Body>
    <d:Probe>
      <d:Types>{types_str}</d:Types>
    </d:Probe>
  </e:Body>
</e:Envelope>"""
    return probe_xml


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def get_subnet_prefix(local_ip):
    parts = local_ip.split('.')
    if len(parts) == 4:
        return '.'.join(parts[:3]) + '.'
    return '192.168.1.'


def scan_onvif_device_multicast(device_types, timeout=3):
    MULTICAST_ADDR = "239.255.255.250"
    PORT = 3702
    results = []
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.settimeout(timeout)
        local_ip = get_local_ip()
        sock.bind((local_ip, 0))
        probe = build_probe(device_types)
        sock.sendto(probe.encode('utf-8'), (MULTICAST_ADDR, PORT))
        start = time.time()
        while True:
            try:
                data, addr = sock.recvfrom(65507)
                ip = addr[0]
                device = {"ip": ip, "xaddr": "", "scopes": [], "brand": "", "model": ""}
                xml_data = data.decode(errors='ignore').lstrip('\ufeff')
                try:
                    root = ET.fromstring(xml_data)
                except Exception:
                    continue
                for x in root.iter():
                    tag = x.tag.split('}')[-1]
                    if tag == 'XAddrs' and x.text:
                        device["xaddr"] = x.text
                    elif tag in ['Scope', 'Scopes'] and x.text:
                        scopes_list = x.text.split()
                        device["scopes"].extend(scopes_list)
                        for scope in scopes_list:
                            if "onvif://www.onvif.org/name/" in scope:
                                device["brand"] = scope.split("/")[-1]
                            elif "onvif://www.onvif.org/hardware/" in scope:
                                device["model"] = scope.split("/")[-1]
                if not any(d["ip"] == device["ip"] for d in results):
                    results.append(device)
            except socket.timeout:
                break
            if time.time() - start > timeout + 1:
                break
    except Exception as e:
        print("multicast scan error:", e)
    finally:
        try:
            sock.close()
        except:
            pass
    return results


def scan_onvif_device_iploop(subnet_prefix="192.168.1.", port=80, timeout_per=0.4):
    results = []
    session = requests.Session()
    headers = {'User-Agent': 'onvif-scan/1.0'}
    for i in range(1, 255):
        ip = f"{subnet_prefix}{i}"
        url = f"http://{ip}:{port}/onvif/device_service"
        try:
            r = session.get(url, headers=headers, timeout=timeout_per)
            if r.status_code == 200:
                device = {"ip": ip, "xaddr": url, "scopes": [], "brand": "", "model": ""}
                try:
                    root = ET.fromstring(r.text)
                    for x in root.iter():
                        tag = x.tag.split('}')[-1]
                        if tag in ['Scopes', 'Scope'] and x.text:
                            scopes_list = x.text.split()
                            device["scopes"].extend(scopes_list)
                            for scope in scopes_list:
                                if "onvif://www.onvif.org/name/" in scope:
                                    device["brand"] = scope.split("/")[-1]
                                elif "onvif://www.onvif.org/hardware/" in scope:
                                    device["model"] = scope.split("/")[-1]
                except Exception:
                    pass
                results.append(device)
        except requests.exceptions.RequestException:
            continue
    return results


def scan_onvif_device(device_types, preferred_port=80):
    results = scan_onvif_device_multicast(device_types, timeout=2)
    if results:
        return results
    local_ip = get_local_ip()
    subnet_prefix = get_subnet_prefix(local_ip)
    results = scan_onvif_device_iploop(subnet_prefix=subnet_prefix, port=preferred_port, timeout_per=0.35)
    return results


def get_camera_type_by_brand(brand):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM camera_type WHERE brand = %s", (brand,))
    info = cursor.fetchone()
    cursor.close()
    conn.close()
    return info


def build_camera_urls(ip, username, password, brand_info):
    urls = {}
    link = brand_info.get('link', '')
    if brand_info.get('port_rtsp'):
        rtsp_url = f"rtsp://{username}:{password}@{ip}:{brand_info['port_rtsp']}{link}"
        urls['rtsp'] = urllib.parse.unquote(rtsp_url)
    if brand_info.get('port_hrrp'):
        http_url = f"http://{username}:{password}@{ip}:{brand_info['port_hrrp']}{link}"
        urls['hrrp'] = urllib.parse.unquote(http_url)
    if brand_info.get('port_onvif'):
        urls['onvif'] = f"http://{ip}:{brand_info['port_onvif']}/onvif/device_service"
    return urls


def check_camera_stream(urls):
    for key, url in urls.items():
        try:
            if key == 'rtsp':
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    cap.release()
                    continue
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return key, url
            else:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    return key, url
        except Exception:
            continue
    return None, None


def add_rtsp_url_to_devices(devices):
    for device in devices:
        for port_type, info in device.get('ports', {}).items():
            if port_type == 'rtsp' and info['status'] == 'Stream Success':
                device['rtsp_url'] = info['url']
    return devices


# =========================================================
# FIXED FUNCTION: Video streaming with AI Detection
# =========================================================

def generate_frames_with_ai(rtsp_url, index):
    global active_readers, monitor

    # 1. เริ่มต้น Reader (ตัวอ่านวิดีโอ)
    reader = active_readers.get(rtsp_url)
    if reader is None or not reader.is_alive():
        reader = VideoFileReader(rtsp_url)
        reader.start()
        active_readers[rtsp_url] = reader
        time.sleep(2) 
    
    frame_count = 0
    
    while True:
        try:
            # 2. ดึงภาพล่าสุด
            ret, frame = reader.get_latest_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # 3. เตรียมภาพ
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            # 4. ให้ AI ทำงาน (ข้ามเฟรมบ้างเพื่อลดภาระ)
            frame_count += 1
            run_ai = (frame_count % 3 == 0) 
            
            with monitor_lock:
                pose_results = monitor.pose_processor.process(rgb_frame)
                hands_results = monitor.hands_processor.process(rgb_frame)
            
            fall_detected = False
            help_requested = False

            if run_ai and pose_results.pose_landmarks:
                fall_detected, help_requested = monitor.detect_fall(pose_results, hands_results, index)

            # 5. วาดภาพ
            rgb_frame.flags.writeable = True
            annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # ✅✅✅ จุดที่แก้ไข ERROR draw_landmarks ✅✅✅
            if pose_results.pose_landmarks:
                # ลบ status_color ออก เพื่อให้จำนวนตัวแปรถูกต้อง
                monitor.draw_landmarks(annotated_frame, pose_results.pose_landmarks)

            # ส่วนแสดงข้อความสถานะ (ย้ายมาทำตรงนี้แทน)
            status_text = "OK"
            status_color = (0, 255, 0)
            if fall_detected:
                status_text = "FALL DETECTED!"
                status_color = (0, 0, 255)
            elif help_requested:
                status_text = "HELP REQUESTED"
                status_color = (0, 255, 255)

            cv2.putText(annotated_frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            # 6. ส่งภาพ
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
        except Exception as e:
            print(f"Stream Error: {e}")
            time.sleep(1)


# =============================== #
#             ROUTES              #
# =============================== #

@app.route('/')
def home():
    return render_template('users.html')


@app.route('/room_config', methods=['GET', 'POST'])
def room_config():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM room_type")
    room_types = cursor.fetchall()
    success_message = None
    if request.method == 'POST':
        room_type_id = request.form.get('room_type_id')
        detail = request.form.get('detail')
        cursor.execute("SELECT room_type_name, room_type_name_en FROM room_type WHERE room_type_id = %s", (room_type_id,))
        rt = cursor.fetchone()
        if rt:
            room_name = rt['room_type_name']
            room_name_en = rt['room_type_name_en']
            cursor.execute("""
                INSERT INTO room (room_name, room_name_en, detail)
                VALUES (%s, %s, %s)
            """, (room_name, room_name_en, detail))
            conn.commit()
            success_message = "Save Successfully ✅"
    cursor.execute("SELECT * FROM room")
    rooms = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template(
        'room_config.html',
        room_types=room_types,
        rooms=rooms,
        success_message=success_message
    )


@app.route('/users', methods=['GET'])
def get_users():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(users)


@app.route('/add_user', methods=['POST'])
def add_user():
    data = request.get_json()
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = "INSERT INTO users (username, password, full_name, role) VALUES (%s, %s, %s, %s)"
    cursor.execute(sql, (data['username'], data['password'], data['full_name'], data['role']))
    conn.commit()
    cursor.close()
    return jsonify({'message': 'User added successfully'})


@app.route('/update_user/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    data = request.get_json()
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = "UPDATE users SET username=%s, password=%s, full_name=%s, role=%s WHERE user_id=%s"
    cursor.execute(sql, (data['username'], data['password'], data['full_name'], data['role'], user_id))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({'message': 'User updated successfully'})


@app.route('/delete_user/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE user_id=%s AND role !='admin'", (user_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'User deleted successfully'})


@app.route('/scan_onvif', methods=['GET', 'POST'])
def scan_onvif():
    results = []
    selected = "all"
    default_onvif_port = 80
    if request.method == 'POST':
        selected = request.form.get('device_type', 'all')
        try:
            port_from_form = int(request.form.get('onvif_port', default_onvif_port))
        except Exception:
            port_from_form = default_onvif_port
        if selected == "camera":
            types = ["dn:NetworkVideoTransmitter", "dn:Device"]
        elif selected == "nvr":
            types = ["dn:NetworkVideoRecorder", "dn:Device"]
        else:
            types = ["dn:NetworkVideoTransmitter", "dn:NetworkVideoRecorder", "dn:Device"]
        results = scan_onvif_device(types, preferred_port=port_from_form)
    return render_template('index.html', results=results, selected=selected)


@app.route('/device_info', methods=['POST'])
def device_info():
    global selected_devices_global
    selected_devices = request.form.getlist('selected_devices')
    devices = []
    for dev_str in selected_devices:
        try:
            ip, brand, model, xaddr = dev_str.split('|')
        except Exception:
            parts = dev_str.split('|')
            ip = parts[0] if parts else ''
            brand = parts[1] if len(parts) > 1 else ''
            model = parts[2] if len(parts) > 2 else ''
            xaddr = parts[3] if len(parts) > 3 else ''
        devices.append({
            "ip": ip,
            "brand": brand,
            "model": model,
            "xaddr": xaddr
        })
    selected_devices_global = devices
    return render_template('device_info.html', selected_devices=devices)


@app.route('/save_device', methods=['POST'])
def save_device():
    global selected_devices_global
    conn = get_db_connection()
    cursor = conn.cursor()
    saved_devices = []
    for i, device in enumerate(selected_devices_global, start=1):
        device_name = request.form.get(f'device_name_{i}')
        location = request.form.get(f'location_{i}')
        username = request.form.get(f'username_{i}')
        password = request.form.get(f'password_{i}')
        cursor.execute("""
            INSERT INTO camera_info
            (device_name, location, brand, model, username, password, ipAddress, url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            device_name,
            location,
            device['brand'],
            device['model'],
            username,
            password,
            device['ip'],
            device['xaddr']
        ))
        conn.commit()
        saved_devices.append({
            "device_name": device_name,
            "location": location,
            "brand": device['brand'],
            "model": device['model'],
            "username": username,
            "password": password,
            "ipAddress": device['ip'],
            "url": device['xaddr']
        })
    cursor.close()
    conn.close()
    return jsonify({"status": "success", "message": "บันทึกสำเร็จ"})


@app.route('/config_room', methods=['GET', 'POST'])
def config_room():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM room_type")
    room_types = cursor.fetchall()
    success_message = None
    if request.method == 'POST':
        room_type_id = request.form.get('room_type_id')
        detail = request.form.get('detail')
        cursor.execute("SELECT room_type_name, room_type_name_en FROM room_type WHERE room_type_id = %s", (room_type_id,))
        rt = cursor.fetchone()
        if rt:
            room_name = rt['room_type_name']
            room_name_en = rt['room_type_name_en']
            cursor.execute("""
                INSERT INTO room (room_name, room_name_en, detail)
                VALUES (%s, %s, %s)
            """, (room_name, room_name_en, detail))
            conn.commit()
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                cursor.close()
                conn.close()
                return jsonify({'success': True, 'message': 'Room saved successfully!'})
            success_message = "Save Successfully ✅"
    cursor.execute("SELECT * FROM room")
    rooms = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('room_config.html', room_types=room_types, rooms=rooms, success_message=success_message)


@app.route('/delete_room/<int:room_id>', methods=['POST'])
def delete_room(room_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM room WHERE room_id = %s", (room_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'success': True, 'message': 'Room deleted successfully'})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})


@app.route('/check_camera_stream', methods=['GET', 'POST'])
def check_camera_stream_route():
    import urllib.parse
    global selected_devices_global
    result = None
    manual_result = None
    all_failed = False
    if request.method == 'POST':
        ip = request.form.get('ip')
        username = request.form.get('username')
        password = request.form.get('password')
        brand = request.form.get('brand')
        manual_url = request.form.get('manual_url')
        if manual_url:
            manual_url_decoded = urllib.parse.unquote(manual_url)
            cap = cv2.VideoCapture(manual_url_decoded)
            if cap.isOpened():
                ret, frame = cap.read()
                manual_result = "Stream Success" if ret else "Stream Failed"
                selected_devices_global = [{'device_name': 'Manual Stream', 'rtsp_url': manual_url_decoded}]
            else:
                manual_result = "Stream Failed"
                selected_devices_global = []
            cap.release()
        else:
            brand_info = get_camera_type_by_brand(brand)
            if not brand_info:
                return render_template('camera_checker.html', error_message='Brand ไม่พบในฐานข้อมูล')
            urls = build_camera_urls(ip, username, password, brand_info)
            port_type, working_url = check_camera_stream(urls)
            result = {}
            for key, url in urls.items():
                if port_type == key:
                    status = 'Stream Success'
                else:
                    status = 'Stream Failed'
                url_decoded = urllib.parse.unquote(url)
                result[key] = {'port': brand_info.get(f'port_{key}', ''), 'url': url_decoded, 'status': status}
            all_failed = all(v['status'] == 'Stream Failed' for v in result.values())
            selected_devices_global = []
            if port_type == 'rtsp' and working_url:
                selected_devices_global.append({
                    'device_name': f"{brand} ({ip})",
                    'rtsp_url': urllib.parse.unquote(working_url)
                })
    return render_template(
        'camera_checker.html',
        result=result,
        manual_result=manual_result,
        all_failed=all_failed,
        selected_devices_global=selected_devices_global
    )


@app.route('/camera_checker')
def camera_checker():
    return render_template('camera_checker.html')
 

@app.route('/video_feed/<int:cam_index>') 
@requires_auth
def video_feed(cam_index): 
    global selected_devices_global
    

    if not selected_devices_global or cam_index >= len(selected_devices_global):
        return "No RTSP URL available", 404
        
    device = selected_devices_global[cam_index]
    if 'rtsp_url' not in device:
        return "RTSP URL not found in device data", 404
        
    rtsp_url = device['rtsp_url']
    

    return Response(

        generate_frames_with_ai(rtsp_url, cam_index),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/video_feed_playback')
def video_feed_playback():
    from datetime import datetime, timedelta
    ip = request.args.get('ip')
    user = request.args.get('user')
    pwd = request.args.get('pwd')
    channel = request.args.get('channel', '1')
    subtype = request.args.get('subtype', '1')
    date = request.args.get('date')      # YYYY-MM-DD
    start = request.args.get('start')    # HH:MM
    end = request.args.get('end')        # HH:MM

    if not all([ip, user, pwd, date, start, end]):
        return "Missing parameters", 400

    def format_time(date, time_str):
        d = date.split("-")
        t = time_str.split(":")
        return f"{d[0]}_{d[1]}_{d[2]}_{t[0]}_{t[1]}_00"

    starttime = format_time(date, start)
    endtime = format_time(date, end)

    user_enc = quote(user)
    pwd_enc = quote(pwd)

    rtsp_url = (
        f"rtsp://{user_enc}:{pwd_enc}@{ip}:554/"
        f"cam/playback?channel={channel}&subtype={subtype}"
        f"&starttime={starttime}&endtime={endtime}"
    )
    print("Playback URL:", rtsp_url)

    def generate():
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print("Cannot open RTSP playback")
            return
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            ret2, buffer = cv2.imencode('.jpg', frame)
            if not ret2:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() +
                   b'\r\n')
        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/playback')
def playback():
    return render_template('playback.html')


@app.route('/live_stream')
def live_stream():
    device_index = request.args.get('device_index', 0, type=int)
    if device_index >= len(selected_devices_global):
        return "Device not found", 404
    device = selected_devices_global[device_index]
    stream_url = device.get('rtsp_url')
    device_name = device.get('device_name')
    return render_template('live_stream.html', stream_url=stream_url, device_name=device_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
