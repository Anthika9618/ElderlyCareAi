import cv2
import numpy as np
import threading
import tensorflow as tf
import mediapipe as mp
import time

# GPU config
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"✅ ใช้ GPU ได้: {physical_devices}")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except Exception as e:
        print(f"⚠️ Warning setting memory growth: {e}")
else:
    print("❌ ไม่พบ GPU")

# โหลดโมเดล
model = tf.keras.models.load_model("/mnt/c/ElderlyCareSystem/models/falldetect_bi_lstm_testmodel.h5")
# กล้องทั้งหมด
stream_sources = [
    'rtsp://admin:admin@123@192.168.0.5:554/Streaming/Channels/101',
    1,
    'rtsp://admin:admin@123@192.168.0.5:554/Streaming/Channels/201',
    'rtsp://admin:admin@123@192.168.0.5:554/Streaming/Channels/401'
]

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
FALL_ALERT_DELAY = 30  # วินาทีที่ต้องนิ่งก่อนแจ้งเตือน

# ===== Helper Function =====
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

# ตรวจจับท่า OK (นิ้วโป้งกับนิ้วชี้ใกล้กัน)
def detect_gesture_ok(landmarks):
    try:
        right_thumb = [landmarks[4].x, landmarks[4].y]
        right_index = [landmarks[8].x, landmarks[8].y]
        dist = np.linalg.norm(np.array(right_thumb) - np.array(right_index))
        return dist < 0.05
    except Exception:
        return False

# ===== Core Detection =====
def detect_fall(sequence, results, index, gesture_ok_flag):
    global last_log_time, last_person_detected, fall_counters, last_debug_log_time, fall_start_time

    current_time = time.time()

    if results.pose_landmarks:
        keypoints = []
        visibility = []
        for lm in results.pose_landmarks.landmark:
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

    # เช็คท่ายืน
    standing = is_standing_pose(keypoints)

    # เช็คท่านั่ง
    if is_squat_pose(keypoints):
        print("NO FALLEN (นั่ง/นั่งยอง)")
        fall_counters[index] = 0
        fall_start_time[index] = None
        return False

    # เช็คว่าล้มจากเงื่อนไข body angle y และ height_diff
    is_laying_down = False
    ANGLE_Y_THRESHOLD = 45
    HEIGHT_DIFF_THRESHOLD = 0.3
    if body_angle_y is not None and body_angle_y > ANGLE_Y_THRESHOLD and height_diff > HEIGHT_DIFF_THRESHOLD:
        is_laying_down = True

    # ถ้าท่ายืน ให้ถือว่าไม่ล้ม
    if standing:
        print("NO FALLEN (กำลังยืน)")
        fall_counters[index] = 0
        fall_start_time[index] = None
        return False

    # รอจนมีเฟรมพอ
    if len(sequence) < 29:
        sequence.append(keypoints)
        return False

    input_seq = np.expand_dims(np.array(sequence + [keypoints]), axis=0)
    prediction = model.predict(input_seq, verbose=0)[0][0]
    sequence.append(keypoints)
    if len(sequence) > 29:
        sequence.pop(0)

    fallen_by_z_lock = is_fallen_by_locked_z(keypoints)
    is_flat = z_variance < 0.2

    # ลดความถี่ log เหลือพิมพ์ทุก 2 วินาที
    if current_time - last_debug_log_time[index] > 2:
        print(f"[DEBUG] Cam {index+1} | pred={prediction:.3f} | body_angle={body_angle:.1f} | body_angle_y={body_angle_y:.1f} | knee_angle={knee_angle:.1f} | z_var={z_variance:.3f} | height_diff={height_diff:.3f}")
        last_debug_log_time[index] = current_time

    # ถ้าล้มแล้วตรวจเจอ OK gesture ให้รีเซ็ตสถานะล้ม
    if fall_counters[index] >= FALL_CONFIRM_FRAMES:
        if gesture_ok_flag:
            print(f"[INFO] Cam {index+1} | OK Gesture detected after fall. Reset fall counter.")
            fall_counters[index] = 0
            fall_start_time[index] = None
            return False
        else:
            # เช็คเวลาว่านิ่งเกิน 30 วินาทีหรือยัง
            if fall_start_time[index] is None:
                fall_start_time[index] = current_time
            else:
                elapsed = current_time - fall_start_time[index]
                if elapsed > FALL_ALERT_DELAY:
                    print(f"[ALERT] Cam {index+1} | FALL DETECTED and no response for {FALL_ALERT_DELAY} seconds!")
                    # เพิ่มการแจ้งเตือนอื่นๆที่นี่ เช่น ส่ง SMS หรือ สัญญาณเตือน
                    # ... 
                    # เพื่อไม่ให้ alert ซ้ำๆ ให้รีเซ็ตเวลา (หรือปรับ logic ตามต้องการ)
                    fall_start_time[index] = current_time  # หรือจะตั้งเป็น None เพื่อหยุดแจ้งเตือน
            return True

    # ตรวจจับล้ม
    if prediction > 0.9 or (is_laying_down and prediction > 0.6):
        fall_counters[index] += 1
        print(f"FALL DETECTING... ({fall_counters[index]}/{FALL_CONFIRM_FRAMES})")
        if fall_counters[index] >= FALL_CONFIRM_FRAMES:
            print("FALL DETECTED ✅")
        return False

    if prediction > 0.7 and body_angle is not None and body_angle < 45 and is_flat:
        fall_counters[index] += 1
        print(f"FALL DETECTING (combined factors)... ({fall_counters[index]}/{FALL_CONFIRM_FRAMES})")
        if fall_counters[index] >= FALL_CONFIRM_FRAMES:
            print("FALL DETECTED ✅ (from body angle + flat Z)")
        return False

    print("NO FALLEN (ไม่เข้าเกณฑ์)")
    fall_counters[index] = 0
    fall_start_time[index] = None
    return False

# ===== Visualization =====
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

# ===== Threaded Stream Capture =====
def capture_stream(index, source):
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
    
    while True:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[WARN] กล้อง {index+1} ไม่เชื่อมต่อ")
            with lock:
                temp = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(temp, "Reconnecting...", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                frames[index] = temp
            time.sleep(3)
            continue

        print(f"[INFO] กล้อง {index+1} เชื่อมต่อแล้ว")
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] กล้อง {index+1} หลุดการเชื่อมต่อ")
                cap.release()
                break

            frame = cv2.resize(frame, (320, 240))
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            hands_results = hands_detector.process(img_rgb)

            # วาด skeleton
            if results.pose_landmarks:
                draw_landmarks(frame, results.pose_landmarks)

            # ตรวจจับ gesture OK
            ok_gesture = False
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    if detect_gesture_ok(hand_landmarks.landmark):
                        ok_gesture = True
                        break

            # ตรวจจับล้ม
            result = detect_fall(sequence_list[index], results, index, ok_gesture)
            if result is True:
                cv2.putText(frame, " FALL DETECTED ", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # ถ้าต้องการแสดงเตือนนิ่งเกิน 30 วินาที สามารถเพิ่มตรงนี้ได้ด้วย

            # แสดงสถานะ OK gesture
            if ok_gesture:
                cv2.putText(frame, " OK Gesture Detected ", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.rectangle(frame, (0, 0), (320, 25), (0, 0, 0), -1)
            cv2.putText(frame, f"Camera {index + 1}", (10, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            with lock:
                frames[index] = frame
        time.sleep(2)

def display_grid():
    while True:
        with lock:
            current = frames.copy()
        row1 = np.hstack(current[0:2])
        row2 = np.hstack(current[2:4])
        grid = np.vstack([row1, row2])
        final = grid

        cv2.imshow("Fall Detection - Real-time", final)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

# ===== Start Threads =====
for i, src in enumerate(stream_sources):
    t = threading.Thread(target=capture_stream, args=(i, src))
    t.daemon = True
    t.start()

display_grid()
