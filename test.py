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
import setting_page
import tkinter as tk
import register_gui
import add_camera_gui
import mediapipe as mp
import tensorflow as tf
from tkinter import ttk
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
MAX_CAMERAS = 12
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

    load_dotenv(ENV_PATH, override=True) 



# ================================== ประกาศ Global history  ของตำแหน่งข้อมือ ===============================


wrist_history = {
    'left_x': deque(maxlen=15),
    'left_y': deque(maxlen=15),
    'right_x': deque(maxlen=15),
    'right_y': deque(maxlen=15)
}


# ================================================= Config ===============================================

MAX_ATTEMPTS = 3
LOCK_TIME = 50

# ================================================= State ================================================

attempts = 0
is_locked = False
lock_end_time = 0

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

model = tf.keras.models.load_model("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/models/falldetect_bi_lstm_testmodel.h5")

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
FALL_ALERT_DELAY = 20  

# ✅ เพิ่มใหม่ สำหรับ Help Request
wrist_help_wave_history = [[] for _ in range(len(stream_sources))]
help_counter_hand_raised = [0 for _ in range(len(stream_sources))]


# ========================================================= Login GUI FORM =================================================

def login_and_start():
    def verify_login():
        global attempts, is_locked, lock_end_time

        if is_locked:
            remaining = int(lock_end_time - time.time())
            if remaining > 0:
                # สร้าง popup แสดงเวลาล็อค
                popup = tk.Toplevel()
                popup.title("Locked")
                popup.geometry("320x140")
                popup.configure(bg="#f0f2f5")
                popup.attributes("-topmost", True)

            
                try:
                    icon_img = Image.open(
                        "/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/hourglass.png"
                    ).resize((40, 40), Image.LANCZOS)
                    icon_photo = ImageTk.PhotoImage(icon_img)
                    icon_label = tk.Label(popup, image=icon_photo, bg="#f0f2f5")
                    icon_label.image = icon_photo
                    icon_label.pack(pady=(10, 5))
                except:
                    tk.Label(popup, text="⏳", font=("Arial", 24),
                            bg="#f0f2f5", fg="#d9534f").pack(pady=(10, 5))

                
                countdown_label = tk.Label(popup, font=("Helvetica", 11, "bold"),
                                        bg="#f0f2f5", fg="#262627")
                countdown_label.pack(pady=(0, 10))

                def update_countdown():
                    remaining_time = int(lock_end_time - time.time())
                    if remaining_time > 0:
                        
                        try:
                            icon_img = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/time.png")
                            icon_img = icon_img.resize((20, 20), Image.LANCZOS)
                            icon_photo = ImageTk.PhotoImage(icon_img)
                            countdown_label.config(
                                image=icon_photo,
                                text=f"Too many failed attempts.\nPlease wait {remaining_time} seconds.",
                                compound="left",
                                font=("Helvetica", 11, "bold"),
                                fg="#262627",
                                bg="#f0f2f5",
                                padx=10,
                                pady=10,
                                anchor="w",
                                justify="left"
                            )
                            countdown_label.image = icon_photo  
                        except Exception as e:
                            countdown_label.config(
                                text=f"Too many failed attempts.\nPlease wait {remaining_time} seconds.",
                                font=("Helvetica", 11, "bold"),
                                fg="#262627",
                                bg="#f0f2f5",
                                padx=10,
                                pady=10,
                                anchor="w",
                                justify="left"
                            )

                        popup.after(1000, update_countdown)
                    else:
                        popup.destroy()
                        global is_locked, attempts
                        is_locked = False
                        attempts = 0

                update_countdown()
                return
            else:
                is_locked = False
                attempts = 0

        username = username_entry.get()
        password = password_entry.get()
        username_env, password_env = get_credentials()

        if username == username_env and password == password_env:
            # สร้าง popup loading
            popup = tk.Toplevel()
            popup.title("Logging in...")
            popup.geometry("300x150")
            popup.resizable(False, False)
            popup.configure(bg="#f0f2f5")
            popup.attributes("-topmost", True)

            msg_label = tk.Label(popup, text="Logging in...", font=("Helvetica", 12, "bold"), bg="#f0f2f5")
            msg_label.pack(pady=(20, 10))

            spin_img = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/deadline.png")
            spin_img = spin_img.resize((50, 50), Image.LANCZOS)
            spin_photo = ImageTk.PhotoImage(spin_img)

            spin_label = tk.Label(popup, image=spin_photo, bg="#f0f2f5")
            spin_label.image = spin_photo
            spin_label.pack(pady=5)

            duration = 5000  
            start_time = time.time()

            def wait_and_show_success():
                elapsed = (time.time() - start_time) * 1000
                if elapsed < duration:
                    popup.after(100, wait_and_show_success)
                else:
                
                    try:
                        success_img = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/check-mark.png")
                        success_img = success_img.resize((50, 50), Image.LANCZOS)
                        success_photo = ImageTk.PhotoImage(success_img)
                        spin_label.config(image=success_photo)
                        spin_label.image = success_photo
                    except:
                        spin_label.config(text="✅", font=("Arial", 28))

                    msg_label.config(text="Login Successful")
                    popup.after(900, lambda: [popup.destroy(), login_win.destroy(), main()])

            wait_and_show_success()


    def reset_password():
        # Custom popup
        confirm_win = tk.Toplevel()
        confirm_win.title("Confirm Reset")
        confirm_win.geometry("350x180")
        confirm_win.resizable(False, False)
        confirm_win.configure(bg="#f0f2f5")
        confirm_win.attributes("-topmost", True)

    
        try:
            icon_img = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/question.png").resize((30, 30), Image.LANCZOS)
            icon_photo = ImageTk.PhotoImage(icon_img)
            icon_label = tk.Label(confirm_win, image=icon_photo, bg="#f0f2f5")
            icon_label.image = icon_photo
            icon_label.pack(pady=(15, 5))
        except:
            tk.Label(confirm_win, text="❓", font=("Arial", 30), bg="#f0f2f5").pack(pady=(15, 5))

        # ===============================  ข้อความ  ================================

        tk.Label(confirm_win, text="Do you want to reset your password?", 
                font=("Helvetica", 12, "bold"), bg="#f0f2f5").pack(pady=(0, 15))

        # =========================  ตัวแปรเก็บค่า confirm  ===========================

        result = {"value": False}

        # =========================  ปุ่ม Yes / No  =================================
        def on_yes():
            result["value"] = True
            confirm_win.destroy()

        def on_no():
            result["value"] = False
            confirm_win.destroy()

        btn_frame = tk.Frame(confirm_win, bg="#f0f2f5")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Yes", command=on_yes,
                bg="#4a90e2", fg="white", font=("Helvetica", 11, "bold"),
                activebackground="#357ABD", cursor="hand2", width=10).pack(side="left", padx=10)

        tk.Button(btn_frame, text="No", command=on_no,
                bg="#e74c3c", fg="white", font=("Helvetica", 11, "bold"),
                activebackground="#c0392b", cursor="hand2", width=10).pack(side="left", padx=10)

        confirm_win.wait_window()  

        if result["value"]:

            new_pass_win = Toplevel()
            new_pass_win.title("Reset Password")
            new_pass_win.geometry("350x170")
            new_pass_win.resizable(False, False)
            new_pass_win.configure(bg="#f0f2f5")
            new_pass_win.attributes("-topmost", True)

            # ============================================  ไอคอน  ========================================================

            try:
                icon_img = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/pass.png").resize((40, 40), Image.LANCZOS)
                icon_photo = ImageTk.PhotoImage(icon_img)
                tk.Label(new_pass_win, image=icon_photo, bg="#f0f2f5").pack(pady=(10, 5))
            except Exception as e:
                print("Icon not found:", e)
                tk.Label(new_pass_win, text="🔒", font=("Arial", 28), bg="#f0f2f5").pack(pady=(10, 5))

            # ============================================= ข้อความ  ========================================================

            tk.Label(new_pass_win, text="Enter your new password:", font=("Helvetica", 12, "bold"), bg="#f0f2f5").pack(pady=(0,5))

            # ===========================================  Entry password  ==================================================

            new_pass_entry = tk.Entry(new_pass_win, font=("Helvetica", 12), width=25, show="*", bd=1, relief="solid")
            new_pass_entry.pack(pady=(0,10))

            # =========================================== ปุ่ม Submit  ========================================================

            def submit_pass():
                new_pass = new_pass_entry.get()
                if new_pass:
                    update_env_password(new_pass)

                    # ปิดหน้าต่างกรอกรหัสทันที
                    new_pass_win.destroy()

                    # =================================== Success popup =====================================================

                    popup = Toplevel()
                    popup.title("Success")
                    popup.geometry("300x150")
                    popup.resizable(False, False)
                    popup.configure(bg="#f0f2f5")
                    popup.attributes("-topmost", True)
                    popup.attributes("-alpha", 0.0)
                    popup.protocol("WM_DELETE_WINDOW", lambda: None)  

                    # ===================================== fade-in ========================================================

                    def fade_in(step=0.05):
                        alpha = popup.attributes("-alpha") + step
                        if alpha < 1.0:
                            popup.attributes("-alpha", alpha)
                            popup.after(30, fade_in)
                        else:
                            popup.attributes("-alpha", 1.0)
                    fade_in()

                    # ======================================= icon =========================================================
                    
                    try:
                        icon_img2 = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/check.png").resize((40, 40), Image.LANCZOS)
                        icon_photo2 = ImageTk.PhotoImage(icon_img2)
                        tk.Label(popup, image=icon_photo2, bg="#f0f2f5").pack(pady=(15,5))
                        popup.icon_photo = icon_photo2
                    except:
                        tk.Label(popup, text="✅", font=("Arial", 28), bg="#f0f2f5").pack(pady=(15,5))

                    tk.Label(popup, text="Password reset successfully!", 
                            font=("Helvetica", 12, "bold"), bg="#f0f2f5").pack(pady=(0,10))

                    # ==================================== progress bar ====================================================

                    progress = ttk.Progressbar(popup, orient="horizontal", length=220, mode="determinate")
                    progress.pack(pady=(0,10))
                    progress['value'] = 0

                    def update_progress():
                        progress['value'] += 5
                        if progress['value'] < 100:
                            popup.after(230, update_progress)
                        else:
                            popup.destroy()

                    update_progress()
                    new_pass_win.destroy()

            tk.Button(new_pass_win, text="Submit", command=submit_pass, bg="#4a90e2", fg="white",
                    font=("Helvetica", 12, "bold"), width=12, cursor="hand2").pack()

            new_pass_win.grab_set() 
            new_pass_win.wait_window()


    login_win = tk.Tk()
    login_win.title("Welcome Admin Login")
    w, h = 600, 600
    screen_w = login_win.winfo_screenwidth()
    screen_h = login_win.winfo_screenheight()
    x = (screen_w // 2) - (w // 2)
    y = (screen_h // 2) - (h // 2)
    login_win.geometry(f"{w}x{h}+{x}+{y}")
    login_win.resizable(False, False)
    login_win.configure(bg="#f0f2f5")

    # Header (โลโก้ + ชื่อระบบ)
    header_frame = tk.Frame(login_win, bg="#f0f2f5")
    header_frame.pack(pady=15)

     # ===============================================  Logo Forth  =====================================================

    try:
        logo_img = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/elderlogo.png")
        logo_img = logo_img.resize((95, 95), Image.LANCZOS)
        logo = ImageTk.PhotoImage(logo_img)
        logo_label = tk.Label(header_frame, image=logo, bg="#f0f2f5")
        logo_label.image = logo
        logo_label.grid(row=0, column=0, padx=(5, 15), sticky="w")
    except Exception as e:
        print("Logo not found:", e)

    title_label = tk.Label(header_frame, text="FORTH INTELLIGENT CARE",
                           bg="#f0f2f5", fg="#262627",font=("Arial", 18, "bold"))
    title_label.grid(row=0, column=1, sticky="w", padx=(5, 0))


    # ===============================================  Logo Admin Login =====================================================

    try:
        admin_logo_img = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/worker.png")
        admin_logo_img = admin_logo_img.resize((50, 50), Image.LANCZOS)
        admin_logo = ImageTk.PhotoImage(admin_logo_img)
        admin_label = tk.Label(
            header_frame,
            text="Admin Login",
            image=admin_logo,
            compound="left",  
            bg="#f0f2f5",
            fg="#262627",
            font=("Arial", 15, "bold")
        )
        admin_label.image = admin_logo

    except Exception as e:
        print("Admin logo not found:", e)
        admin_label = tk.Label(
            header_frame,
            text="Admin Login",
            bg="#f0f2f5",
            fg="#262627",
            font=("Arial", 12, "bold")
        )

    admin_label.grid(row=1, column=0, columnspan=2, sticky="n", padx=(5,0))

    header_frame.grid_columnconfigure(0, weight=1)
    header_frame.grid_columnconfigure(1, weight=1)

    # Form
    form_frame = tk.Frame(login_win, bg="white", bd=0, relief="flat", padx=20, pady=20)
    form_frame.pack(pady=10)

    # Username
    tk.Label(form_frame, text="Username", bg="white", font=("Helvetica", 12)).pack(anchor="w", pady=(5, 0))

    username_frame = tk.Frame(form_frame, bg="#f5f6fa")  
    username_frame.pack(fill="x", pady=(0,10))

    # Icon
    user_icon = ImageTk.PhotoImage(Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/userlock.png").resize((25, 25), Image.LANCZOS))
    user_label = tk.Label(username_frame, image=user_icon, bg="#f5f6fa")
    user_label.pack(side="left", padx=5)

    # Entry อยู่ใน Frame เดียวกับ Icon
    username_entry = tk.Entry(username_frame, font=("Helvetica", 12), bd=0, bg="#f5f6fa", relief="flat", fg="grey")
    username_entry.pack(side="left", fill="x", expand=True, ipady=8)
    username_entry.insert(0, "Enter your username")

    # Placeholder events for username
    def on_focus_in_username(event):
        if username_entry.get() == "Enter your username":
            username_entry.delete(0, "end")
            username_entry.config(fg="black")

    def on_focus_out_username(event):
        if username_entry.get() == "":
            username_entry.insert(0, "Enter your username")
            username_entry.config(fg="grey")

    username_entry.bind("<FocusIn>", on_focus_in_username)
    username_entry.bind("<FocusOut>", on_focus_out_username)


    # Password
    tk.Label(form_frame, text="Password", bg="white", font=("Helvetica", 12)).pack(anchor="w", pady=(5, 0))

    pass_frame = tk.Frame(form_frame, bg="white")
    pass_frame.pack(fill="x", pady=(0, 15))


    pass_icon_img = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/passlock.png").resize((25, 25), Image.LANCZOS)
    pass_icon = ImageTk.PhotoImage(pass_icon_img)
    pass_label = tk.Label(pass_frame, image=pass_icon, bg="#f5f6fa")
    pass_label.pack(side="left", padx=5)

    password_entry = tk.Entry(pass_frame, font=("Helvetica", 12), width=27, bd=0, bg="#f5f6fa", relief="flat", fg="grey")
    password_entry.pack(side="left", ipady=8, fill="x", expand=True)
    password_entry.insert(0, "Enter your password")

    # Placeholder events for password
    def on_focus_in_password(event):
        if password_entry.get() == "Enter your password":
            password_entry.delete(0, "end")
            password_entry.config(fg="black", show="*")

    def on_focus_out_password(event):
        if password_entry.get() == "":
            password_entry.insert(0, "Enter your password")
            password_entry.config(fg="grey", show="")

    password_entry.bind("<FocusIn>", on_focus_in_password)
    password_entry.bind("<FocusOut>", on_focus_out_password)

    # Eye icon
    eye_open_img = ImageTk.PhotoImage(Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/eye_open.png").resize((20, 20), Image.LANCZOS))
    eye_closed_img = ImageTk.PhotoImage(Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/eye_closed.png").resize((20, 20), Image.LANCZOS))

    def toggle_password():
        if password_entry.cget("show") == "":
            password_entry.config(show="*")
            eye_btn.config(image=eye_closed_img)
        else:
            password_entry.config(show="")
            eye_btn.config(image=eye_open_img)

    eye_btn = tk.Button(pass_frame, image=eye_closed_img, bd=0, bg="white", cursor="hand2", command=toggle_password)
    eye_btn.pack(side="left", padx=5)

    # Login button
    def on_enter(e): e.widget['bg'] = '#5aa0f2'
    def on_leave(e): e.widget['bg'] = '#4a90e2'

    login_icon = ImageTk.PhotoImage(
        Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/key.png")
        .resize((25, 25), Image.LANCZOS) 
    )

    login_btn = tk.Button(
        login_win,
        text="Login",
        command=verify_login,
        font=("Helvetica", 15, "bold"),
        bg="#4a90e2",
        fg="white",
        activebackground="#357ABD",
        activeforeground="white",
        bd=0,
        relief="flat",
        cursor="hand2",
        image=login_icon,
        compound="left",   
        padx=20,           
    pady=10           
    )

    login_btn.image = login_icon
    login_btn.pack(pady=15) 
    login_btn.bind("<Enter>", on_enter)
    login_btn.bind("<Leave>", on_leave)


    # ========================== Load icons ==========================

    def load_icon(path, size=(30, 30)):
        try:
            img = Image.open(path)
            img = img.resize(size, Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Error loading {path}:{e}")
            return None

    forgot_icon = load_icon("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/forgotpassword.png")
    register_icon = load_icon("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/regis.png")

    # ========================== Forgot Password ==========================


    def on_enter_forgot(e): e.widget.config(bg="#fdecea")
    def on_leave_forgot(e): e.widget.config(bg="#f0f2f5")

    forgot_btn = tk.Button(
        login_win,
        text="Forgot Password ?",
        command=reset_password,
        font=("Helvetica", 11, "bold"),
        fg="#e74c3c",
        bg="#f0f2f5",
        activebackground="#fdecea",
        activeforeground="#c0392b",
        cursor="hand2",
        image=forgot_icon,
        compound="left",
        bd=0,
        relief="flat",
        padx=10,
        pady=5
    )

    forgot_btn.image = forgot_icon
    forgot_btn.pack(pady=5, ipadx=15, padx=50)
    forgot_btn.bind("<Enter>", on_enter_forgot)
    forgot_btn.bind("<Leave>", on_leave_forgot)
    forgot_btn.config(highlightthickness=0, borderwidth=0, relief="ridge", font=("Helvetica", 12, "bold"))


    # ========================== Register Button ==========================

    def on_enter_register(e): e.widget.config(bg="#e8f8f5")
    def on_leave_register(e): e.widget.config(bg="#f0f2f5")

    register_btn = tk.Button(
        login_win,
        text="Register",
        command=lambda: __import__('register_gui').register_user(),
        font=("Helvetica", 12, "bold"),
        fg="#27ae60",
        bg="#f0f2f5",
        activebackground="#e8f8f5",
        activeforeground="#2ecc71",
        cursor="hand2",
        image=register_icon,
        compound="left",
        bd=0,
        relief="flat",
        padx=10,
        pady=5
    )
    register_btn.image = register_icon
    register_btn.pack(pady=5, ipadx=15, padx=50)
    register_btn.bind("<Enter>", on_enter_register)
    register_btn.bind("<Leave>", on_leave_register)
    register_btn.config(highlightthickness=0, borderwidth=0, relief="ridge")


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
    def __init__(self, master, frames, lock, pose_results=None):
        self.master = master
        self.frames = frames
        self.lock = lock
        self.pose_results = pose_results if pose_results is not None else []
        self.video_labels = []
        self.update_interval = 30  

        self.master.title("Fall Detection - Real-time")
        self.master.geometry("1080x720")

        self.fall_detected_flags = [False] * len(self.frames)
        self.popup_shown_flags = [False] * len(self.frames)

        # ================= Scrollable canvas =================

        self.canvas = tk.Canvas(master)
        self.scrollbar = ttk.Scrollbar(master, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # ================= โหลดไอคอน =================

        add_img = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/addstream.png")
        add_img = add_img.resize((30, 30), Image.LANCZOS)
        self.add_icon = ImageTk.PhotoImage(add_img)

        remove_img = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/removestream.png")
        remove_img = remove_img.resize((30, 30), Image.LANCZOS)
        self.remove_icon = ImageTk.PhotoImage(remove_img)

        setting_img = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/settingpage.png")
        setting_img = setting_img.resize((30, 30), Image.LANCZOS)
        self.setting_icon = ImageTk.PhotoImage(setting_img)

        # ================= Toolbar Buttons =================

        self.button_frame = ttk.Frame(self.scrollable_frame)
        self.button_frame.grid(row=0, column=0, columnspan=3, sticky="w", pady=10, padx=10)

        #Add Cam
        self.btn_add_camera = ttk.Button(
            self.button_frame, 
            image=self.add_icon, 
            text="Add Camera", 
            compound="left",   
            command=self.add_camera_dialog
        )
        self.btn_add_camera.pack(side="left", padx=(0, 5))

        #Remove cam
        self.btn_remove_camera = ttk.Button(
            self.button_frame, 
            image=self.remove_icon, 
            text="Remove Camera", 
            compound="left",
            command=self.remove_camera_dialog
        )
        self.btn_remove_camera.pack(side="left", padx=(5, 0))

        #Setting
        self.btn_setting = ttk.Button (
            self.button_frame,
            image=self.setting_icon,
            text="Setting",
            compound="left",
            command=self.open_settings
        )
        self.btn_setting.pack(side="left", padx=(5, 0))


        self.create_video_labels()
        self.update_videos()

    # ================= Create dynamic camera labels =================

    def create_video_labels(self):
        for lbl_frame in self.video_labels:
            lbl_frame["frame"].destroy()
        self.video_labels.clear()

        with self.lock:
            num_cams = len(self.frames)

        cols = 3  # number of columns per row
        for i in range(num_cams):
            frame = ttk.Frame(self.scrollable_frame, relief="ridge", borderwidth=2)
            frame.grid(row=(i // cols) + 1, column=i % cols, padx=10, pady=10, sticky="nsew")

            lbl_num = ttk.Label(frame, text=f"Camera {i+1}", font=("Arial", 10, "bold"))
            lbl_num.pack(anchor="n")

            lbl_video = ttk.Label(frame)
            lbl_video.pack()
            
            lbl_status = ttk.Label(frame, text="Status: Normal", foreground="green")
            lbl_status.pack(anchor="s", pady=(5, 0))

            self.video_labels.append({
                "frame": frame,
                "video": lbl_video,
                "status": lbl_status
            })

    def sync_with_backend(self):
        """ซิงค์ list ทั้งหมดให้มีขนาดตรงกับจำนวนกล้องใน backend"""
        try:
            url = "http://localhost:5001/list_cameras"
            resp = requests.get(url)
            if resp.status_code == 200:
                cameras = resp.json().get("cameras", [])
                num_cams = len(cameras)

                while len(self.frames) < num_cams:
                    self.frames.append(None)
                while len(self.fall_detected_flags) < num_cams:
                    self.fall_detected_flags.append(False)
                while len(self.popup_shown_flags) < num_cams:
                    self.popup_shown_flags.append(False)
                while len(self.pose_results) < num_cams:
                    self.pose_results.append(None)

                # ถ้ากล้องถูกลบออก
                if len(self.frames) > num_cams:
                    self.frames = self.frames[:num_cams]
                if len(self.fall_detected_flags) > num_cams:
                    self.fall_detected_flags = self.fall_detected_flags[:num_cams]
                if len(self.popup_shown_flags) > num_cams:
                    self.popup_shown_flags = self.popup_shown_flags[:num_cams]
                if len(self.pose_results) > num_cams:
                    self.pose_results = self.pose_results[:num_cams]

                self.create_video_labels()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to sync cameras:\n{e}")


    # ================= Update video frames =================

    def update_videos(self):
        with self.lock:
            frames_copy = self.frames.copy()

        # ซิงค์ length ของ fall_detected_flags และ popup_shown_flags
        while len(self.fall_detected_flags) < len(frames_copy):
            self.fall_detected_flags.append(False)
        while len(self.popup_shown_flags) < len(frames_copy):
            self.popup_shown_flags.append(False)

        for i, frame in enumerate(frames_copy):
            if isinstance(frame, np.ndarray) and frame.size != 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((320, 240))
                imgtk = ImageTk.PhotoImage(image=img)

                if i < len(self.video_labels):
                    self.video_labels[i]["video"].imgtk = imgtk
                    self.video_labels[i]["video"].config(image=imgtk)

            # Update Fall & Help status
            if self.fall_detected_flags[i]:
                if not self.popup_shown_flags[i]:
                    self.popup_shown_flags[i] = True
                    self.show_fall_popup(i)
                self.video_labels[i]["status"].config(text="Status: FALL!", foreground="red")
            elif i < len(self.pose_results) and detect_help_request(self.pose_results[i], i):
                if not self.popup_shown_flags[i]:
                    self.popup_shown_flags[i] = True
                    self.show_help_popup(i)
                self.video_labels[i]["status"].config(text="Status: HELP!", foreground="orange")
            else:
                self.video_labels[i]["status"].config(text="Status: Normal", foreground="green")

        # Refresh labels if number of frames changed
        if len(frames_copy) != len(self.video_labels):
            self.create_video_labels()

        self.master.after(self.update_interval, self.update_videos)


    # ================= Add Camera =================

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
                self.sync_with_backend()   # <== เรียก sync
            else:
                error = resp.json().get("error", "Unknown error")
                messagebox.showerror("Error", f"Failed to add camera:\n{error}")
        except Exception as e:
            messagebox.showerror("Error", f"Error connecting to API:\n{e}")


    # ================= Remove Camera =================

    def remove_camera_dialog(self):
        with self.lock:
            num_cams = len(self.frames)

        if num_cams == 0:
            messagebox.showwarning("No Camera", "No cameras to remove.")
            return

        cam_order = simpledialog.askinteger("Remove Camera", "Enter camera number:")
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
                messagebox.showinfo("Success", f"Camera removed successfully")
                self.sync_with_backend()   # <== เรียก sync
            else:
                error = resp.json().get("error", "Unknown error")
                messagebox.showerror("Error", f"Failed to remove camera:\n{error}")
        except Exception as e:
            messagebox.showerror("Error", f"Error connecting to API:\n{e}")


    # ===================== Setiing =======================

    def open_settings(self):
        setting_page.open_settings_window(
            self.master,
            on_logout=self.back_to_login,
            on_back=self.back_to_camera
        )

    def back_to_login(self):
        self.master.destroy()
        print("กลับไปหน้า Login แล้ว")

    def back_to_camera(self):
        print("กลับมายัง CameraApp")


    # ================= Popup for Fall Detection =================
    def show_fall_popup(self, index):
        popup = Toplevel(self.master)
        popup.title("Notification")
        popup.geometry("300x100")
        popup.configure(bg="white")

        label = tk.Label(popup, text=f"Camera {index+1} FALL DETECTED!", fg="red",
                         font=("Arial", 12, "bold"), bg="white")
        label.pack(pady=20)

        def destroy_popup():
            popup.destroy()
            self.fall_detected_flags[index] = False
            self.popup_shown_flags[index] = False

        popup.after(5000, destroy_popup)

    # ================= Popup for Help Detection =================
    def show_help_popup(self, index):
        popup = Toplevel(self.master)
        popup.title("Notification")
        popup.geometry("300x100")
        popup.configure(bg="white")

        label = tk.Label(popup, text=f"Camera {index+1} HELP DETECTED!", fg="orange",
                         font=("Arial", 12, "bold"), bg="white")
        label.pack(pady=20)

        def destroy_popup():
            popup.destroy()
            self.popup_shown_flags[index] = False

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
        print(f"[DEBUG-OK] is_patient_ok: gesture_ok={gesture_ok}, hand_raised={hand_raised}, waving_hand={waving_hand}")

        return gesture_ok or hand_raised or waving_hand  

    except Exception as e:
        print(f"[ERROR] is_patient_ok: {e}")
        return False

# =============================================  กรณีที่ไม่ล้ม แต่อาจเกิดAccidentอื่นๆ  ==============================================

def detect_fast_hand_wave(pose_results, index, window_size=15, min_swings=2):
    if not pose_results or not pose_results.pose_landmarks:
        wrist_help_wave_history[index] = []
        return False


    landmarks = pose_results.pose_landmarks.landmark
    right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x

    # เก็บประวัติ
    wrist_help_wave_history[index].append(right_wrist_x)
    if len(wrist_help_wave_history[index]) > window_size:
        wrist_help_wave_history[index].pop(0)

    # ต้องมีข้อมูลครบ window_size ก่อนคำนวณ
    if len(wrist_help_wave_history[index]) < window_size:
        return False

    swings = 0
    direction = 0
    for i in range(1, len(wrist_help_wave_history[index])):
        diff = wrist_help_wave_history[index][i] - wrist_help_wave_history[index][i - 1]
        if diff > 0 and direction != 1:
            direction = 1
            swings += 1
        elif diff < 0 and direction != -1:
            direction = -1
            swings += 1

    return swings >= min_swings


def is_near_shoulder(wrist, shoulder, max_dist=0.12):
    dist = ((wrist.x - shoulder.x)**2 + (wrist.y - shoulder.y)**2)**0.5
    return dist < max_dist


def detect_help_request(pose_results, index):
    global help_counter_hand_raised

    try:
        if not pose_results or not pose_results.pose_landmarks:
            help_counter_hand_raised[index] = 0
            return False

        landmarks = pose_results.pose_landmarks.landmark

        # ฟังก์ชันช่วยเช็กว่า landmark ใช้ได้จริง
        def is_valid(lm, min_vis=0.6):
            return (lm.visibility > min_vis and 0 <= lm.x <= 1 and 0 <= lm.y <= 1)

        # (1) กวักมือเร็ว
        fast_wave = detect_fast_hand_wave(pose_results, index)

        # (2) มือทาบหน้าอก (เข้มงวดมากขึ้น)
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        # ถ้า landmark ไม่ชัด → ตัดออกเลย
        if not (is_valid(left_wrist) and is_valid(right_wrist) and
                is_valid(left_shoulder) and is_valid(right_shoulder)):
            chest_help = False
        else:
            chest_top_y = min(left_shoulder.y, right_shoulder.y)
            chest_bottom_y = chest_top_y + 0.08  # ลดช่วงลงเหลือ 8%
            mid_x = (left_shoulder.x + right_shoulder.x) / 2
            max_x_offset = 0.10

            chest_help = (
                (chest_top_y < left_wrist.y < chest_bottom_y and
                 abs(left_wrist.x - mid_x) < max_x_offset and
                 is_near_shoulder(left_wrist, left_shoulder)) or
                (chest_top_y < right_wrist.y < chest_bottom_y and
                 abs(right_wrist.x - mid_x) < max_x_offset and
                 is_near_shoulder(right_wrist, right_shoulder))
            )

        # (3) กุมหัว
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        if not is_valid(nose):
            head_hold = False
        else:
            head_y_threshold = nose.y - 0.1
            head_hold = (
                (is_valid(left_wrist) and left_wrist.y < head_y_threshold) or
                (is_valid(right_wrist) and right_wrist.y < head_y_threshold)
            )

        # ตัดสิน
        if fast_wave or chest_help or head_hold:
            print(f"[INFO-HELP] Cam {index+1} : Help gesture detected"
                  f"(Wave={fast_wave}, Chest={chest_help}, Head={head_hold})")
            print(f"[ALERT-HELP] Cam {index+1} : Help request detect immediately"
                  f"(Wave={fast_wave}, Chest={chest_help}, Head={head_hold})")
            help_counter_hand_raised[index] = 0
            return True

        help_counter_hand_raised[index] = 0
        return False

    except Exception as e:
        print(f"[ERROR] detect_help_request: {e}")
        return False


# ================================================ Core Detection ===============================================================

def detect_fall(sequence, pose_results, hands_results, index):
    global last_log_time, last_person_detected, fall_counters, last_debug_log_time, fall_start_time, help_counter_hand_raised

    current_time = time.time()

    # ✅ ป้องกัน NoneType error
    if pose_results is None or getattr(pose_results, "pose_landmarks", None) is None:
        keypoints = [0.0] * 99
        visibility = [0.0] * 33
    else:
        keypoints = []
        visibility = []
        for lm in pose_results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
            visibility.append(lm.visibility)

    if sum(keypoints) == 0.0:
        fall_counters[index] = 0
        fall_start_time[index] = None
        help_counter_hand_raised[index] = 0  # Reset help counter ถ้าไม่มีคน
        if last_person_detected[index]:
            print(f"[LOG-CAM] Cam {index+1} | No people detected")
            last_person_detected[index] = False
        return False, False

    if not last_person_detected[index]:
        print(f"[LOG-CAM] Cam {index+1} | Person detected")
        last_person_detected[index] = True

    gesture_ok_flag = is_patient_ok(pose_results=pose_results, hands_results=hands_results)
    if gesture_ok_flag:
        print(f"[INFO-OK] Cam {index+1} | Patient OK (OK gesture or hand raised or waving hand)")

    important_indices = [23, 24, 25, 26]
    avg_visibility = np.mean([visibility[i] for i in important_indices])
    if avg_visibility < 0.3:
        print(f"[SKIP] Cam {index+1} | ความมั่นใจต่ำ (avg_visibility={avg_visibility:.2f}) → ข้าม")
        fall_counters[index] = 0
        fall_start_time[index] = None
        return False, False

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
        # ตรวจจับขอความช่วยเหลือด้วย
        help_requested = detect_help_request(pose_results, index)
        return False, help_requested

    ANGLE_Y_THRESHOLD = 45
    HEIGHT_DIFF_THRESHOLD = 0.3

    is_laying_down = body_angle_y is not None and body_angle_y > ANGLE_Y_THRESHOLD and height_diff > HEIGHT_DIFF_THRESHOLD

    if standing:
        print("NO FALLEN (กำลังยืน)")
        fall_counters[index] = 0
        fall_start_time[index] = None
        if random.random() < 0.05:
            save_training_data(keypoints, 0)
        # ตรวจจับขอความช่วยเหลือด้วย
        help_requested = detect_help_request(pose_results, index)
        return False, help_requested

    if len(sequence) < 29:
        sequence.append(keypoints)
        # ตรวจจับขอความช่วยเหลือด้วย
        help_requested = detect_help_request(pose_results, index)
        return False, help_requested

    input_seq = np.expand_dims(np.array(sequence + [keypoints]), axis=0)
    prediction = model.predict(input_seq, verbose=0)[0][0]
    sequence.append(keypoints)
    if len(sequence) > 29:
        sequence.pop(0)

    fallen_by_z_lock = is_fallen_by_locked_z(keypoints)
    is_flat = z_variance < 0.2

    if current_time - last_debug_log_time[index] > 2:
        print(f"[DEBUG] Cam {index+1} | pred={prediction:.3f} | body_angle={body_angle:.1f} | "
              f"body_angle_y={body_angle_y:.1f} | knee_angle={knee_angle:.1f} | "
              f"z_var={z_variance:.3f} | height_diff={height_diff:.3f}")
        last_debug_log_time[index] = current_time

    fall_detected = False

    if fall_counters[index] >= FALL_CONFIRM_FRAMES:
        if gesture_ok_flag:
            print(f"[INFO-OK] Cam {index+1} | OK Gesture detected after fall. Reset fall counter.")
            fall_counters[index] = 0
            fall_start_time[index] = None
            fall_detected = False
        else:
            if fall_start_time[index] is None:
                fall_start_time[index] = current_time
            else:
                elapsed = current_time - fall_start_time[index]
                if elapsed > FALL_ALERT_DELAY:
                    print(f"[ALERT] Cam {index+1} | FALL DETECTED and no response for {FALL_ALERT_DELAY} seconds!")
                    fall_start_time[index] = current_time

            save_training_data(keypoints, 1)
            fall_detected = True

    if prediction > 0.9 or (is_laying_down and prediction > 0.6):
        fall_counters[index] += 1
        print(f"FALL DETECTING... ({fall_counters[index]}/{FALL_CONFIRM_FRAMES})")
        if fall_counters[index] >= FALL_CONFIRM_FRAMES:
            print("FALL DETECTED ✅")
            save_training_data(keypoints, 1)
        fall_detected = False

    elif prediction > 0.7 and body_angle is not None and body_angle < 45 and is_flat:
        fall_counters[index] += 1
        print(f"FALL DETECTING (combined factors)... ({fall_counters[index]}/{FALL_CONFIRM_FRAMES})")
        if fall_counters[index] >= FALL_CONFIRM_FRAMES:
            print("FALL DETECTED ✅ (from body angle + flat Z)")
            save_training_data(keypoints, 1)
        fall_detected = False

    else:
        print("NO FALLEN (ไม่เข้าเกณฑ์)")
        fall_counters[index] = 0
        fall_start_time[index] = None
        if random.random() < 0.05:
            save_training_data(keypoints, 0)
        fall_detected = False

    # ตรวจจับขอความช่วยเหลือด้วย
    help_requested = detect_help_request(pose_results, index)

    return fall_detected, help_requested

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

    # Mediapipe setup
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
    last_frame_time = time.time()

    while not stop_event.is_set():
        try:
            # reconnect ถ้า cap ไม่มีหรือไม่เปิด
            if cap is None or not cap.isOpened():
                if cap:
                    cap.release()
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                if not cap.isOpened():
                    print(f"[WARN] กล้อง {index + 1} ไม่เชื่อมต่อ")
                    with lock:
                        temp = np.zeros((240, 320, 3), dtype=np.uint8)
                        cv2.putText(temp, "Reconnecting...", (30, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if 0 <= index < MAX_CAMERAS:
                            frames[index] = temp
                    time.sleep(3)
                    continue
                else:
                    print(f"[INFO] กล้อง {index + 1} เชื่อมต่อแล้ว")

            # skip buffer frame
            if not cap.grab():
                print(f"[WARN] กล้อง {index + 1} ไม่มี frame (grab fail)")
                time.sleep(0.1)
                continue

            ret, frame = cap.retrieve()
            if not ret or frame is None:
                print(f"[WARN] กล้อง {index + 1} อ่าน frame ไม่ได้ (retrieve fail)")
                time.sleep(0.1)
                continue

            last_frame_time = time.time()

            # resize + convert
            frame = cv2.resize(frame, (320, 240))
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # process Mediapipe ครึ่ง frame
            if index % 2 == 0:
                try:
                    pose_results = pose.process(img_rgb)
                except Exception as e:
                    print(f"[WARN] กล้อง {index + 1} pose processing failed: {e}")
                    pose_results = None
                try:
                    hands_results = hands_detector.process(img_rgb)
                except Exception as e:
                    print(f"[WARN] กล้อง {index + 1} hands processing failed: {e}")
                    hands_results = None
            else:
                pose_results, hands_results = None, None

            # วาด landmarks ปลอดภัย
            if pose_results is not None and getattr(pose_results, 'pose_landmarks', None) is not None:
                draw_landmarks(frame, pose_results.pose_landmarks)

            # ตรวจจับล้ม
            if 0 <= index < MAX_CAMERAS:
                fall_detected, help_requested = detect_fall(
                    sequence_list[index],
                    pose_results,
                    hands_results,
                    index
                )
            else:
                fall_detected, help_requested = False, False

            if fall_detected:
                app_gui.fall_detected_flags[index] = True
                app_gui.popup_shown_flags[index] = False
                cv2.putText(frame, " FALL DETECTED ", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                threading.Thread(
                    target=fall_response_process,
                    args=(source,),
                    daemon=True
                ).start()

            # update frame ปลอดภัย
            try:
                with lock:
                    if 0 <= index < MAX_CAMERAS:
                        frames[index] = frame
            except Exception as e:
                print(f"[WARN] GUI update skipped: {e}")

            # reconnect ถ้าไม่ได้ frame นานเกิน 5 วินาที
            if time.time() - last_frame_time > 5:
                print(f"[ERROR] กล้อง {index + 1} timeout → reconnect")
                cap.release()
                cap = None
                time.sleep(2)

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


# ============================================= Flask API: เพิ่มกล้อง ======================================================


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
    

# ========================================= Flask API: Remove Camera ==================================================================


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

        
        if index in capture_stop_flags:
            capture_stop_flags[index].set()
            capture_threads[index].join(timeout=5)
            del capture_stop_flags[index]
            del capture_threads[index]

    
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


# ======================================================== Main ==================================================================

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
    app_gui = CameraApp(root, frames, lock) 

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