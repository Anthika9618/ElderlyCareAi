import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter import ttk
import threading, time
import numpy as np 
import json
import os


CONFIG_FILE = "stream_config.json"


frames = []
fall_detected_flags = []
popup_shown_flags = []


def create_modern_button(parent, text, icon_path, command=None):
    try:
        img = Image.open(icon_path).resize((35, 35))
        photo = ImageTk.PhotoImage(img)
        btn = tk.Button(
            parent,
            text=f" {text}",
            image=photo,
            compound="left",
            command=command,
            font=("Arial", 12, "bold"),
            bg="#DFDFE0",
            fg="black",
            activebackground="#BFBFC0",
            activeforeground="black",
            relief="raised",
            bd=3,
            width=150,
            height=50,
            padx=15,
            pady=10
        )
        btn.image = photo  
        return btn
    except Exception as e:
        print("Load icon failed", e)
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Arial", 12, "bold"),
            bg="#DFDFE0",
            fg="black",
            activebackground="#BFBFC0",
            activeforeground="black",
            relief="raised",
            bd=3,
            width=150,
            height=50,
            padx=10,
            pady=5
        )

def load_streams():
    if not os.path.exists(CONFIG_FILE):
        return []
    with open(CONFIG_FILE, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and "streams" in data:
        return data["streams"]
    elif isinstance(data, list):
        return data
    else:
        return []

def save_streams(streams):
    with open(CONFIG_FILE, 'w') as f:
        json.dump({"streams": streams}, f, indent=2)


def open_config_gui():
    global root, listbox, streams, frames, fall_detected_flags, popup_shown_flags
    streams = load_streams()


    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in streams]
    fall_detected_flags = [False] * len(streams)
    popup_shown_flags = [False] * len(streams)

    root = tk.Tk()
    root.title("Camera Configuration")
    root.geometry("700x500")
    root.configure(bg="#f0f2f5")

    # Listbox
    listbox = tk.Listbox(root, height=15, font=("Arial", 12))
    listbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def refresh_listbox():
        listbox.delete(0, tk.END)
        for s in streams:
            listbox.insert(tk.END, str(s))

    # ===================================================  Add stream  ==============================================================

    def add_stream():
        if len(streams) >= 12:
            messagebox.showwarning("Limit Reached", "You can add up to 12 cameras only.")
            return

        dialog = tk.Toplevel(root)
        dialog.title("Add Camera")
        dialog.geometry("400x250")

        tk.Label(dialog, text="Enter RTSP URL:").pack(padx=10, pady=5)
        entry = tk.Entry(dialog, width=50)
        entry.pack(padx=10, pady=5)
        entry.focus()
        entry.bind("<Button-3>", lambda e: entry.event_generate("<<Paste>>"))

        def on_submit():
            val = entry.get().strip()
            if not val:
                messagebox.showwarning("Input Error", "Please enter RTSP URL.")
                return

            progress = ttk.Progressbar(dialog, orient="horizontal", length=300, mode="determinate")
            progress.pack(pady=10)
            status_label = tk.Label(dialog,text="Processing...", font=("Arial", 10))
            status_label.pack()

            def task():
                for i in range(101):
                    time.sleep(0.05)
                    progress["value"] = i
                    dialog.update_idletasks()

                streams.append(val)  # เก็บ URL
                frames.append(np.zeros((240, 320, 3), dtype=np.uint8))  # สร้าง placeholder frame
                fall_detected_flags.append(False)
                popup_shown_flags.append(False)
                refresh_listbox()

                try:
                    img_correct = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/check-mark.png").resize((30, 30))
                    photo_correct = ImageTk.PhotoImage(img_correct)
                    status_label.config(image=photo_correct, text="")
                    status_label.image = photo_correct
                except:
                    status_label.config(text="Successfully")

                time.sleep(0.5)
                dialog.destroy()

            threading.Thread(target=task).start()


        add_btn = create_modern_button(
            dialog,
            "Add",
            "/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/add-camera.png",
            on_submit
        )
        add_btn.pack(pady=5)


    # ======================================================  Edit stream  =================================================================


    def edit_stream():
        sel = listbox.curselection()
        if not sel:
            return
        index = sel[0]
        current = streams[index]

        dialog = tk.Toplevel(root)
        dialog.title("Edit Camera")
        dialog.geometry("400x150")

        tk.Label(dialog, text="Enter RTSP URL:").pack(padx=10, pady=5)
        entry = tk.Entry(dialog, width=50)
        entry.insert(0, str(current))
        entry.pack(padx=10, pady=5)
        entry.focus()
        entry.bind("<Button-3>", lambda e: entry.event_generate("<<Paste>>"))

        def on_submit():
            val = entry.get().strip()
            if val:
                streams[index] = val
                refresh_listbox()
            dialog.destroy()

        save_btn = create_modern_button(
            dialog,
            "Save",
            "/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/save.png",
            on_submit
        )
        save_btn.pack(pady=5)


    # ===============================================================  Remove stream  =========================================================


    def remove_stream():
        sel = listbox.curselection()
        if not sel:
            return
        index = sel[0]

        confirmed = []

        def confirm_delete():
            confirmed.append(True)
            popup.destroy()

        def cancel_delete():
            popup.destroy()

        popup = tk.Toplevel(root)
        popup.title("Confirm Removed")
        popup.geometry("400x200")
        popup.resizable(False, False)
        popup.transient(root)
        popup.grab_set()

        # ไอคอน
        img = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/question-del.png")
        img = img.resize((50, 50))
        icon = ImageTk.PhotoImage(img)

        lbl_icon = tk.Label(popup, image=icon)
        lbl_icon.image = icon
        lbl_icon.pack(pady=10)

        # ข้อความ
        lbl_text = tk.Label(
            popup,
            text="Are you sure you want to remove this camera?",
            wraplength=400,
            justify="center",
            font=("Arial", 12, "bold"), 
            fg="#333333"
        )
        lbl_text.pack(pady=10)

        # ปุ่ม
        btn_frame = tk.Frame(popup)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Yes", command=confirm_delete, bg="#4a90e2", fg="white", width=11).pack(side="left", padx=10)
        tk.Button(btn_frame, text="No", command=cancel_delete, bg="#e74c3c", fg="white", width=11).pack(side="left", padx=10)

        root.wait_window(popup)

        
        if confirmed:
            del streams[index]
            refresh_listbox()

            # โมเดิร์น popup แทน messagebox
            popup_info = tk.Toplevel(root)
            popup_info.title("Removed")
            popup_info.geometry("400x180")
            popup_info.resizable(False, False)
            popup_info.transient(root)
            popup_info.grab_set()
            popup_info.configure(bg="#f0f2f5")

            
            try:
                img_ok = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/check-mark.png")
                img_ok = img_ok.resize((64,64))
                icon_ok = ImageTk.PhotoImage(img_ok)
                lbl_icon = tk.Label(popup_info, image=icon_ok, bg="#f0f2f5")
                lbl_icon.image = icon_ok
                lbl_icon.pack(pady=10)
            except:
                pass

            
            lbl_text = tk.Label(
                popup_info,
                text="The camera has been removed successfully!",
                font=("Arial", 14, "bold"),
                fg="#2c3e50",
                bg="#f0f2f5",
                wraplength=350,
                justify="center"
            )
            lbl_text.pack(pady=10)

            popup_info.after(5000, popup_info.destroy)


    # ==============================================================  Save and close  ==========================================================


    def on_save_and_close():
        save_streams(streams)
        root.destroy()

    # ==============================================================  Buttons frame  ===========================================================

    btn_frame = tk.Frame(root, bg="#f0f2f5")
    btn_frame.pack(pady=15)

    # ==============================================================  Add Camera button  =======================================================

    try:
        img_add = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/addcamera.png").resize((30,30))
        photo_add = ImageTk.PhotoImage(img_add)
        btn_add = tk.Button(
            btn_frame, text=" Add Camera", image=photo_add, compound="left", command=add_stream, font=("Arial",10,"bold"),
            bg="#DFDFE0", fg="black", activebackground="#BFBFC0", activeforeground="black",
            relief="raised", bd=3, padx=10, pady=5
        )
        btn_add.image = photo_add
        btn_add.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
    except:
        tk.Button(btn_frame, text="Add Camera", command=add_stream).grid(row=0, column=0, padx=10, pady=5)

    # ===============================  Edit button  =============================

    try:
        img_edit = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/edit.png").resize((30,30))
        photo_edit = ImageTk.PhotoImage(img_edit)
        btn_edit = tk.Button(
            btn_frame, text=" Edit", image=photo_edit, compound="left", command=edit_stream, font=("Arial",10,"bold"),
            bg="#DFDFE0", fg="black", activebackground="#BFBFC0", activeforeground="black",
            relief="raised", bd=3, padx=10, pady=5
        )
        btn_edit.image = photo_edit
        btn_edit.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
    except:
        tk.Button(btn_frame, text="Edit", command=edit_stream).grid(row=0, column=1, padx=10, pady=5)

    # ===============================  Delete button  =============================

    try:
        img_del = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/delete.png").resize((30,30))
        photo_del = ImageTk.PhotoImage(img_del)
        btn_del = tk.Button(
            btn_frame, text=" Delete", image=photo_del, compound="left", command=remove_stream, font=("Arial",10,"bold"),
            bg="#DFDFE0", fg="black", activebackground="#BFBFC0", activeforeground="black",
            relief="raised", bd=3, padx=10, pady=5
        )
        btn_del.image = photo_del
        btn_del.grid(row=0, column=2, padx=10, pady=5, sticky="ew")
    except:
        tk.Button(btn_frame, text="Delete", command=remove_stream).grid(row=0, column=2, padx=10, pady=5)

    # ===============================  Save & Close button  =============================

    try:
        img_save = Image.open("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/saveandclose.png").resize((30,30))
        photo_save = ImageTk.PhotoImage(img_save)
        btn_save = tk.Button(
            btn_frame, text=" Save & Close", image=photo_save, compound="left", command=on_save_and_close, font=("Arial",10,"bold"),
            bg="#DFDFE0", fg="black", activebackground="#BFBFC0", activeforeground="black",
            relief="raised", bd=3, padx=10, pady=5
        )
        btn_save.image = photo_save
        btn_save.grid(row=0, column=3, padx=10, pady=5, sticky="ew")
    except:
        tk.Button(btn_frame, text="Save & Close", command=on_save_and_close).grid(row=0, column=3, padx=10, pady=5)


    refresh_listbox()
    root.mainloop()

# # Run GUI
# open_config_gui()


