import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from dotenv import load_dotenv
import os

ENV_FILE = ".env"

if not os.path.exists(ENV_FILE):
    with open(ENV_FILE, "w") as f:
        f.write("")
load_dotenv(ENV_FILE)

def set_key(env_file, key, value):
    lines = []
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            lines = f.readlines()
    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}\n")
    with open(env_file, "w") as f:
        f.writelines(lines)


def register_user():
    reg_win = tk.Toplevel()
    reg_win.title("Register")
    reg_win.geometry("600x690")
    reg_win.configure(bg="#f0f2f5")
    reg_win.resizable(False, False)

    # ================================================== Load Icons ======================================================================
    
    def load_icon(path, size=(20, 20)):
        try:
            img = Image.open(path)
            img = img.resize(size, Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    icons = {
        "fullname": load_icon("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/fullname.png", size=(30, 30)),
        "username": load_icon("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/username.png", size=(30, 30)),
        "password": load_icon("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/password.png", size=(30, 30)),
        "confrimpassword": load_icon("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/confrimpassword.png", size=(30, 30)),
        "email": load_icon("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/email.png", size=(30, 30)),
        "header": load_icon("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/createaccount.png", size=(45, 45)),
        "register": load_icon("/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/register.png", size=(30, 30)),
    }

    register_icon = icons["register"]


    # ======================================================== Header ========================================================================

    header_frame = tk.Frame(reg_win, bg="#f0f2f5")
    header_frame.pack(pady=20)
    if icons["header"]:
        tk.Label(header_frame, image=icons["header"], bg="#f0f2f5").pack()
    tk.Label(header_frame, text="Create Account", font=("Helvetica", 18, "bold"), bg="#f0f2f5", fg="#333").pack()

    # ======================================================== Form Frame =====================================================================
    

    form_frame = tk.Frame(reg_win, bg="#f0f2f5")
    form_frame.pack(pady=10, padx=20, fill="both", expand=True)


    # ============================================================= Submit Button =================================================================

    def submit():
        username = username_entry.get().strip()
        password = password_entry.get().strip()

        if not username or not password:
            messagebox.showerror("Error", "Username and Password are required!")
            return

        set_key(ENV_FILE, "ADMIN_USERNAME", username)
        set_key(ENV_FILE, "ADMIN_PASSWORD", password)
        set_key(ENV_FILE, "FULL_NAME", name_entry.get().strip())
        set_key(ENV_FILE, "EMAIL", email_entry.get().strip())

        messagebox.showinfo("Success", "Registration complete! Saved to .env")
        reg_win.destroy()

    def create_entry(label_text, icon=None, show=None, placeholder=""):
        tk.Label(form_frame, text=label_text, bg="#f0f2f5", fg="#555", font=("Helvetica", 12)).pack(anchor="w", pady=(10, 2))
        frame = tk.Frame(form_frame, bg="#e0e0e0")
        frame.pack(fill="x", pady=(0,5))
        if icon:
            lbl_icon = tk.Label(frame, image=icon, bg="#e0e0e0")
            lbl_icon.pack(side="left", padx=5)

        entry = tk.Entry(frame, font=("Helvetica", 12), relief="flat", bg="#e0e0e0", fg="#999", show=show)
        entry.pack(fill="x", padx=5, ipady=8)

        entry.insert(0, placeholder)
        def on_focus_in(event):
            if entry.get() == placeholder:
                entry.delete(0, "end")
                entry.config(fg="#333")
        def on_focus_out(event):
            if entry.get() == "":
                entry.insert(0, placeholder)
                entry.config(fg="#999")
        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)

        return entry


    name_entry = create_entry("FullName", icon=icons["fullname"], placeholder= "Fullname")
    username_entry = create_entry("Username", icon=icons["username"], placeholder="Username")
    password_entry = create_entry("Password", icon=icons["password"], placeholder="Password")
    confrimpassword_entry = create_entry("ConfrimPassword", icon=icons["confrimpassword"], placeholder="ConfrimPass")
    email_entry = create_entry("Email", icon=icons["email"], placeholder="example@email.com")

    gender_var = tk.StringVar(value="Male")
    gender_frame = tk.Frame(form_frame, bg="#f0f2f5")
    gender_frame.pack(fill="x", pady=(5,10))
    tk.Label(gender_frame, text="Gender : ", bg="#f0f2f5", fg="#555", font=("Helvetica", 12)).pack(side="left")
    tk.Radiobutton(gender_frame, text="Male", variable=gender_var, value="Male", bg="#f0f2f5").pack(side="left", padx=5)
    tk.Radiobutton(gender_frame, text="Female", variable=gender_var, value="Female", bg="#f0f2f5").pack(side="left", padx=5)


    submit_btn = tk.Button(
        reg_win,
        text="Register",
        image=register_icon,     
        compound="left",          
        command=submit,
        font=("Helvetica", 13, "bold"),
        bg="#4a90e2",
        fg="white",
        activebackground="#357ABD",
        activeforeground="white",
        bd=0,
        relief="flat",
        cursor="hand2"
    )
    submit_btn.image = register_icon  
    
    submit_btn.pack(pady=20, ipadx=5, ipady=5)

    reg_win.grab_set()
    reg_win.mainloop()
