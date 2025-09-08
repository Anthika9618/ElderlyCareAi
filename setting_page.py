import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from dotenv import load_dotenv
import os

# โหลดค่า .env
load_dotenv()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
FULL_NAME = os.getenv("FULL_NAME", "")
EMAIL = os.getenv("EMAIL", "")


class AdminSettings:
    def __init__(self, master, on_logout=None, on_back=None):
        self.master = master
        self.master.title("Admin Profile")
        self.master.geometry("420x520")
        self.master.configure(bg="#E9ECEF")  
        self.master.resizable(False, False)
        self.on_logout = on_logout
        self.on_back = on_back
        self.entries = {}

        # Modern ttk style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.map("TButton", 
                  background=[("active", "#3498db")], 
                  foreground=[("active", "white")])

        style.configure("TEntry", padding=5)

        # Card Frame
        self.card = tk.Frame(master, bg="white", bd=0, relief="solid")
        self.card.place(relx=0.5, rely=0.5, anchor="center", width=360, height=480)

        # Shadow effect (fake by lower bg frame)
        shadow = tk.Frame(master, bg="#d1d1d1")
        shadow.place(relx=0.5, rely=0.5, anchor="center", width=365, height=485)
        self.card.lift()

        # ==================================== Avatar =================================

        avatar_img = Image.open(
            "/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/avatar_admin.png"
        ).resize((90, 90))
        self.avatar_photo = ImageTk.PhotoImage(avatar_img)
        lbl_avatar = tk.Label(self.card, image=self.avatar_photo, bg="white")
        lbl_avatar.pack(pady=25)

        # Info fields
        self.form_frame = tk.Frame(self.card, bg="white")
        self.form_frame.pack(pady=10)

        self.create_label_entry("Username", ADMIN_USERNAME, 0,
            "/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/username_admin.png")
        self.create_label_entry("Full Name", FULL_NAME, 1,
            "/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/fullname_admin.png")
        self.create_label_entry("Email", EMAIL, 2,
            "/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/email_admin.png")

        # =======================================  Buttons frame  ===============================

        btn_frame = tk.Frame(self.card, bg="white")
        btn_frame.pack(pady=20)

        # =======================================  Buttons with icons  ===========================

        self.add_button(btn_frame, " Edit Password", 
                        "/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/adminedit.png",
                        self.edit_password)

        self.add_button(btn_frame, " Logout", 
                        "/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/user-logout.png",
                        self.logout)


        back_frame = tk.Frame(self.card, bg="white")
        back_frame.pack(pady=(5, 0))

        self.add_button(back_frame, " Back", 
                        "/mnt/c/projectaunaun/Elderly/ElderlyCareSystem/ElderlyCareSystem/logoandemoji/back.png",
                        self.back)



    def add_button(self, frame, text, icon_path, command):
        icon = Image.open(icon_path).resize((20, 20))
        photo = ImageTk.PhotoImage(icon)
        btn = ttk.Button(frame, text=text, image=photo, compound="left", command=command)
        btn.image = photo
        btn.pack(side="left", padx=8)

    def create_label_entry(self, text, value, row, icon_path):
        icon = Image.open(icon_path).resize((20, 20))
        photo = ImageTk.PhotoImage(icon)
        label = ttk.Label(self.form_frame, text=text, image=photo, 
                          compound="left", background="white", font=("Segoe UI", 10, "bold"))
        label.image = photo
        label.grid(row=row, column=0, padx=10, pady=8, sticky="w")

        entry = ttk.Entry(self.form_frame, font=("Segoe UI", 10))
        entry.insert(0, value)
        entry.config(state="readonly")
        entry.grid(row=row, column=1, padx=10, pady=8, sticky="w")
        self.entries[text] = entry

    def edit_password(self):
        popup = tk.Toplevel(self.master)
        popup.title("Edit Password")
        popup.geometry("320x280")
        popup.configure(bg="#E9ECEF")
        popup.resizable(False, False)

        tk.Label(popup, text="New Password:", bg="#E9ECEF", font=("Segoe UI", 10)).pack(pady=(15, 5))
        entry_new = ttk.Entry(popup, show="*")
        entry_new.pack()

        tk.Label(popup, text="Confirm Password:", bg="#E9ECEF", font=("Segoe UI", 10)).pack(pady=(15, 5))
        entry_confirm = ttk.Entry(popup, show="*")
        entry_confirm.pack()

        def save_password():
            new_pw = entry_new.get()
            confirm_pw = entry_confirm.get()
            if not new_pw or not confirm_pw:
                messagebox.showerror("Error", "Password cannot be empty.")
                return
            if new_pw != confirm_pw:
                messagebox.showerror("Error", "Passwords do not match!")
                return
            self.update_env_password(new_pw)
            messagebox.showinfo("Success", "Password updated successfully!")
            popup.destroy()

        ttk.Button(popup, text="Save", command=save_password).pack(pady=20)

    def update_env_password(self, new_pw):
        env_path = ".env"
        lines = []
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith("ADMIN_PASSWORD="):
                    lines.append(f"ADMIN_PASSWORD={new_pw}\n")
                else:
                    lines.append(line)
        with open(env_path, "w") as f:
            f.writelines(lines)

        self.refresh_profile()

    def logout(self):
        self.master.destroy()
        if self.on_logout:
            self.on_logout()

    def back(self):
        self.master.destroy()
        if self.on_back:
            self.on_back()


    def load_admin_env(self) :
        load_dotenv()
        return{
            "username" : os.getenv("ADMIN_USERNAME", ""),
            "fullname" : os.getenv("FULL_NAME", ""),
            "email" : os.getenv("EMAIL","")
        }

    def refresh_profile(self):
        self.env_data = self.load_admin_env()
        for key, entry in self.entries.items():
            entry.config(state="Normal")
            if key == "Username":
                entry.delete(0, tk.END)
                entry.insert(0, self.env_data["username"])
            elif key == "Full name" :
                entry.delete(0, tk.END)
                entry.insert(0, self.env_data["full name"])
            elif key == "Email" :
                entry.delete(0, tk.END)
                entry.insert(0, self.env_data["Email"])
            entry.config(state="readonly")

    
def open_settings_window(parent, on_logout=None, on_back=None):
    win = tk.Toplevel(parent)
    app = AdminSettings(win, on_logout=on_logout, on_back=on_back)
    win.grab_set()



# if __name__ == "__main__":
#     root = tk.Tk()
#     app = AdminSettings(root, on_logout=lambda: root.destroy())
#     root.mainloop()
