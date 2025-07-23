import tkinter as tk
import requests

def open_open_camera_gui():
    def submit():
        ip = entry.get()
        if not ip:
            status_label.config(text="Please enter camera IP", fg="red")
            return
        try:
            res = requests.post("http://127.0.0.1:5001/add_camera", json={"ip": ip})
            data = res.json()
            if res.status_code == 200:
                status_label.config(text=f"{data['message']}", fg="green")
                entry.delete(0, tk.END)
            else:
                status_label.config(text=f"{data.get('error', 'Unknown error')}", fg="red")
        except:
            status_label.config(text="Unable to connect to API", fg="red")

    def paste(event=None):
        entry.insert(tk.INSERT, window.clipboard_get())

    window = tk.Tk()
    window.title("Add new camera")
    window.geometry("400x150")

    tk.Label(window, text="New camera IP:").pack(pady=5)
    entry = tk.Entry(window, width=30)
    entry.pack()

    # รองรับ Ctrl+V และเมาส์ขวา Paste
    entry.bind("<Control-v>", paste)
    entry.bind("<Button-3>", paste)

    tk.Button(window, text="Submit", command=submit).pack(pady=10)
    status_label = tk.Label(window, text="", fg="blue")
    status_label.pack()

    window.mainloop()
