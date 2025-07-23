import tkinter as tk
from tkinter import messagebox
import json
import os

CONFIG_FILE = "stream_config.json"

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
    global root, listbox, streams
    streams = load_streams()

    root = tk.Tk()
    root.title("RTSP Camera Configuration")
    root.geometry("480x400")

    listbox = tk.Listbox(root, height=15, font=("Arial", 12))
    listbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def refresh_listbox():
        listbox.delete(0, tk.END)
        for s in streams:
            listbox.insert(tk.END, str(s))

    def add_stream():
        if len(streams) >= 9:
            messagebox.showwarning("Limit Reached", "You can add up to 9 cameras only.")
            return

        dialog = tk.Toplevel(root)
        dialog.title("Add Camera")
        dialog.geometry("400x150")

        tk.Label(dialog, text="Enter RTSP URL:").pack(padx=10, pady=5)

        entry = tk.Entry(dialog, width=50, show="*")
        entry.pack(padx=10, pady=5)
        entry.focus()

        # Support right-click paste
        def paste(event):
            entry.event_generate("<<Paste>>")
        entry.bind("<Button-3>", paste)

        def on_submit():
            val = entry.get().strip()
            if val:
                if val.isdigit():
                    val = int(val)
                streams.append(val)
                refresh_listbox()
            dialog.destroy()

        tk.Button(dialog, text="Add", command=on_submit).pack(pady=5)

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

        def paste(event):
            entry.event_generate("<<Paste>>")
        entry.bind("<Button-3>", paste)

        def on_submit():
            val = entry.get().strip()
            if val:
                if val.isdigit():
                    val = int(val)
                streams[index] = val
                refresh_listbox()
            dialog.destroy()

        tk.Button(dialog, text="Save", command=on_submit).pack(pady=5)

    def remove_stream():
        sel = listbox.curselection()
        if not sel:
            return
        index = sel[0]
        if messagebox.askyesno("Confirm", "Are you sure you want to delete this camera?"):
            del streams[index]
            refresh_listbox()

    def on_save_and_close():
        save_streams(streams)
        root.destroy()

    # Control buttons
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)

    tk.Button(btn_frame, text="Add Camera", command=add_stream).grid(row=0, column=0, padx=5)
    tk.Button(btn_frame, text="Edit", command=edit_stream).grid(row=0, column=1, padx=5)
    tk.Button(btn_frame, text="Delete", command=remove_stream).grid(row=0, column=2, padx=5)
    tk.Button(btn_frame, text="Save and Close", command=on_save_and_close).grid(row=0, column=3, padx=5)

    refresh_listbox()
    root.mainloop()


