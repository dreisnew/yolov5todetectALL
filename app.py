import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import subprocess
import os
import glob
import threading

# Change to your yolov5 folder
os.chdir(r"C:\Users\Andrea Chiang\Downloads\python\yolo_thesis\yolov5")

selected_folder = ""
save_folder = ""

# Create main window
root = tk.Tk()
root.title("ALL Detection App")
root.geometry("1200x550")

# --- Button frame ---
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# --- Functions ---
def upload_folder():
    global selected_folder
    selected_folder = filedialog.askdirectory(title="Select folder with images")
    if not selected_folder:
        return
    folder_label.config(text=f"Selected folder:\n{selected_folder}")

def get_latest_detect_folder():
    folders = glob.glob("runs/detect/*")
    if not folders:
        return "runs/detect/demo"
    latest = max(folders, key=os.path.getctime)
    return latest

def run_model():
    global selected_folder, save_folder
    if not selected_folder:
        status_label.config(text="Please select a folder first.")
        return

    # Ask user where to save results
    save_folder = filedialog.askdirectory(title="Select folder to save results")
    if not save_folder:
        return

    image_files = [f for f in os.listdir(selected_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not image_files:
        status_label.config(text="No images found in the selected folder.")
        return

    for img_name in image_files:
        img_path = os.path.join(selected_folder, img_name)

        # Update status
        status_label.config(text=f"Running model on: {img_name}")
        root.update_idletasks()  # refresh GUI

        # Display current input image
        img_input = Image.open(img_path)
        img_input_tk = ImageTk.PhotoImage(img_input)  # use original size
        input_image_label.config(image=img_input_tk)
        input_image_label.image = img_input_tk

        # Run YOLO detection
        command = [
            "python",
            "detect.py",
            "--weights", "runs/train/ALL_HEM_80-10-10_with_augments_100Ev5l/weights/best.pt",
            "--img", "640",
            "--conf", "0.70",
            "--source", img_path,
            "--project", save_folder,  # write directly to user folder
            "--name", "",              # no extra subfolder
            "--exist-ok",
        ]
        subprocess.run(command)

        # Get result image (YOLO saves as same filename in save_folder)
        result_path = os.path.join(save_folder, img_name)

        # Display output image
        img_output = Image.open(result_path)
        img_output_tk = ImageTk.PhotoImage(img_output)  # original size
        output_image_label.config(image=img_output_tk)
        output_image_label.image = img_output_tk

    # Finished
    status_label.config(text=f"All images processed.\nResults saved in: {save_folder}")

# --- Buttons ---
btn_upload = tk.Button(button_frame, text="Upload Folder", command=upload_folder)
btn_upload.grid(row=0, column=0, padx=10)

btn_run = tk.Button(button_frame, text="Run Detection", command=lambda: threading.Thread(target=run_model).start())
btn_run.grid(row=0, column=1, padx=10)

# --- Labels ---
folder_label = tk.Label(root, text="No folder selected", fg="blue", wraplength=800)
folder_label.pack(pady=5)

# --- Image frame ---
image_frame = tk.Frame(root)
image_frame.pack(pady=10)

# Input image
input_image_label = tk.Label(image_frame, text="Input Image")
input_image_label.grid(row=0, column=0, padx=10)

# Output image
output_image_label = tk.Label(image_frame, text="Output Image")
output_image_label.grid(row=0, column=1, padx=10)

# Status label below images
status_label = tk.Label(root, text="", fg="green", font=("Arial", 12))
status_label.pack(pady=10)

# --- Run Tkinter loop ---
root.mainloop()