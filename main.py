import argparse
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import tkinter.simpledialog as simpledialog
import src.predictions as predictions

def get_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_path', required=False, type=str, help='Path to the input video.')
    return parser.parse_args()

def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        predictions.ImageProcessor.predict_video(file_path)

def record_webcam():
    camera_id = simpledialog.askstring("Input", "Enter IP address (e.g., http://192.168.0.100:8080/video):")
    try:
        camera_id = int(camera_id)
        predictions.ImageProcessor.predict_webcam(camera_id)
    except:
        if camera_id is not None:
            predictions.ImageProcessor.predict_webcam(camera_id)

def open_gui():
    root = tk.Tk()
    root.title("Depth Estimation")
    root.geometry("300x150")

    style = ttk.Style()
    style.configure('TButton', padding=6, relief="flat")

    select_button = ttk.Button(root, text="Select Video", command=select_file, style='TButton')
    select_button.pack(pady=10)

    webcam_button = ttk.Button(root, text="Record from Webcam", command=record_webcam, style='TButton')
    webcam_button.pack(pady=10)

    root.mainloop()
if __name__ == "__main__":
    open_gui()