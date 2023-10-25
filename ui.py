# ui.py
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import processing

polygon_points = []
video_path = ""

MAX_WIDTH = 640
MAX_HEIGHT = 480

def resize_image(image, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    global resize_ratio
    width_ratio = max_width / image.width
    height_ratio = max_height / image.height
    resize_ratio = min(width_ratio, height_ratio)
    new_width = int(image.width * resize_ratio)
    new_height = int(image.height * resize_ratio)
    return image.resize((new_width, new_height), Image.LANCZOS)

is_webcam_mode = False

def load_video():
    global video_path, is_webcam_mode, cap
    if is_webcam_mode:
        cap = cv2.VideoCapture(0)  # Open the default camera (webcam)
        update_canvas_with_webcam_feed()
        if ret:
            img = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
            img = resize_image(img)
            imgtk = ImageTk.PhotoImage(image=img)
            video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            video_canvas.image = imgtk
    else:
        video_path = filedialog.askopenfilename(title="Select a Video")
        video_input_label.config(text=video_path)

        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()
        if ret:
            img = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
            img = resize_image(img)
            imgtk = ImageTk.PhotoImage(image=img)
            video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            video_canvas.image = imgtk

def toggle_input_mode():
    global is_webcam_mode
    is_webcam_mode = not is_webcam_mode
    if is_webcam_mode:
        video_input_button.config(text="Load Webcam Feed")
        video_input_label.config(text="Webcam Mode Active")
    else:
        video_input_button.config(text="Load Video")
        video_input_label.config(text="No video selected.")

def update_canvas_with_webcam_feed():
    ret, frame = cap.read()
    if ret:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = resize_image(img)
        imgtk = ImageTk.PhotoImage(image=img)
        video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        video_canvas.image = imgtk
        root.after(10, update_canvas_with_webcam_feed)  # Refresh every 10ms
    else:
        print("Failed to grab frame from webcam.")


def on_canvas_click(event):
    global polygon_points
    x, y = event.x, event.y
    video_canvas.create_oval(x-2, y-2, x+2, y+2, fill="red", width=2)
    polygon_points.append([x, y])

# def rescale_coordinates(coords):
#     """Rescale coordinates to original size."""
#     width_ratio = MAX_WIDTH / video_canvas.winfo_width()
#     height_ratio = MAX_HEIGHT / video_canvas.winfo_height()
#
#     # Assuming uniform scaling for both width and height
#     scaling_factor = min(width_ratio, height_ratio)
#
#     return [[int(x / scaling_factor) for x in point] for point in coords]

def rescale_coordinates(coords):
    """Rescale coordinates to original size."""

    global cap, is_webcam_mode

    if is_webcam_mode:
        # If in webcam mode, open the default camera
        cap = cv2.VideoCapture(0)
    else:
        # If not in webcam mode, use the previously set video_path
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Failed to open the video or webcam.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Failed to read frame. Check the video or webcam source.")
        return None

    original_height, original_width, _ = frame.shape
    aspect_ratio = original_width / original_height

    # Calculate the width and height of the resized video as displayed on the canvas
    if original_width > original_height:
        new_width = MAX_WIDTH
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = MAX_HEIGHT
        new_width = int(new_height * aspect_ratio)

    width_ratio = original_width / new_width
    height_ratio = original_height / new_height

    # Scale the coordinates
    return [[int(x * width_ratio), int(y * height_ratio)] for x, y in coords]



def start_processing():
    global video_path, polygon_points, is_webcam_mode, cap, resize_ratio
    if is_webcam_mode:
        cap.release()  # Release the webcam
        rescaled_polygon = rescale_coordinates(polygon_points)
        root.destroy()
        processing.process_webcam_with_annotations(rescaled_polygon)
    elif video_path and polygon_points:
        rescaled_polygon = rescale_coordinates(polygon_points)
        root.destroy()
        processing.process_video_with_annotations(video_path, rescaled_polygon)




# GUI Styling
DARK_COLOR = "#2E2E2E"
LIGHT_COLOR = "#6E6E6E"
BUTTON_COLOR = "#99ffff"  # Pale blue color
TEXT_COLOR = "#FFFFFF"


root = tk.Tk()
root.title("Video Annotator")
root.configure(bg=DARK_COLOR)

# Custom button styling with rounded edges
style = {
    "bg": BUTTON_COLOR,
    "fg": TEXT_COLOR,
    "borderwidth": 0,
    "highlightthickness": 0,
    "font": ("Arial", 10, "bold"),
    "activebackground": LIGHT_COLOR,
    "activeforeground": TEXT_COLOR
}
button_style = {**style, "relief": tk.RAISED, "padx": 10, "pady": 5}

control_frame = tk.Frame(root, bg=DARK_COLOR)
control_frame.pack(pady=20)

toggle_mode_button = tk.Button(control_frame, text="Toggle Webcam/Video Mode", command=toggle_input_mode, **button_style)
toggle_mode_button.grid(row=0, column=2, padx=10)

video_input_label = tk.Label(control_frame, text="No video selected.", bg=DARK_COLOR, fg=TEXT_COLOR)
video_input_label.grid(row=0, column=0, padx=10)

video_input_button = tk.Button(control_frame, text="Load Video", command=load_video, **button_style)
video_input_button.grid(row=0, column=1, padx=10)

video_canvas = tk.Canvas(root, bg="white", width=MAX_WIDTH, height=MAX_HEIGHT)
video_canvas.pack(pady=20)
video_canvas.bind("<Button-1>", on_canvas_click)

start_button = tk.Button(root, text="Start Processing", command=start_processing, **button_style)
start_button.pack(pady=20)

root.mainloop()
