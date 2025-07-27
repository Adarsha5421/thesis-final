# import cv2
# import os
# from ultralytics import YOLO, solutions
# import numpy as np
# import time





# RedLight = np.array([[998, 125],[998, 155],[972, 152],[970, 127]])
# GreenLight = np.array([[971, 200],[996, 200],[1001, 228],[971, 230]])
# ROI = np.array([[910, 372],[388, 365],[338, 428],[917, 441]])


# model = YOLO("yolov8m.pt")

# coco = model.model.names

# TargetLabels = ["bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]


# def is_region_light(image, polygon, brightness_threshold=128):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     mask = np.zeros_like(gray_image)
    
#     cv2.fillPoly(mask, [np.array(polygon)], 255)
    
#     roi = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    
#     mean_brightness = cv2.mean(roi, mask=mask)[0]
    
#     return mean_brightness > brightness_threshold


# def draw_text_with_background(frame, text, position, font, scale, text_color, background_color, border_color, thickness=2, padding=5):
#     """Draw text with background and border on the frame."""
#     (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
#     x, y = position
#     # Background rectangle
#     cv2.rectangle(frame, 
#                   (x - padding, y - text_height - padding), 
#                   (x + text_width + padding, y + baseline + padding), 
#                   background_color, 
#                   cv2.FILLED)
#     # Border rectangle
#     cv2.rectangle(frame, 
#                   (x - padding, y - text_height - padding), 
#                   (x + text_width + padding, y + baseline + padding), 
#                   border_color, 
#                   thickness)
#     # Text
#     cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, lineType=cv2.LINE_AA)


# cap = cv2.VideoCapture("tr.mp4")

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         print("number of frames have finished.")
#         break
#     else:
#         frame = cv2.resize(frame, (1100, 700))
#         cv2.polylines(frame, [RedLight], True, [0, 0, 255], 1)
#         cv2.polylines(frame, [GreenLight], True, [0, 255, 0], 1)
#         cv2.polylines(frame, [ROI], True, [255, 0, 0], 2)
        
#         results = model.predict(frame, conf=0.75)
#         for result in results:
#             boxes = result.boxes.xyxy
#             confs = result.boxes.conf
#             classes = result.boxes.cls
            
#             for box, conf, cls in zip(boxes, confs, classes):
#                 if coco[int(cls)] in TargetLabels:
#                     x, y, w, h = box
#                     x, y, w, h = int(x), int(y), int(w), int(h)
#                     cv2.rectangle(frame, (x, y), (w, h), [0, 255, 0], 2)
#                     draw_text_with_background(frame, 
#                                       f"{coco[int(cls)].capitalize()}, conf:{(conf)*100:0.2f}%", 
#                                       (x, y - 10), 
#                                       cv2.FONT_HERSHEY_COMPLEX, 
#                                       0.6, 
#                                       (255, 255, 255),  # White text
#                                       (0, 0, 0),  # Black background
#                                       (0, 0, 255))  # Red border

#                 if is_region_light(frame, RedLight):
#                     if cv2.pointPolygonTest(ROI, (x, y), False) >= 0 or cv2.pointPolygonTest(ROI, (w, h), False) >= 0:
#                         draw_text_with_background(frame, 
#                                       f"The {coco[int(cls)].capitalize()} violated the traffic signal.", 
#                                       (10, 30), 
#                                       cv2.FONT_HERSHEY_COMPLEX, 
#                                       0.6, 
#                                       (255, 255, 255),  # White text
#                                       (0, 0, 0),  # Black background
#                                       (0, 0, 255))  # Red border

#                         cv2.polylines(frame, [ROI], True, [0, 0, 255], 2)
#                         cv2.rectangle(frame, (x, y), (w, h), [0, 0, 255], 2)
#                         # time.sleep(1)
                        
    
#         cv2.imshow("frame", frame)
#         if cv2.waitKey(1) == 27:
#             break
        
        
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import time
# import threading
# import playsound
# from ultralytics import YOLO

# # ---- Define Regions ----
# RedLight = np.array([[998, 125], [998, 155], [972, 152], [970, 127]])
# GreenLight = np.array([[971, 200], [996, 200], [1001, 228], [971, 230]])
# ROI = np.array([[910, 372], [388, 365], [338, 428], [917, 441]])

# # ---- Load YOLOv8 Model ----
# model = YOLO("yolov8m.pt")
# coco = model.model.names
# TargetLabels = ["bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]

# # ---- Helper Functions ----
# def is_region_light(image, polygon, brightness_threshold=128):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     mask = np.zeros_like(gray_image)
#     cv2.fillPoly(mask, [np.array(polygon)], 255)
#     roi = cv2.bitwise_and(gray_image, gray_image, mask=mask)
#     mean_brightness = cv2.mean(roi, mask=mask)[0]
#     return mean_brightness > brightness_threshold

# def draw_text_with_background(frame, text, position, font, scale, text_color, background_color, border_color, thickness=2, padding=5):
#     (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
#     x, y = position
#     cv2.rectangle(frame, (x - padding, y - text_height - padding), (x + text_width + padding, y + baseline + padding), background_color, cv2.FILLED)
#     cv2.rectangle(frame, (x - padding, y - text_height - padding), (x + text_width + padding, y + baseline + padding), border_color, thickness)
#     cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, lineType=cv2.LINE_AA)

# # ---- Open Video ----
# cap = cv2.VideoCapture("tr.mp4")

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         print("Video finished.")
#         break

#     frame = cv2.resize(frame, (1100, 700))
#     cv2.polylines(frame, [RedLight], True, [0, 0, 255], 1)
#     cv2.polylines(frame, [GreenLight], True, [0, 255, 0], 1)
#     cv2.polylines(frame, [ROI], True, [255, 0, 0], 2)

#     results = model.predict(frame, conf=0.75)
#     for result in results:
#         boxes = result.boxes.xyxy
#         confs = result.boxes.conf
#         classes = result.boxes.cls

#         for box, conf, cls in zip(boxes, confs, classes):
#             label = coco[int(cls)]
#             if label in TargetLabels:
#                 x1, y1, x2, y2 = map(int, box)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 255, 0], 2)
#                 draw_text_with_background(
#                     frame,
#                     f"{label.capitalize()}, conf: {float(conf)*100:.2f}%",
#                     (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_COMPLEX,
#                     0.6,
#                     (255, 255, 255),
#                     (0, 0, 0),
#                     (0, 0, 255)
#                 )

#                 # --- Violation Detection ---
#                 if is_region_light(frame, RedLight):
#                     if cv2.pointPolygonTest(ROI, (x1, y1), False) >= 0 or cv2.pointPolygonTest(ROI, (x2, y2), False) >= 0:
#                         draw_text_with_background(
#                             frame,
#                             f"The {label.capitalize()} violated the traffic signal!",
#                             (10, 30),
#                             cv2.FONT_HERSHEY_COMPLEX,
#                             0.6,
#                             (255, 255, 255),
#                             (0, 0, 0),
#                             (0, 0, 255)
#                         )

#                         # Highlight violation
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 2)
#                         cv2.polylines(frame, [ROI], True, [0, 0, 255], 2)

#                         # Play alert sound (non-blocking)
#                         threading.Thread(target=playsound.playsound, args=("alert.mp3",), daemon=True).start()

#     cv2.imshow("Traffic Violation Detection", frame)
#     if cv2.waitKey(1) == 27:  # ESC to quit
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import time
import threading
import playsound
from ultralytics import YOLO
from pydub import AudioSegment
from pydub.generators import Sine
from pydub.playback import play
import threading

# ---- Define Regions ----
RedLight = np.array([[998, 125], [998, 155], [972, 152], [970, 127]])
GreenLight = np.array([[971, 200], [996, 200], [1001, 228], [971, 230]])
ROI = np.array([[910, 372], [388, 365], [338, 428], [917, 441]])

# ---- Load YOLOv8 Model ----
model = YOLO("yolov8m.pt")
coco = model.model.names
TargetLabels = ["bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]

# ---- Global Variables for Siren Control ----
is_siren_playing = False
siren_thread = None

def play_siren_loop():
    global is_siren_playing
    while is_siren_playing:
        playsound.playsound("siren.mp3", block=True)  # Use a siren sound file

def start_siren():
    global is_siren_playing, siren_thread
    if not is_siren_playing:
        is_siren_playing = True
        siren_thread = threading.Thread(target=play_siren_loop, daemon=True)
        siren_thread.start()

def stop_siren():
    global is_siren_playing
    is_siren_playing = False

# ---- Helper Functions ----
def is_region_light(image, polygon, brightness_threshold=128):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_image)
    cv2.fillPoly(mask, [np.array(polygon)], 255)
    roi = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    mean_brightness = cv2.mean(roi, mask=mask)[0]
    return mean_brightness > brightness_threshold

def draw_text_with_background(frame, text, position, font, scale, text_color, background_color, border_color, thickness=2, padding=5):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x - padding, y - text_height - padding), (x + text_width + padding, y + baseline + padding), background_color, cv2.FILLED)
    cv2.rectangle(frame, (x - padding, y - text_height - padding), (x + text_width + padding, y + baseline + padding), border_color, thickness)
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, lineType=cv2.LINE_AA)

def generate_siren():
    while True:
        # Rising tone
        rising = Sine(800).to_audio_segment(duration=500).fade_in(100).fade_out(100)
        # Falling tone
        falling = Sine(1200).to_audio_segment(duration=500).fade_in(100).fade_out(100)
        siren = rising + falling
        play(siren)

# ---- Open Video ----
cap = cv2.VideoCapture("tr.mp4")
violation_detected = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video finished.")
        break

    frame = cv2.resize(frame, (1100, 700))
    cv2.polylines(frame, [RedLight], True, [0, 0, 255], 1)
    cv2.polylines(frame, [GreenLight], True, [0, 255, 0], 1)
    cv2.polylines(frame, [ROI], True, [255, 0, 0], 2)

    current_violation = False

    results = model.predict(frame, conf=0.75)
    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        classes = result.boxes.cls

        for box, conf, cls in zip(boxes, confs, classes):
            label = coco[int(cls)]
            if label in TargetLabels:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 255, 0], 2)
                draw_text_with_background(
                    frame,
                    f"{label.capitalize()}, conf: {float(conf)*100:.2f}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.6,
                    (255, 255, 255),
                    (0, 0, 0),
                    (0, 0, 255)
                )
                # --- Violation Detection ---
                if is_region_light(frame, RedLight):
                    if cv2.pointPolygonTest(ROI, (x1, y1), False) >= 0 or cv2.pointPolygonTest(ROI, (x2, y2), False) >= 0:
                        current_violation = True
                        draw_text_with_background(
                            frame,
                            f"The {label.capitalize()} violated the traffic signal!",
                            (10, 30),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.6,
                            (255, 255, 255),
                            (0, 0, 0),
                            (0, 0, 255)
                        )

                        # Highlight violation
                        cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 2)
                        cv2.polylines(frame, [ROI], True, [0, 0, 255], 2)

    # Siren control logic
    if current_violation:
        if not violation_detected:  # New violation detected
            start_siren()
            violation_detected = True
    else:
        if violation_detected:  # Violation ended
            stop_siren()
            violation_detected = False

    cv2.imshow("Traffic Violation Detection", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        stop_siren()
        break

stop_siren()  # Ensure siren stops when exiting
cap.release()
cv2.destroyAllWindows()