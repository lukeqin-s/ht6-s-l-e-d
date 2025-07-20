import os
import sys
import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO
from trilobot import Trilobot
import mediapipe as mp

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model file')
parser.add_argument('--source', required=True, help='Image/video/camera source')
parser.add_argument('--resolution', default=None, help='Resolution WxH (e.g. 640x480)')
args = parser.parse_args()

model_path = args.model
img_source = args.source
user_res = args.resolution

# Check model file
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

# Load YOLO model
model = YOLO(model_path, task='detect')
labels = model.names

# Determine source type
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    print('Folder input not supported for robot following. Exiting.')
    sys.exit(0)
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Set up video/camera source
if source_type == 'image':
    print('Image input not supported for robot following. Exiting.')
    sys.exit(0)
elif source_type == 'video':
    cap = cv2.VideoCapture(img_source)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'usb':
    cap = cv2.VideoCapture(usb_idx)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Initialize robot
robot = Trilobot()

# Initialize MediaPipe Hands
cmp_hands = mp.solutions.hands
hands = cmp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Main loop: robot follows the largest detected person
last_person_time = time.time()  # Track last time a person was seen
spin_mode = False  # Whether robot is spinning to scan
last_seen_direction = None  # Track which side the person was last seen on

frame_count = 0
last_hand_detected = False
last_hand_box = None

while True:
    # Get frame from source
    if source_type == 'video' or source_type == 'usb':
        ret, frame = cap.read()
        if not ret or frame is None:
            print('No more frames or camera error. Exiting.')
            break
    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if frame is None:
            print('Camera error. Exiting.')
            break

    # Resize if needed
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # --- MediaPipe Hand Detection every 5 frames ---
    frame_count += 1
    if frame_count % 5 == 0:
        hand_detected = False
        hand_box = None
        hand_box_area = 0
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(frame_rgb)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Get bounding box from landmarks
                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                xmin, xmax = int(min(x_coords)), int(max(x_coords))
                ymin, ymax = int(min(y_coords)), int(max(y_coords))
                area = (xmax - xmin) * (ymax - ymin)
                if area > hand_box_area:
                    hand_box_area = area
                    hand_box = (xmin, ymin, xmax, ymax)
            # Set threshold for 'big enough' hand (e.g., 1/6 of frame area)
            hand_area_thresh = (frame.shape[0] * frame.shape[1]) // 6
            if hand_box_area > hand_area_thresh:
                hand_detected = True
        last_hand_detected = hand_detected
        last_hand_box = hand_box
    else:
        hand_detected = last_hand_detected
        hand_box = last_hand_box

    # Run YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Find largest person
    person_box = None
    person_box_area = 0
    frame_center_x = frame.shape[1] // 2
    for i in range(len(detections)):
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        if classname == 'person':
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            area = (xmax - xmin) * (ymax - ymin)
            if area > person_box_area:
                person_box_area = area
                person_box = (xmin, ymin, xmax, ymax)

    # --- Robot following logic ---
    # If a large hand is detected, stop the robot and skip person following
    if hand_detected:
        robot.coast()
        robot.stop()
        if hand_box is not None:
            cv2.rectangle(frame, (hand_box[0], hand_box[1]), (hand_box[2], hand_box[3]), (255,0,0), 2)
            cv2.putText(frame, 'Hand Detected', (hand_box[0], hand_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
    elif person_box is not None:
        last_person_time = time.time()  # Update last seen time
        spin_mode = False  # Stop spinning if we see a person
        xmin, ymin, xmax, ymax = person_box
        box_center_x = (xmin + xmax) // 2
        box_width = xmax - xmin
        center_thresh = resW // 4  # pixels, how close to center is 'centered'
        close_thresh = frame.shape[1] // 1.5  # stop if person is close (box is half the frame width)
        # Track which side the person is on for lost detection
        if box_center_x < frame_center_x - center_thresh:
            last_seen_direction = 'left'
        elif box_center_x > frame_center_x + center_thresh:
            last_seen_direction = 'right'
        else:
            last_seen_direction = 'center'
        if box_width > close_thresh:
            robot.coast()
            robot.stop()  # Person is close, stop
        else:
            if box_center_x < frame_center_x - center_thresh:
                robot.coast()
                robot.curve_forward_left(0.75)  # Person left, turn left
            elif box_center_x > frame_center_x + center_thresh:
                robot.coast()
                robot.curve_forward_right(0.75)  # Person right, turn right
            else:
                robot.forward(1.0)  # Person centered, go forward
    else:
        # If no person, check if it's time to spin
        if time.time() - last_person_time > 3.0:
            # Spin in the direction where the person was last seen
            spin_mode = True
            if last_seen_direction == 'left':
                robot.coast()
                robot.turn_left(0.75)
            elif last_seen_direction == 'right':
                robot.coast()
                robot.turn_right(0.75)
            else:
                robot.coast()
                robot.turn_right(0.75)  # Default to right if unknown
        else:
            robot.coast()
            robot.stop()  # No person, stop

    # Show frame with bounding box for debug (optional)
    if person_box is not None:
        cv2.rectangle(frame, (person_box[0], person_box[1]), (person_box[2], person_box[3]), (0,255,0), 2)
    cv2.imshow('Robot Following', frame)
    key = cv2.waitKey(5)
    if key == ord('q') or key == ord('Q'):
        break

# Cleanup
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
cv2.destroyAllWindows()
