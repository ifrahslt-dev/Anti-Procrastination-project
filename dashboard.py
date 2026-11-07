

import cv2
import time
import os
import pygame

from collections import Counter
from ultralytics import YOLO


# =================================
# 1. LOAD AI MODEL AND ALARM
# =================================

print("Loading YOLO model...")

model = YOLO("yolo26n.pt")

pygame.mixer.init()

alarm = pygame.mixer.Sound("alarm.mp3")
alarm_channel = None

print("AI model and alarm loaded!")


# =================================
# 2. OPEN WEBCAM
# =================================

camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not camera.isOpened():
    print("Error: Cannot access webcam.")
    pygame.mixer.quit()
    exit()


# =================================
# 3. INITIALIZE VARIABLES
# =================================

previous_time = time.time()

screenshot_number = 0

confidence_threshold = 0.40

print("\nComputer Vision Dashboard Started!")
print("Q = Quit")
print("S = Save Screenshot")
print("+ = Increase confidence")
print("- = Decrease confidence")


# =================================
# 4. MAIN PROGRAM LOOP
# =================================

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera frame unavailable.")
            break


        # =================================
        # 5. RUN OBJECT DETECTION
        # =================================

        results = model(
            frame,
            conf=confidence_threshold,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        output = result.plot()


        # =================================
        # 6. COUNT DETECTED OBJECTS
        # =================================

        object_names = []

        phone_detected = False

        for box in result.boxes:

            class_id = int(box.cls[0])

            object_name = model.names[class_id]

            confidence = float(box.conf[0])

            object_names.append(object_name)

            # Check for cellphone
            if object_name == "cell phone":
                phone_detected = True


        object_counts = Counter(object_names)

        total_objects = len(object_names)


        # =================================
        # 7. CELLPHONE ALARM
        # =================================

        if phone_detected:

            if alarm_channel is None or not alarm_channel.get_busy():
                alarm_channel = alarm.play(loops=-1)

            cv2.putText(
                output,
                "WARNING: CELLPHONE DETECTED!",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3
            )

        else:

            if alarm_channel is not None:
                alarm_channel.stop()
                alarm_channel = None


        # =================================
        # 8. CALCULATE FPS
        # =================================

        current_time = time.time()

        elapsed_time = current_time - previous_time

        fps = 1 / elapsed_time if elapsed_time > 0 else 0

        previous_time = current_time


        # =================================
        # 9. DISPLAY DASHBOARD INFORMATION
        # =================================

        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"Objects detected: {total_objects}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Confidence threshold: {confidence_threshold:.2f}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # =================================
        # 10. DISPLAY OBJECT COUNTS
        # =================================

        y_position = 145

        for name, count in object_counts.items():

            text = f"{name}: {count}"

            cv2.putText(
                output,
                text,
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 200, 0),
                2
            )

            y_position += 30


        # =================================
        # 11. DISPLAY CAMERA WINDOW
        # =================================

        cv2.imshow("Computer Vision Dashboard", output)


        # =================================
        # 12. KEYBOARD CONTROLS
        # =================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):

            screenshot_number += 1

            filename = f"screenshot_{screenshot_number}.jpg"

            cv2.imwrite(filename, output)

            print("Screenshot saved:", filename)

        elif key in (ord("+"), ord("=")):

            confidence_threshold = min(
                0.95,
                confidence_threshold + 0.05
            )

            print("Confidence:", confidence_threshold)

        elif key == ord("-"):

            confidence_threshold = max(
                0.05,
                confidence_threshold - 0.05
            )

            print("Confidence:", confidence_threshold)


# =================================
# 13. CLEANUP
# =================================

finally:

    if alarm_channel is not None:
        alarm_channel.stop()

    camera.release()

    cv2.destroyAllWindows()

    pygame.mixer.quit()

    print("Dashboard closed.")

