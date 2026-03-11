import cv2
import torch
from ultralytics import YOLO
import os
import time
import csv
from datetime import datetime

# -----------------------------
# 1. Setup Model
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    model = YOLO("yolo11n.pt")
    model.to(device)
    print(f"✅ YOLO Loaded on {device.upper()}")
except Exception as e:
    print(f"⚠️ Model loading failed: {e}")
    exit()

# -----------------------------
# 2. Reference Logic
# -----------------------------
VEHICLE_WEIGHTS = {2: 2, 3: 1, 5: 4, 7: 4}
MIN_GREEN = 10
MAX_GREEN = 60


def run_smart_traffic(video_source, mode_name):

    if isinstance(video_source, str) and not os.path.exists(video_source):
        print(f"❌ Error: {video_source} not found.")
        return

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"❌ Error: Could not start video source {video_source}")
        return

    print(f"🚀 Running {mode_name} Scenario... (Press 'q' or 'Esc' to STOP)")

    start_time = time.time()
    peak_weighted_score = 0
    total_frames = 0

    try:

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            if isinstance(video_source, str):
                time.sleep(0.03)

            total_frames += 1

            # Faster resolution
            frame = cv2.resize(frame, (960, 540))

            # -----------------------------
            # ROI ZONE
            # -----------------------------
            zx1, zy1, zx2, zy2 = 250, 250, 700, 500
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 255, 0), 2)

            # -----------------------------
            # YOLO INFERENCE
            # -----------------------------
            results = model.predict(
                frame,
                classes=[2, 3, 5, 7],
                device=device,
                conf=0.3,
                verbose=False
            )[0]

            current_weighted_score = 0
            v_counts = {2: 0, 3: 0, 5: 0, 7: 0}
            emergency_detected = False

            if results.boxes is not None:

                for box in results.boxes:

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls.item())

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if zx1 < cx < zx2 and zy1 < cy < zy2:

                        weight = VEHICLE_WEIGHTS.get(cls, 1)

                        current_weighted_score += weight
                        v_counts[cls] += 1

                        if mode_name == "EMERGENCY" and cls in [5, 7]:
                            emergency_detected = True

                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            peak_weighted_score = max(peak_weighted_score, current_weighted_score)

            # -----------------------------
            # SIGNAL LOGIC
            # -----------------------------
            suggested_green = MIN_GREEN + (current_weighted_score * 2)
            suggested_green = max(MIN_GREEN, min(suggested_green, MAX_GREEN))

            sig_color = (0, 0, 255)
            status_msg = "SIGNAL: RED"

            if emergency_detected:
                sig_color = (255, 0, 0)
                status_msg = "🚨 EMERGENCY PRIORITY 🚨"
                suggested_green = 99

            elif current_weighted_score > 12:
                sig_color = (0, 255, 0)
                status_msg = "🔥 HIGH TRAFFIC"

            elif current_weighted_score > 4:
                sig_color = (0, 165, 255)
                status_msg = "⚡ MODERATE TRAFFIC"

            else:
                status_msg = "❄️ LOW TRAFFIC"

            # -----------------------------
            # DASHBOARD
            # -----------------------------
            overlay = frame.copy()

            cv2.rectangle(overlay, (0, 0), (960, 80), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            cv2.putText(frame, f"MODE: {mode_name}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, status_msg, (320, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, sig_color, 2)

            cv2.putText(frame, f"DENSITY: {current_weighted_score}",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            cv2.putText(frame, f"TIMER: {suggested_green}s",
                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)

            cv2.circle(frame, (900, 40), 25, sig_color, -1)

            display = cv2.addWeighted(results.plot(), 0.6, frame, 0.4, 0)

            cv2.imshow("SMART TRAFFIC AI", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                print("🛑 Interrupted by user")
                break

    finally:

        cap.release()
        cv2.destroyAllWindows()
        print("📷 Camera released")

    # -----------------------------
    # LOGGING
    # -----------------------------
    duration = round(time.time() - start_time, 2)

    report_file = "traffic_reports.csv"

    file_exists = os.path.isfile(report_file)

    with open(report_file, 'a', newline='') as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Timestamp", "Scenario", "Peak_Density", "Duration"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            mode_name,
            peak_weighted_score,
            duration
        ])

    print("\n--- SESSION SUMMARY ---")
    print(f"⏱ Duration: {duration}s")
    print(f"🚗 Peak Density: {peak_weighted_score}")
    print(f"📂 Report saved: {report_file}")


# -----------------------------
# MENU
# -----------------------------
if __name__ == "__main__":

    while True:

        print("\n--- AI TRAFFIC SYSTEM SELECTOR ---")
        print("1. Normal Flow")
        print("2. Heavy Traffic")
        print("3. Emergency Mode")
        print("4. Webcam")
        print("5. Exit")

        choice = input("\nEnter Choice (1-5): ")

        if choice == '1':
            run_smart_traffic("normal_traffic.mp4", "NORMAL")

        elif choice == '2':
            run_smart_traffic("heavy_traffic.mp4", "HEAVY")

        elif choice == '3':
            run_smart_traffic("emergency.mp4", "EMERGENCY")

        elif choice == '4':
            run_smart_traffic(0, "LIVE_CAMERA")

        elif choice == '5':
            print("👋 Exiting system")
            break

        else:
            print("❌ Invalid choice")