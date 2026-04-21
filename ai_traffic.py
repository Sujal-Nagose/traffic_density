"""
3-LED Smart Traffic Demo — SERIAL FIXED VERSION
=================================================
SERIAL FIXES APPLIED:
  [FIX 1] Auto-detects Arduino COM port — no more hardcoding COM7
  [FIX 2] Added time.sleep(2) after Serial() — Arduino needs reset time
  [FIX 3] send_cmd() now flushes output buffer after every write
  [FIX 4] Serial errors no longer crash the program — graceful fallback
  [FIX 5] Port is rechecked on reconnection attempts
  [FIX 6] arduino.write() wrapped in try/except everywhere
  [FIX 7] arduino.in_waiting cleared on connect to avoid stale bytes
  [FIX 8] Baud rate matches Arduino sketch exactly (9600)

DETECTION FIXES (from previous session):
  [FIX A] yolo11s.pt instead of yolo11n
  [FIX B] conf=0.45 (was 0.25)
  [FIX C] iou=0.45 added (removes duplicate boxes)
  [FIX D] EMERGENCY_CLASSES = set() — buses no longer trigger ambulance
  [FIX E] Per-class confidence override for bus/truck

ARDUINO SKETCH (paste into Arduino IDE, upload, CLOSE IDE):
─────────────────────────────────────────────────────────────
  #define G_PIN 9
  #define Y_PIN 6
  #define R_PIN 3

  void setup() {
    Serial.begin(9600);
    pinMode(G_PIN, OUTPUT);
    pinMode(Y_PIN, OUTPUT);
    pinMode(R_PIN, OUTPUT);
    allOff();
    digitalWrite(R_PIN, HIGH);
  }

  void loop() {
    if (Serial.available() > 0) {
      char c = Serial.read();
      allOff();
      if      (c == 'G') digitalWrite(G_PIN, HIGH);
      else if (c == 'Y') digitalWrite(Y_PIN, HIGH);
      else if (c == 'R') digitalWrite(R_PIN, HIGH);
      else if (c == 'E') emergencyFlash();
    }
  }

  void allOff() {
    digitalWrite(G_PIN, LOW);
    digitalWrite(Y_PIN, LOW);
    digitalWrite(R_PIN, LOW);
  }

  void emergencyFlash() {
    for (int i = 0; i < 8; i++) {
      digitalWrite(G_PIN, HIGH); delay(120);
      digitalWrite(G_PIN, LOW);  delay(120);
    }
    digitalWrite(G_PIN, HIGH);
  }
─────────────────────────────────────────────────────────────

WIRING:
  Arduino Pin 9 → 220Ω → Green  LED (+) → GND
  Arduino Pin 6 → 220Ω → Yellow LED (+) → GND
  Arduino Pin 3 → 220Ω → Red    LED (+) → GND
"""

import cv2
import torch
import time
import os
import csv
import serial
import serial.tools.list_ports          # FIX 1: for auto-detection
from ultralytics import YOLO
from datetime import datetime


# ═══════════════════════════════════════════════════════════
# SECTION 1 — SERIAL COMMUNICATION  (all 8 serial fixes here)
# ═══════════════════════════════════════════════════════════

def find_arduino_port():
    """
    FIX 1: Scan all COM ports and return the one that looks like an Arduino.
    Works on Windows (COM3, COM7...) and Linux (/dev/ttyUSB0, /dev/ttyACM0).
    Never hardcodes a port — always discovers it fresh.
    """
    ports = list(serial.tools.list_ports.comports())

    if not ports:
        print("  No COM ports found at all. Is Arduino plugged in via USB?")
        return None

    print("  Available serial ports:")
    for p in ports:
        print(f"    {p.device}  |  {p.description}")

    # Keywords that appear in Arduino / CH340 / CP2102 descriptions
    arduino_keywords = [
        'ch340', 'ch341',           # Cheap Arduino clones (most common in India)
        'arduino',                  # Official Arduino
        'usb serial',               # Generic USB-serial
        'cp210',                    # CP2102 chip (some Nanos)
        'ftdi',                     # FTDI chip (older Arduinos)
        'ttyusb', 'ttyacm',         # Linux device names
    ]

    for p in ports:
        desc_lower = p.description.lower()
        dev_lower  = p.device.lower()
        if any(kw in desc_lower or kw in dev_lower for kw in arduino_keywords):
            print(f"\n  Detected Arduino on: {p.device} ({p.description})")
            return p.device

    # If no keyword match, return the last port as a fallback and warn
    fallback = ports[-1].device
    print(f"\n  No Arduino keyword match. Trying last port as fallback: {fallback}")
    print("  If this is wrong, set FORCE_PORT below to your actual port.")
    return fallback


# Set this to a specific port string like 'COM7' or '/dev/ttyUSB0'
# to skip auto-detection. Leave as None to auto-detect.
FORCE_PORT = None   # ← e.g. FORCE_PORT = 'COM7'


def connect_arduino():
    """
    FIX 1+2+7: Find port, open serial, wait for Arduino reset,
    clear any stale bytes in the buffer.
    Returns serial.Serial object, or None if connection failed.
    """
    port = FORCE_PORT if FORCE_PORT else find_arduino_port()

    if port is None:
        print("\n  Running in SIMULATION MODE — no hardware connected.")
        print("  LED states will be printed to terminal only.\n")
        return None

    try:
        ard = serial.Serial(
            port=port,
            baudrate=9600,          # FIX 8: must match Arduino sketch exactly
            timeout=1,
            write_timeout=1,        # FIX 3: prevents write() from hanging
        )

        # FIX 2: CRITICAL — Arduino resets when serial opens.
        # Must wait at least 1.5s before sending any commands.
        print(f"  Waiting for Arduino to reset on {port}...")
        time.sleep(2)

        # FIX 7: Clear any garbage bytes left in the receive buffer
        ard.reset_input_buffer()
        ard.reset_output_buffer()

        # Send boot state (red LED on)
        ard.write(b'R')
        ard.flush()                 # FIX 3: force bytes out immediately

        print(f"  Arduino ready on {port}\n")
        return ard

    except serial.SerialException as e:
        print(f"\n  Could not open {port}: {e}")
        print("  Common causes:")
        print("    1. Arduino IDE Serial Monitor is still open — close it")
        print("    2. Wrong port — check Device Manager")
        print("    3. Driver not installed — install CH340 driver")
        print("  Running in simulation mode.\n")
        return None


# Connect on startup
arduino = connect_arduino()


def send_cmd(cmd: bytes, label: str = ""):
    """
    FIX 3+4+6: Send one byte to Arduino with error handling.
    - Flushes after write so bytes leave immediately
    - Catches SerialException and tries to reconnect once
    - Falls back to terminal display if hardware not available
    """
    global arduino

    symbol = {b'G': "GREEN", b'Y': "YELLOW", b'R': "RED", b'E': "EMERGENCY"}.get(cmd, "?")
    print(f"    LED → {symbol}  {label}", end="")

    if arduino is None:
        print()
        return

    try:
        arduino.write(cmd)
        arduino.flush()             # FIX 3: flush output buffer immediately
        print()

    except serial.SerialException as e:
        print(f"  (serial error: {e})")
        print("  Attempting to reconnect Arduino...")
        try:
            arduino.close()
        except Exception:
            pass
        arduino = connect_arduino()   # FIX 5: re-discover port on reconnect
        if arduino:
            try:
                arduino.write(cmd)
                arduino.flush()
                print("  Reconnected and command sent.")
            except Exception:
                print("  Reconnect failed. Continuing without hardware.")
                arduino = None


# ═══════════════════════════════════════════════════════════
# SECTION 2 — AI MODEL  (detection fixes A-E)
# ═══════════════════════════════════════════════════════════

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading YOLO11s on {device.upper()}...")
print("(Downloads ~18MB on first run)\n")

# FIX A: yolo11s not yolo11n
model = YOLO("yolo11s.pt").to(device)
print(f"YOLO11s ready on {device.upper()}\n")


# ═══════════════════════════════════════════════════════════
# SECTION 3 — SIGNAL CONFIGURATION
# ═══════════════════════════════════════════════════════════

VEHICLE_WEIGHTS = {
    3: 1,    # motorcycle
    2: 3,    # car
    5: 8,    # bus
    7: 8,    # truck
}

# FIX D: EMPTY — buses/trucks NO LONGER trigger emergency
# This was the main bug: {5, 7} meant every bus = ambulance
EMERGENCY_CLASSES = set()
# Uncomment below when you have a custom-trained ambulance model:
# EMERGENCY_CLASSES = {8}   # where 8 is your custom ambulance class ID

# FIX E: Higher confidence for bus/truck (they look similar — need more certainty)
CLASS_CONF_OVERRIDE = {
    5: 0.55,    # bus
    7: 0.55,    # truck
    2: 0.45,    # car
    3: 0.40,    # motorcycle
}
DEFAULT_CONF = 0.45     # FIX B: was 0.25
NMS_IOU      = 0.45     # FIX C: removes duplicate boxes on same vehicle

MIN_GREEN  = 15
MAX_GREEN  = 90
YELLOW_SEC = 5
RED_GAP    = 3

# Tighter ROI — ignores distant/noisy detections
ROI_X1 = 0.10
ROI_Y1 = 0.35    # was 0.25 — now ignores top 35% (skyline noise)
ROI_X2 = 0.90
ROI_Y2 = 0.92


# ═══════════════════════════════════════════════════════════
# SECTION 4 — DETECTION ENGINE
# ═══════════════════════════════════════════════════════════

def analyse_frame(frame):
    """
    Run YOLO on one frame.
    Returns: (score, is_emergency, detections_list, yolo_results)
    """
    h, w = frame.shape[:2]
    rx1 = int(w * ROI_X1); ry1 = int(h * ROI_Y1)
    rx2 = int(w * ROI_X2); ry2 = int(h * ROI_Y2)

    detect_classes = list(set(list(VEHICLE_WEIGHTS.keys()) + list(EMERGENCY_CLASSES)))

    results = model.predict(
        frame,
        classes=detect_classes,
        conf=DEFAULT_CONF,          # FIX B
        iou=NMS_IOU,                # FIX C
        verbose=False,
        device=device,
    )[0]

    score     = 0
    emergency = False
    dets      = []

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten()
            cls  = int(box.cls.item())
            conf = float(box.conf.item())

            # FIX E: enforce per-class confidence threshold
            if conf < CLASS_CONF_OVERRIDE.get(cls, DEFAULT_CONF):
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Only count vehicles whose center is in the ROI
            if not (rx1 < cx < rx2 and ry1 < cy < ry2):
                continue

            # FIX D: emergency only from EMERGENCY_CLASSES — never from bus/truck
            if cls in EMERGENCY_CLASSES:
                emergency = True
            elif cls in VEHICLE_WEIGHTS:
                score += VEHICLE_WEIGHTS[cls]
                dets.append({"cls": cls, "conf": conf, "cx": cx, "cy": cy})

    return score, emergency, dets, results


def score_to_phase(score, is_emergency):
    """Map density score → (phase_name, duration_seconds, arduino_cmd)"""
    if is_emergency:
        return "EMERGENCY", 99, b'E'
    if score > 30:
        return "GREEN", MAX_GREEN, b'G'
    if score > 15:
        green_t = max(MIN_GREEN, min(MIN_GREEN + score * 2, MAX_GREEN))
        return "GREEN", green_t, b'G'
    return "RED", MIN_GREEN, b'R'


# ═══════════════════════════════════════════════════════════
# SECTION 5 — DISPLAY OVERLAY
# ═══════════════════════════════════════════════════════════

CLASS_NAMES = {2: "car", 3: "motorbike", 5: "BUS", 7: "TRUCK"}
LED_COLORS  = {
    "GREEN":     (30, 210,  30),
    "YELLOW":    ( 0, 200, 220),
    "RED":       (40,  40, 210),
    "EMERGENCY": (30, 120, 255),
}

def draw_overlay(frame, score, phase, green_sec, remaining, detections, yolo_results):
    h, w = frame.shape[:2]

    # ROI rectangle
    cv2.rectangle(frame,
                  (int(w * ROI_X1), int(h * ROI_Y1)),
                  (int(w * ROI_X2), int(h * ROI_Y2)),
                  (255, 220, 0), 2)
    cv2.putText(frame, "Detection zone",
                (int(w * ROI_X1) + 8, int(h * ROI_Y1) + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 0), 1)

    # Dark dashboard bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (8, 8, 8), -1)
    frame = cv2.addWeighted(overlay, 0.70, frame, 0.30, 0)

    col = LED_COLORS.get(phase, (200, 200, 200))

    cv2.putText(frame, "AI TRAFFIC DEMO",
                (18, 38), cv2.FONT_HERSHEY_DUPLEX, 0.80, (255, 255, 255), 2)
    cv2.putText(frame, f"PHASE: {phase}",
                (18, 78), cv2.FONT_HERSHEY_DUPLEX, 0.80, col, 2)
    cv2.putText(frame, f"DENSITY: {score}",
                (w // 2 - 90, 38), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 230, 230), 2)
    cv2.putText(frame, f"GREEN: {green_sec}s",
                (w // 2 - 90, 78), cv2.FONT_HERSHEY_DUPLEX, 0.65, (80, 255, 80), 2)

    # Countdown on right
    cv2.putText(frame, f"{remaining}s",
                (w - 110, 75), cv2.FONT_HERSHEY_DUPLEX, 1.4, col, 3)

    # LED circle indicator
    cv2.circle(frame, (w - 45, 58), 30, col, -1)
    cv2.circle(frame, (w - 45, 58), 30, (255, 255, 255), 1)
    lbl = {"GREEN": "G", "YELLOW": "Y", "RED": "R", "EMERGENCY": "E"}.get(phase, "?")
    cv2.putText(frame, lbl, (w - 57, 66),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0), 2)

    # Vehicle breakdown at bottom
    ovr2 = frame.copy()
    cv2.rectangle(ovr2, (0, h - 50), (w, h), (8, 8, 8), -1)
    frame = cv2.addWeighted(ovr2, 0.65, frame, 0.35, 0)
    cls_counts = {}
    for d in detections:
        cls_counts[d["cls"]] = cls_counts.get(d["cls"], 0) + 1
    info = "  |  ".join(f"{CLASS_NAMES.get(c, str(c))}: {n}"
                        for c, n in cls_counts.items()) or "No vehicles in zone"
    cv2.putText(frame, info, (16, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    # Blend YOLO boxes
    if yolo_results is not None:
        frame = cv2.addWeighted(yolo_results.plot(), 0.38, frame, 0.62, 0)

    return frame


# ═══════════════════════════════════════════════════════════
# SECTION 6 — MAIN SIGNAL LOOP
# ═══════════════════════════════════════════════════════════

def run_3led_demo(video_source, label="LANE"):
    """
    Main loop: analyse → send signal → display → cycle.
    Handles video files, looping, and live RTSP/IP streams.
    """
    if isinstance(video_source, str) and not video_source.startswith("http"):
        if not os.path.exists(video_source):
            print(f"File not found: {video_source}")
            return

    print(f"\nStarting | Lane: {label} | Source: {video_source}")
    print("Press Q in the window to quit.\n")

    cap      = cv2.VideoCapture(video_source)
    log_rows = []
    cycle    = 0

    send_cmd(b'R', "(boot — red)\n")

    def safe_read():
        """Read frame, handle stream drops and end-of-file."""
        ok, frm = cap.read()
        if not ok:
            if isinstance(video_source, str) and "http" in video_source:
                time.sleep(0.5)
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frm = cap.read()
            if not ok:
                return None
        return cv2.resize(frm, (1280, 720))

    try:
        while True:
            # ── Grab frame for initial analysis ────────────────
            frame = safe_read()
            if frame is None:
                continue

            score, is_emg, dets, results = analyse_frame(frame)
            phase, green_t, hw_cmd       = score_to_phase(score, is_emg)
            cycle += 1

            print(f"\n[Cycle {cycle}] score={score}  emergency={is_emg}"
                  f"  phase={phase}  green={green_t}s")

            # ── GREEN / EMERGENCY phase ─────────────────────────
            send_cmd(hw_cmd, f"  score={score}  timer={green_t}s\n")
            t_start = time.time()

            while True:
                elapsed   = time.time() - t_start
                remaining = max(0, int(green_t - elapsed))
                if remaining == 0:
                    break

                frame = safe_read()
                if frame is None:
                    continue

                # Re-analyse every frame for live density update
                live_score, _, live_dets, live_res = analyse_frame(frame)
                display = draw_overlay(frame.copy(), live_score, phase,
                                       green_t, remaining, live_dets, live_res)
                cv2.imshow("3-LED Traffic Demo", display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

            # ── YELLOW phase ────────────────────────────────────
            send_cmd(b'Y', f"  yellow {YELLOW_SEC}s\n")
            t_yellow = time.time()

            while time.time() - t_yellow < YELLOW_SEC:
                frame = safe_read()
                if frame is None:
                    continue
                rem_y = max(0, int(YELLOW_SEC - (time.time() - t_yellow)))
                display = draw_overlay(frame.copy(), score, "YELLOW",
                                       YELLOW_SEC, rem_y, dets, None)
                cv2.imshow("3-LED Traffic Demo", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

            # ── RED gap before re-analysis ──────────────────────
            send_cmd(b'R', f"  red gap {RED_GAP}s\n")
            t_red = time.time()

            while time.time() - t_red < RED_GAP:
                frame = safe_read()
                if frame is None:
                    continue
                rem_r = max(0, int(RED_GAP - (time.time() - t_red)))
                display = draw_overlay(frame.copy(), score, "RED",
                                       0, rem_r, [], None)
                cv2.imshow("3-LED Traffic Demo", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

            # Log
            log_rows.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "cycle": cycle, "score": score, "phase": phase,
                "green_sec": green_t, "emergency": is_emg,
            })

    except KeyboardInterrupt:
        print("\n\nStopped by operator.")

    finally:
        cap.release()
        send_cmd(b'R', "(safe shutdown)\n")
        cv2.destroyAllWindows()

        # Save log
        log_path = "traffic_log.csv"
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "cycle", "score", "phase", "green_sec", "emergency"
            ])
            if os.stat(log_path).st_size == 0:
                writer.writeheader()
            writer.writerows(log_rows)
        print(f"Log saved → {log_path}")

        if arduino:
            arduino.close()


# ═══════════════════════════════════════════════════════════
# SECTION 7 — SERIAL TEST UTILITY
# ═══════════════════════════════════════════════════════════

def test_serial_only():
    """
    Run this to verify Arduino serial communication works
    BEFORE running the full YOLO traffic system.
    Cycles all 3 LEDs without needing any camera or video.
    """
    print("\n=== SERIAL CONNECTION TEST ===")
    if arduino is None:
        print("No Arduino connected. Cannot run hardware test.")
        return

    test_sequence = [
        (b'R', "RED",       3),
        (b'G', "GREEN",     3),
        (b'Y', "YELLOW",    3),
        (b'E', "EMERGENCY", 4),
        (b'R', "RED",       2),
    ]

    print("Testing all LED states...")
    for cmd, name, duration in test_sequence:
        print(f"  Sending: {name} for {duration}s")
        send_cmd(cmd)
        time.sleep(duration)

    print("\nTest complete.")
    print("If all 3 LEDs lit up correctly → serial communication is working.")
    print("If LEDs did NOT light up → check wiring: Pin9=Green, Pin6=Yellow, Pin3=Red")


# ═══════════════════════════════════════════════════════════
# SECTION 8 — ENTRY POINT
# ═══════════════════════════════════════════════════════════

MOBILE_IP = "http://192.168.36.121:8080/video"   # IP Webcam app

if __name__ == "__main__":
    print("\n" + "═" * 50)
    print("  3-LED SMART TRAFFIC DEMO — SERIAL FIXED")
    print(f"  Model: YOLO11s | conf=0.45 | iou=0.45")
    hw_status = f"Arduino: {arduino.port}" if arduino else "Arduino: NOT CONNECTED (simulation)"
    print(f"  {hw_status}")
    print("═" * 50)
    print("\n  1. Test serial only (flash LEDs without camera)")
    print("  2. Run with video file")
    print("  3. Run with live mobile camera")
    print("  4. Exit")

    choice = input("\n  Choice (1-4): ").strip()

    if choice == "1":
        test_serial_only()

    elif choice == "2":
        path = input("  Video file path (Enter = emergency.mp4): ").strip()
        run_3led_demo(path or "emergency.mp4", "DEMO")

    elif choice == "3":
        run_3led_demo(MOBILE_IP, "LIVE_IOT")

    elif choice == "4":
        if arduino:
            arduino.close()
        print("  Goodbye.")
