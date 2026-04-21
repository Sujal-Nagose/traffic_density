# 🚦 AI-Based Smart Traffic Signal System

> Real-time vehicle detection using YOLOv11 to dynamically control traffic signals based on lane density. Built for Indian road conditions with Arduino LED demonstration.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![YOLOv11](https://img.shields.io/badge/YOLO-v11s-green?style=flat-square)
![Arduino](https://img.shields.io/badge/Arduino-Uno%2FNano-teal?style=flat-square&logo=arduino)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 What This Does

This system uses a camera feed and AI object detection to:

- Count vehicles in a lane using weighted density scoring
- Calculate how long the green signal should last based on traffic density
- Control physical LEDs via Arduino in real time
- Support single lane and 4-lane intersection modes
- Detect emergency vehicles (ambulance / fire brigade) for priority override

No fixed timers. The signal adapts to how many vehicles are actually present.

---

## 🧠 How the Logic Works

### Vehicle Weights

Each vehicle type contributes differently to the density score:

| Vehicle | YOLO Class | Weight | Reason |
|---------|-----------|--------|--------|
| Motorcycle | 3 | 1 | Smallest, least blocking |
| Car | 2 | 3 | Standard unit |
| Bus | 5 | 8 | Full lane width |
| Truck | 7 | 8 | Same as bus |

### Green Time Formula

```
Density Score  = Σ (vehicle_count × weight)
Green Time (s) = clamp(15 + score × 2,  min=15s,  max=90s)
```

**Examples:**
- 2 cars = score 6 → GREEN 27s
- 1 bus + 3 cars = score 17 → GREEN 49s
- 2 buses + 5 cars = score 31 → GREEN 90s (capped)

### Signal Cycle

```
RED (min 5s, re-analyses every frame)
  ↓  traffic detected
GREEN (15s – 90s based on density)
  ↓  timer ends or lane clears
YELLOW (fixed 3s)
  ↓
RED → repeat
```

### Emergency Override

When an emergency vehicle is detected in the lane, the system immediately forces GREEN for 99 seconds and the green LED flashes rapidly before holding steady.

---

## 🗂️ Project Structure

```
ai-traffic-signal/
│
├── single_lane_traffic.py        # Main file — single lane with 3 LEDs
├── smart_traffic_FINAL.py        # 4-lane intersection version
├── traffic_signal_arduino.ino    # Arduino sketch for LED control
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
└── logs/
    └── traffic_log.csv           # Auto-generated session logs
```

---

## ⚙️ Hardware Required

| Component | Quantity | Purpose |
|-----------|----------|---------|
| Arduino Uno or Nano | 1 | Controls the LEDs via serial |
| Green LED | 1 | Green signal |
| Yellow LED | 1 | Yellow signal |
| Red LED | 1 | Red signal |
| 220Ω resistor | 3 | One per LED (prevents burnout) |
| USB cable (data) | 1 | Connects Arduino to PC |
| Webcam or video file | 1 | Traffic input |
| Jumper wires | ~10 | Wiring |
| Breadboard | 1 | For prototyping |

**Optional for laptop-free deployment:**
- NVIDIA Jetson Orin Nano Super (~₹22,000) — runs YOLO at 30–60 FPS

---

## 🔌 Wiring Diagram

```
Arduino Pin 9  ──→  220Ω  ──→  Green  LED (+)  ──→  GND
Arduino Pin 6  ──→  220Ω  ──→  Yellow LED (+)  ──→  GND
Arduino Pin 3  ──→  220Ω  ──→  Red    LED (+)  ──→  GND
```

> ⚠️ Always connect the **long leg (+)** of the LED toward the resistor and the **short leg (−)** toward GND. Wrong polarity = LED won't light.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ai-traffic-signal.git
cd ai-traffic-signal
```

### 2. Install Python dependencies

```bash
pip install ultralytics opencv-python torch torchvision pyserial
```

> If you have an NVIDIA GPU, install CUDA-enabled PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) for better performance.

### 3. Upload Arduino sketch

1. Open `traffic_signal_arduino.ino` in Arduino IDE
2. Select your board: `Tools → Board → Arduino Uno`
3. Select your port: `Tools → Port → COMx`
4. Click **Upload**
5. **Close Arduino IDE completely** after uploading (it blocks the serial port)

### 4. Run the system

```bash
python single_lane_traffic.py
```

On first run, YOLO11s weights (~18MB) will download automatically.

---

## 🛠️ Configuration

Open `single_lane_traffic.py` and edit the settings at the top:

```python
VIDEO_SOURCE     = "traffic.mp4"  # video file, 0 for webcam, or IP cam URL
FORCE_PORT       = None           # e.g. 'COM7' to skip auto-detection
GREEN_THRESHOLD  = 5              # minimum score to trigger green
MIN_GREEN        = 15             # minimum green duration (seconds)
MAX_GREEN        = 90             # maximum green duration (seconds)
YELLOW_SEC       = 3              # yellow phase duration
RED_MIN          = 5              # minimum red time before re-checking
```

**For live mobile camera**, install the **IP Webcam** app on Android and set:

```python
VIDEO_SOURCE = "http://192.168.x.x:8080/video"
```

---

## 🐛 Troubleshooting

### LEDs not lighting up

| Symptom | Fix |
|---------|-----|
| No LEDs glow at all | Check USB cable — use data cable, not charge-only |
| Only one LED works | Verify all 3 resistors are connected |
| Python shows "connected" but no LED response | Close Arduino IDE completely, then run Python |
| LEDs flicker randomly | Try a different USB port or shorter cable |

### Serial port issues

```bash
# Find your Arduino port
python -c "import serial.tools.list_ports; [print(p.device, p.description) for p in serial.tools.list_ports.comports()]"
```

Look for `CH340`, `Arduino`, or `USB Serial` in the output. Set `FORCE_PORT = 'COMx'` if auto-detect fails.

### Vehicles not being detected

- Set `ROI_OVERRIDE = True` in the code to detect across the full frame
- Check your video has clear vehicle footage
- Try lowering `GREEN_THRESHOLD = 3`
- Check terminal output for `score=0` every frame (means ROI is wrong)

---

## 📊 Model Information

| Setting | Value | Reason |
|---------|-------|--------|
| Model | `yolo11s.pt` | Best accuracy/speed balance for traffic |
| Confidence | `0.40` | Filters weak detections |
| IOU threshold | `0.45` | Removes duplicate boxes on same vehicle |
| Input size | `1280×720` | Standard HD processing |

> **Why not yolo11n?** The nano model frequently misclassifies buses as trucks and vice versa. The small model (`yolo11s`) has 2× better accuracy with only ~30% speed reduction.

---

## 📋 Requirements

Create a `requirements.txt` with:

```
ultralytics>=8.0.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
pyserial>=3.5
numpy>=1.24.0
```

---

## 🔮 Future Improvements

- [ ] Train custom YOLO model on Indian traffic datasets (autos, e-rickshaws)
- [ ] Add ambulance / fire brigade custom class for real emergency detection
- [ ] Extend to full 4-lane intersection with priority queue
- [ ] Add night mode with adjusted confidence thresholds
- [ ] Web dashboard for live monitoring
- [ ] Deploy on Jetson Orin Nano for laptop-free operation

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the detection model
- [OpenCV](https://opencv.org/) for video processing
- [Arduino](https://www.arduino.cc/) for hardware control

---

*Built as a demonstration of AI-based adaptive traffic management for Indian road conditions.*
