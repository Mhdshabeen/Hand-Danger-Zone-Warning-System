# Hand Danger Zone Warning System (HDZWS)
This project is a real-time camera-based safety alert system that detects a human hand approaching a predefined danger zone and visually warns the user based on proximity. It demonstrates how computer vision can be used to monitor and prevent unsafe interactions with objects or restricted areas.

## 🧠Overview

* Real-time hand detection
* Tracking of an object zone
* Distance-based risk classification
* Visual alerts on a live camera stream

### It can be used for:

* Factory safety training
* Research in PPE compliance
* Human–robot interaction safety
* Virtual boundary enforcement

⚠️ Disclaimer:
This is a research prototype and not a certified safety system.

## ✨Features

🎥 Real-time webcam capture (OpenCV)

✋ Pretrained TensorFlow hand detector

📦 Virtual object zone (machine area)

📏 Distance calculation between hand and object

🚦 Three safety states:

  - 🟢 SAFE – green border
  - 🟡 WARNING – yellow border
  - 🔴 DANGER – red flashing border + text “DANGER DANGER”

⚡ Works on CPU (no GPU required)

🔁 Adjustable thresholds and detection sensitivity

## 🔄 Working Demonstration

1. **Start the program**  
   Run `app.py`. The webcam window will open.

2. **See the danger zone**  
   A white rectangle on the screen represents the **virtual object / danger zone** (for example, object or machine is placed).

3. **Show your hand to the camera**  
   Move your hand into the camera view. The model will detect it and draw a **yellow box** around your hand.

4. **Observe the SAFE state**  
   When your hand is **far from the danger zone**, the border of the screen turns **green** and the state text shows **SAFE**.

5. **Move your hand closer**  
   As your hand moves **towards the danger zone**, the border turns **yellow** and the state text changes to **WARNING**.

6. **Enter the danger zone**  
   When your hand is **very close to or touching** the danger zone, the border turns **red**, the state shows **DANGER**, and a large **“DANGER  DANGER”** message appears on the screen.

7. **Move your hand away**  
   Move your hand back. The system will switch back from **DANGER → WARNING → SAFE** as the distance increases.

8. **Exit the demo**  
   Press **`q`** to close the window and stop the program.


## Project Structure

```
hand-danger-zone-warnig-system/
│
├─ app.py          # main script
│
├─ model/
│   └─ frozen_inference_graph.pb  # pretrained hand detector
│
├─ utils/
│   ├─ label_map_util.py        # helper utilities (from TF Object Detection API)
│   └─ visualization_utils.py   # (optional) drawing helpers
│
├─ requirements.txt
└─ README.md
```
