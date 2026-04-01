# InfiDrive---Smart-Autonomous-Robotics-System
InfiDrive is an advanced autonomous robotics system designed to simulate real-world self-driving vehicle behavior using a combination of computer vision, reinforcement learning, and sensor-based navigation.
# 🚗 InfiDrive - AI-Powered Autonomous Robotics System

## 📌 Overview

**InfiDrive** is an advanced autonomous robotics system designed to simulate real-world self-driving vehicle behavior using a combination of **computer vision, reinforcement learning, and sensor-based navigation**.

The system integrates **deep learning models (YOLO, LLaVA), reinforcement learning trained in AirSim, and real-time motor control** to enable intelligent decision-making, obstacle avoidance, and adaptive navigation in dynamic environments.

---

## 🧠 System Architecture

InfiDrive is built on a modular architecture consisting of:

1. **Perception Layer**

   * Object detection using YOLO
   * Scene understanding using LLaVA (Vision-Language Model)
   * Sensor data processing (Ultrasonic / IR / LiDAR)

2. **Decision Layer**

   * Reinforcement Learning model trained in AirSim
   * Rule-based fallback logic
   * Context-aware navigation decisions

3. **Control Layer**

   * Motor driver control (PWM signals)
   * Direction and speed adjustment
   * Real-time execution of navigation commands

---

## 🔧 Hardware Components

* **Microcontroller / Edge Device**

  * Raspberry Pi (for real-time control and inference)

* **Motor Driver (L298N / similar)**

  * Controls wheel movement using PWM signals

* **Sensors**

  * Ultrasonic sensors → distance measurement
  * IR sensors → obstacle proximity
  * LiDAR  → precise mapping

* **Chassis & Actuators**

  * DC motors for movement
  * Wheels and robotic frame

* **Power System**

  * Battery pack with voltage regulation

---

## 👁️ Perception System

### 🔍 Object Detection (YOLO)

* YOLO (You Only Look Once) is used for **real-time object detection**
* Detects:

  * Obstacles
  * Humans
  * Vehicles
  * Path elements

**Pipeline:**

1. Camera captures frame
2. Frame passed to YOLO model
3. Bounding boxes + class labels generated
4. Output sent to decision layer

---

### 🧠 Scene Understanding (LLaVA)

* LLaVA (Large Language and Vision Assistant) provides **semantic understanding of the scene**
* Converts visual input into **context-aware descriptions**

**Example:**

* Instead of just detecting "object"
* LLaVA interprets:

  > "A person standing in front of the robot"

This improves:

* Decision accuracy
* Context-aware navigation

---

### 📡 Sensor Fusion

Multiple sensors are combined to improve reliability:

| Sensor     | Function                 |
| ---------- | ------------------------ |
| Ultrasonic | Distance measurement     |
| IR         | Close obstacle detection |
| LiDAR      | Mapping                  |

👉 Sensor fusion ensures:

* Redundancy
* Accuracy
* Low-latency response

---

## 🧠 Decision System

### 🎯 Reinforcement Learning (AirSim Training)

* The navigation model is trained using **Microsoft AirSim simulation**
* Uses **Reinforcement Learning (RL)** to learn optimal driving behavior

**Training Process:**

1. Agent interacts with environment
2. Receives:

   * State (camera + sensor input)
   * Reward (based on performance)
3. Learns policy:

   * Move forward
   * Turn left/right
   * Avoid obstacles

**Reward Function Includes:**

* +ve reward → safe navigation
* -ve reward → collision
* Penalty → inefficient path

---

### 🔁 Decision Pipeline

1. Input from:

   * YOLO (objects)
   * LLaVA (context)
   * Sensors (distance)

2. RL model predicts action:

   * Forward / Stop / Turn

3. If uncertain:

   * Rule-based fallback triggers

---

## ⚙️ Control System

### 🚘 Motor Control

* Motor driver (L298N) receives signals from controller
* Uses **PWM (Pulse Width Modulation)** to:

  * Control speed
  * Control direction

**Commands:**

* Forward
* Backward
* Left / Right
* Stop

---

### 🔄 Real-Time Execution Loop

```python
while True:
    frame = capture_camera()
    objects = yolo_detect(frame)
    context = llava_analyze(frame)
    sensor_data = read_sensors()

    action = rl_model.predict(frame, sensor_data)

    execute_motor_command(action)
```

---

## 🔗 Integration Flow

1. Camera captures environment
2. YOLO detects objects
3. LLaVA understands scene
4. Sensors provide distance data
5. RL model decides action
6. Motor controller executes movement

---

## 📸 Project Images



---

## ⚙️ Working Summary

The robot continuously senses its environment using cameras and sensors.
AI models process visual and contextual data, while the RL agent decides optimal actions.
The control system translates these decisions into motor movements, enabling **autonomous navigation in real-time**.

---

## 🚀 Future Improvements

* ROS2-based distributed architecture
* SLAM (Simultaneous Localization and Mapping)
* Edge AI optimization (Jetson Nano / Coral TPU)
* Multi-agent coordination
* Cloud-based monitoring dashboard

---

## 💡 Key Highlights

* Combines **Computer Vision + RL + Robotics**
* Simulation-to-real-world pipeline (AirSim → Hardware)
* Multi-modal AI (Vision + Language + Sensors)
* Real-time autonomous decision system

---

## 👨‍💻 Author

**Kaushal Dholakiya**
