# 📊 Results – Output Analysis

This folder contains sample outputs from the **Vision-Based Navigation System**.
Each image represents a processed frame where the system detects objects and generates a **navigation decision**.

---

## 🧠 How to Read the Output

Each frame includes:

* 🟢 **Green Bounding Boxes** → Detected objects (cars, persons, etc.)
* 🔴 **Red Bounding Boxes** → Critical / very close obstacles
* 📛 **Labels + Confidence Score** → Object type with detection confidence
* 🔵 **Blue Horizontal Line** → Region of Interest (ROI) boundary
* 🟡 **Vertical Yellow Lines** → Sector divisions used for navigation decisions
* 🟢 **Top Text (Decision Output)** → Final navigation command

---

## 🚦 Output Cases

### 1️⃣ Slow Right

**Observation:**

* Multiple obstacles detected in the **center and left sectors**
* Right side has comparatively **lower obstruction**

**Decision:**
👉 `SLOW_RIGHT`

**Interpretation:**

* System chooses to move right
* Speed is reduced due to nearby obstacles

---

### 2️⃣ Left 

**Observation:**

* Right and center areas contain obstacles (cars, pedestrians)
* Left region is relatively **clear**

**Decision:**
👉 `LEFT`

**Interpretation:**

* Safe to turn left
* Avoids detected objects in front/right

---

### 3️⃣ Slow Forward 

**Observation:**

* Obstacles are present ahead but **not too close**
* Forward path is partially clear

**Decision:**
👉 `SLOW_FORWARD`

**Interpretation:**

* Continue moving forward
* Reduce speed for safety

---

### 4️⃣ Stop

**Observation:**

* High density of obstacles directly in front
* Vehicles detected very close to ROI

**Decision:**
👉 `STOP`

**Interpretation:**

* Immediate halt required
* Unsafe to proceed

---

## ⚙️ Decision Strategy (Summary)

| Situation           | Output             |
| ------------------- | ------------------ |
| Clear path          | FORWARD            |
| Partial obstruction | SLOW_FORWARD       |
| Left clear          | LEFT /  SLOW_LEFT |              |
| Right clear         | RIGHT / SLOW_RIGHT |
| Path blocked        | STOP               |

---

## 📌 Key Insight

The system uses **spatial distribution of detected objects** across sectors to determine the safest navigation direction.
Decisions are based on:

* Object position
* Density of obstacles
* Proximity to the ROI

---

## 🚀 Note

These outputs demonstrate:

* Real-time object detection capability
* Effective obstacle-aware navigation logic
* Practical applicability in autonomous driving systems

---
