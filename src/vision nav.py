import cv2
import numpy as np
import os
from collections import deque

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

YOLO_DIR = os.path.join(PROJECT_DIR, "yolo")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

print("CFG:", os.path.exists(os.path.join(YOLO_DIR, "yolov3-tiny.cfg")))
print("WEIGHTS:", os.path.exists(os.path.join(YOLO_DIR, "yolov3-tiny.weights")))
print("NAMES:", os.path.exists(os.path.join(YOLO_DIR, "coco.names")))

# ================= VIDEO ======================
cap = cv2.VideoCapture(os.path.join(DATA_DIR, "dashcam.mp4"))
if not cap.isOpened():
    print("Error: Video not accessible")
    exit()

# ================= YOLO =======================
net = cv2.dnn.readNet(
    os.path.join(YOLO_DIR, "yolov3-tiny.weights"),
    os.path.join(YOLO_DIR, "yolov3-tiny.cfg")
)

with open(os.path.join(YOLO_DIR, "coco.names"), "r") as f:
    classes = f.read().strip().split("\n")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ================= BUFFERS ====================
decision_buffer = deque(maxlen=7)
risk_buffer = deque(maxlen=5)

# ================= FUNCTIONS ==================
def weighted_occupancy(region):
    if region.size == 0:
        return 0.0
    h = region.shape[0]
    weights = np.linspace(0.5, 1.5, h).reshape(h, 1)
    occupied = (region > 0).astype(np.float32)
    return np.sum(occupied * weights) / region.size

# ================= MAIN LOOP ==================
while True:
    ret, img = cap.read()
    if not ret:
        break

    h_img, w_img, _ = img.shape

    # -------- PREPROCESS ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 160)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # -------- ROI ----------
    roi_y_start = int(h_img * 0.6)
    roi = edges[roi_y_start:h_img, :]

    # -------- NAVIGATION (5 SECTORS) ----------
    sectors = np.array_split(roi, 5, axis=1)
    scores = [weighted_occupancy(s) for s in sectors]

    risk_buffer.append(scores)
    smoothed_scores = np.mean(risk_buffer, axis=0)

    mean_risk = np.mean(smoothed_scores)
    stop_threshold = mean_risk * 1.4

    min_idx = np.argmin(smoothed_scores)

    if all(s > stop_threshold for s in smoothed_scores):
        nav_decision = "STOP"
    elif min_idx == 2:
        nav_decision = "FORWARD"
    elif min_idx < 2:
        nav_decision = "LEFT"
    else:
        nav_decision = "RIGHT"

    # ================= YOLO DETECTION =================
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    override_decision = None
    sector_width = w_img // 5

    for output in outputs:
        for det in output:
            scores_det = det[5:]
            class_id = np.argmax(scores_det)
            confidence = scores_det[class_id]

            if confidence < 0.5:
                continue

            label = classes[class_id]
            if label not in ["person", "car", "bus", "truck"]:
                continue

            cx, cy, bw, bh = det[:4]
            x = int((cx - bw / 2) * w_img)
            y = int((cy - bh / 2) * h_img)
            w = int(bw * w_img)
            h = int(bh * h_img)

            bottom_y = y + h

            # -------- PROXIMITY ESTIMATION ----------
            proximity = "FAR"
            if bottom_y > roi_y_start and h > 0.35 * h_img:
                proximity = "VERY_NEAR"
            elif bottom_y > roi_y_start and h > 0.20 * h_img:
                proximity = "NEAR"

            # -------- SECTOR OVERLAP ----------
            box_left = x
            box_right = x + w
            overlaps = []

            for i in range(5):
                s_start = i * sector_width
                s_end = (i + 1) * sector_width
                overlap = max(0, min(box_right, s_end) - max(box_left, s_start))
                overlaps.append(overlap)

            blocked_sector = np.argmax(overlaps)

            # -------- DECISION LOGIC ----------
            if proximity == "VERY_NEAR":
                override_decision = "STOP"
            elif proximity == "NEAR":
                if blocked_sector == 2:
                    override_decision = "SLOW_FORWARD"
                elif blocked_sector < 2:
                    override_decision = "SLOW_RIGHT"
                else:
                    override_decision = "SLOW_LEFT"

            # -------- DRAW BOX ----------
            color = (0, 0, 255) if proximity != "FAR" else (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ================= FINAL DECISION =================
    final_decision = override_decision if override_decision else nav_decision
    decision_buffer.append(final_decision)
    final_decision = max(set(decision_buffer), key=decision_buffer.count)

    # ================= VISUALIZATION =================
    cv2.rectangle(img, (0, roi_y_start), (w_img, h_img), (255, 0, 0), 2)

    for i in range(1, 5):
        cv2.line(img, (i * sector_width, roi_y_start),
                 (i * sector_width, h_img), (0, 255, 255), 2)

    color = (0, 0, 255) if "STOP" in final_decision else (0, 255, 0)
    cv2.putText(img, final_decision, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Vision Navigation", img)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# ================= CLEANUP ====================
cap.release()
cv2.destroyAllWindows()
