import cv2
import numpy as np
import math
import tensorflow as tf
import os

# Use TF1-style graph/session inside TF2
tf.compat.v1.disable_eager_execution()

# ----------------- CONFIG -----------------

# Base folder and model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "frozen_inference_graph.pb")

# Manually define object / danger zone in the frame
# (x1, y1, x2, y2) in pixels after resizing to 640x480
ZONE_BOX = (560, 120, 630, 320)

# Distance thresholds (in pixels)
SAFE_THRESH = 120
DANGER_THRESH = 60


# ----------------- UTILS -----------------

def rect_distance(r1, r2):

    x1_min, y1_min, x1_max, y1_max = r1
    x2_min, y2_min, x2_max, y2_max = r2
    dx = max(x1_min - x2_max, x2_min - x1_max, 0)
    dy = max(y1_min - y2_max, y2_min - y1_max, 0)
    return math.sqrt(dx * dx + dy * dy)


def load_ssd_graph(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")

    sess = tf.compat.v1.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
    boxes_tensor = detection_graph.get_tensor_by_name("detection_boxes:0")
    scores_tensor = detection_graph.get_tensor_by_name("detection_scores:0")
    classes_tensor = detection_graph.get_tensor_by_name("detection_classes:0")
    num_detections_tensor = detection_graph.get_tensor_by_name("num_detections:0")

    return (detection_graph, sess,
            image_tensor, boxes_tensor, scores_tensor, classes_tensor, num_detections_tensor)


def box_center(box):

    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy


def suppress_by_center(boxes, scores, center_thresh=60):

    if not boxes:
        return []

    idxs = np.argsort(scores)[::-1]
    keep = []
    keep_centers = []

    for idx in idxs:
        bx = boxes[idx]
        cx, cy = box_center(bx)

        too_close = False
        for (kc_x, kc_y) in keep_centers:
            dist = math.sqrt((cx - kc_x) ** 2 + (cy - kc_y) ** 2)
            if dist < center_thresh:
                too_close = True
                break

        if not too_close:
            keep.append(idx)
            keep_centers.append((cx, cy))

    return keep


class HandDetector:
    def __init__(self, model_path):
        (self.graph,
         self.sess,
         self.image_tensor,
         self.boxes_tensor,
         self.scores_tensor,
         self.classes_tensor,
         self.num_detections_tensor) = load_ssd_graph(model_path)

    def detect_hands(self, frame, score_thresh=0.5, center_thresh=60):
       
        h, w = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)

        (boxes, scores, classes, num) = self.sess.run(
            [self.boxes_tensor, self.scores_tensor,
             self.classes_tensor, self.num_detections_tensor],
            feed_dict={self.image_tensor: image_expanded}
        )

        boxes = boxes[0]   
        scores = scores[0]

        raw_boxes = []
        raw_scores = []
        for i in range(len(scores)):
            if scores[i] < score_thresh:
                continue

            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(xmin * w)
            y1 = int(ymin * h)
            x2 = int(xmax * w)
            y2 = int(ymax * h)
            raw_boxes.append((x1, y1, x2, y2))
            raw_scores.append(float(scores[i]))

        keep_indices = suppress_by_center(raw_boxes, raw_scores,
                                          center_thresh=center_thresh)

        results = []
        for idx in keep_indices:
            x1, y1, x2, y2 = raw_boxes[idx]
            score = raw_scores[idx]
            results.append((x1, y1, x2, y2, score))

        return results

def main():
    detector = HandDetector(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        bx1, by1, bx2, by2 = ZONE_BOX
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
        cv2.putText(frame, "Zone", (bx1, max(0, by1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        hands = detector.detect_hands(frame, score_thresh=0.5, center_thresh=60)

        # Default state
        state = "NO HAND"
        color = (200, 200, 200)
        min_dist = None  

        if hands:
            for (hx1, hy1, hx2, hy2, score) in hands:
               
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)
                cv2.putText(frame, f"Hand {score:.2f}", (hx1, max(0, hy1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                
                d = rect_distance(ZONE_BOX, (hx1, hy1, hx2, hy2))

                if min_dist is None or d < min_dist:
                    min_dist = d

            # Global state based on closest hand (worst case)
            if min_dist is not None:
                if min_dist > SAFE_THRESH:
                    state = "SAFE"
                    color = (0, 255, 0)
                elif min_dist > DANGER_THRESH:
                    state = "WARNING"
                    color = (0, 255, 255)
                else:
                    state = "DANGER"
                    color = (0, 0, 255)

        cv2.rectangle(frame, (10, 10), (270, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"STATE: {state}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        if min_dist is not None:
            cv2.putText(frame, f"Min Dist: {int(min_dist)} px", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if state in ["SAFE", "WARNING", "DANGER"]:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 5)

        if state == "DANGER":
            cv2.putText(frame, "DANGER  DANGER", (80, 250),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4,
                        (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("Hand Detector - Virtual Danger Zone", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
