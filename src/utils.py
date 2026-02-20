import cv2
import random


_CLASS_COLORS = {}

def _get_color(label):
    if label not in _CLASS_COLORS:
        _CLASS_COLORS[label] = (
            random.randint(80, 255),
            random.randint(80, 255),
            random.randint(80, 255),
        )
    return _CLASS_COLORS[label]


def draw_custom_boxes(frame, detections, allowed_classes=None):
   

    for x1, y1, x2, y2, label, conf in detections:

        
        if allowed_classes is not None:
            if label.lower() not in {c.lower() for c in allowed_classes}:
                continue

        color = _get_color(label)

        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"

        
        (tw, th), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

       
        cv2.rectangle(
            frame,
            (x1 - 1, y1 - th - 10),
            (x1 + tw + 8, y1 + 2),
            (0, 0, 0),
            -1
        )

        
        cv2.rectangle(
            frame,
            (x1, y1 - th - 8),
            (x1 + tw + 6, y1),
            color,
            -1
        )

        
        cv2.putText(
            frame,
            text,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return frame
