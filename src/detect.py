"""
YOLOv8 inference wrapper.

This module is responsible for one thing: take a raw video frame (a NumPy array
from OpenCV) and return a list of what was detected in it. Everything else —
capturing frames, displaying results, logging — is handled elsewhere.

Keeping detection isolated like this makes it easy to swap YOLOv8 out for a
TensorRT-optimized engine later without touching the rest of the pipeline.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detected object in a frame."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates
    track_id: int | None = None      # populated in Phase 2 by ByteTrack


class Detector:
    """Wraps a YOLOv8 model and exposes a simple detect() call."""

    def __init__(self, config: dict) -> None:
        cfg = config["detection"]
        model_path = Path(cfg["model_path"])

        # ultralytics will auto-download the model weights on first run if the
        # file doesn't exist yet — convenient for getting started quickly.
        self.model = YOLO(str(model_path))
        self.confidence_threshold = cfg["confidence_threshold"]
        self.iou_threshold = cfg["iou_threshold"]
        self.device = cfg["device"]
        self.imgsz = cfg["imgsz"]

        logger.info("Detector loaded: %s on %s", model_path.name, self.device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single BGR frame (OpenCV's default format).

        Returns a list of Detection objects — one per object found above the
        confidence threshold. Returns an empty list if nothing is detected.
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,  # suppress per-frame console spam
        )

        detections: list[Detection] = []

        # results[0] holds the output for our single input frame.
        # .boxes gives us the bounding box data in various formats.
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # absolute pixel coords
            detections.append(Detection(
                class_id=int(box.cls),
                class_name=self.model.names[int(box.cls)],
                confidence=float(box.conf),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
            ))

        return detections

    def annotate(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """
        Draw bounding boxes and labels onto a copy of the frame.

        We draw on a copy so the original frame stays unmodified — important
        later when we want to save clean snapshots to disk.
        """
        import cv2
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            label = f"{det.class_name} {det.confidence:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(
                annotated, label, (x1, y1 - 8),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.55,
                color=(0, 255, 0),
                thickness=1,
            )

        return annotated


# --- standalone test ---
# Run `python src/detect.py` to verify the model loads and runs on a test image.
if __name__ == "__main__":
    import sys
    import cv2

    logging.basicConfig(level=logging.INFO)

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    detector = Detector(config)

    # If an image path is passed as an argument, run detection on it.
    # Otherwise just confirm the model loaded correctly.
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            logger.error("Could not read image: %s", sys.argv[1])
            sys.exit(1)
        results = detector.detect(img)
        logger.info("Detected %d objects:", len(results))
        for d in results:
            logger.info("  %s (%.0f%%) at %s", d.class_name, d.confidence * 100, d.bbox)
        annotated = detector.annotate(img, results)
        cv2.imshow("Detection test", annotated)
        cv2.waitKey(0)
    else:
        logger.info("Detector initialized successfully. Pass an image path to test inference.")
