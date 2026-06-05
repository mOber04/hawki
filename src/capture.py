"""
Main webcam capture loop — the entry point for the entire pipeline.

This is where everything comes together: open the camera, grab a frame,
run detection, draw results, display them, repeat. In later phases this
loop will also call the tracker, rule engine, and event logger.
"""

import logging
import time

import cv2
import yaml

from detect import Detector

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def open_camera(config: dict) -> cv2.VideoCapture:
    """
    Open the webcam and set resolution and FPS.

    OpenCV uses integer device IDs (0, 1, 2...) to identify cameras.
    device_id 0 is almost always the first connected camera.
    """
    cfg = config["camera"]
    cap = cv2.VideoCapture(cfg["device_id"])

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera (device_id={cfg['device_id']}). "
            "Check that the camera is connected and not in use by another app."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["fps"])

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info("Camera opened: %dx%d @ %.0f FPS", actual_w, actual_h, actual_fps)

    return cap


def draw_fps(frame, fps: float):
    """Overlay the measured FPS in the top-left corner."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 200, 255),
        thickness=2,
    )


def run(config: dict) -> None:
    """
    Main loop: capture → detect → annotate → display.

    Press 'q' in the display window to quit cleanly.
    """
    detector = Detector(config)
    cap = open_camera(config)

    # FPS tracking: we measure real throughput, not just what the camera reports.
    # This tells us whether our detection is keeping up with the video stream.
    fps_window = 30          # average FPS over this many frames
    frame_times: list[float] = []

    logger.info("Pipeline running — press 'q' to quit")

    try:
        while True:
            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame — camera disconnected?")
                break

            detections = detector.detect(frame)

            # Annotate a copy of the frame so the raw frame is preserved.
            annotated = detector.annotate(frame, detections)

            # FPS calculation: keep a rolling window of the last N frame durations.
            elapsed = time.perf_counter() - t_start
            frame_times.append(elapsed)
            if len(frame_times) > fps_window:
                frame_times.pop(0)
            avg_fps = len(frame_times) / sum(frame_times)

            draw_fps(annotated, avg_fps)

            # Object count overlay
            cv2.putText(
                annotated, f"Objects: {len(detections)}", (10, 65),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 200, 255),
                thickness=2,
            )

            cv2.imshow("hawki — Phase 1", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit requested")
                break

    finally:
        # Always release the camera, even if we crash — otherwise it stays locked.
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released. Goodbye.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config()
    run(config)
