"""Detect a red target in one upright RGB camera frame."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RedBox:
    """Location of the largest red region that passes the area threshold."""

    position: str
    bounds: tuple[int, int, int, int]
    area: float


def find_red_box(frame: np.ndarray) -> RedBox | None:
    """Find the target and classify its horizontal location.

    ``frame`` must be an upright RGB array, such as a decoded image from MicroDuck's camera.
    Both red hue bands are included because red wraps around OpenCV's HSV hue range.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("expected an RGB image with three channels")

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    low_red = cv2.inRange(hsv, np.array((0, 90, 60)), np.array((10, 255, 255)))
    high_red = cv2.inRange(hsv, np.array((170, 90, 60)), np.array((180, 255, 255)))
    mask = cv2.bitwise_or(low_red, high_red)

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    height, width = frame.shape[:2]
    if area < max(80.0, width * height * 0.0004):
        return None

    x, y, box_width, box_height = cv2.boundingRect(contour)
    center_ratio = (x + box_width / 2) / width
    if center_ratio < 0.4:
        position = "left"
    elif center_ratio > 0.6:
        position = "right"
    else:
        position = "center"

    return RedBox(position, (x, y, box_width, box_height), area)


def annotate_red_box(frame: np.ndarray, detection: RedBox | None = None) -> np.ndarray:
    """Return a copy with the target box, if found, and left/center/right guides."""
    detection = detection if detection is not None else find_red_box(frame)
    height, width = frame.shape[:2]
    annotated = frame.copy()
    guide = (210, 210, 210)
    cv2.line(annotated, (int(width * 0.4), 0), (int(width * 0.4), height), guide, 1)
    cv2.line(annotated, (int(width * 0.6), 0), (int(width * 0.6), height), guide, 1)

    if detection is None:
        label = "RED BOX: not found"
    else:
        x, y, box_width, box_height = detection.bounds
        cv2.rectangle(annotated, (x, y), (x + box_width, y + box_height), (255, 255, 0), 2)
        label = f"RED BOX: {detection.position}"

    cv2.putText(
        annotated,
        label,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        label,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated
