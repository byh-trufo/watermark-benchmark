"""
SPDX-FileCopyrightText: © 2024 Trufo™ <engineering@trufo.ai>
SPDX-License-Identifier: MIT

Shared image operations.
"""

import os

import numpy as np
import cv2
from IPython.display import Image, display


RANGE_MIDPOINT = 127.5


def bgr_to_ycc(image_bgr: np.ndarray) -> np.ndarray:
    """
    Converts BGR to YCbCr (float).
    """
    offset = np.asarray([0., 128., 128.])[None, None, :]
    return np.dot(image_bgr, np.asarray([
        [+0.114, +0.587, +0.299],
        [+0.500, -0.331264, -0.168736],
        [-0.081312, -0.418688, +0.500],
    ]).T) + offset


def bgr_to_bytes(image_bgr: np.ndarray, jpeg_quality: int=0) -> bytes:
    """
    Write CV2 image to a bytes object (basic).
    """
    if jpeg_quality == 0:
        return cv2.imencode('.png', image_bgr)[1].tobytes()
    return cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])[1].tobytes()


def bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """
    Read a bytes object to a CV2 image (basic).
    """
    return cv2.imdecode(np.asarray(bytearray(image_bytes)), cv2.IMREAD_COLOR)


def resize_frame(image_bgr: np.ndarray, nh: int, nw: int) -> np.ndarray:
    """
    Resizing a frame.
    """
    h, w = image_bgr.shape[:2]
    interpolation = cv2.INTER_CUBIC if nh * nw > h * w else cv2.INTER_AREA
    return cv2.resize(image_bgr, (nw, nh), interpolation=interpolation)


def rotate_frame(image_bgr: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotating a frame.
    """
    frame_center = tuple(np.array(image_bgr.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(frame_center, angle, 1.0)
    return cv2.warpAffine(image_bgr, rot_mat, image_bgr.shape[1::-1], flags=cv2.INTER_LINEAR)


def display_frame(image_bgr: np.ndarray) -> None:
    """
    Displays a frame using IPython.
    """
    location = 'tmp.jpg'
    cv2.imwrite(location, image_bgr)

    display(Image(filename=location))

    if os.path.exists(location):
        os.remove(location)
