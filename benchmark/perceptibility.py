"""
SPDX-FileCopyrightText: © 2023 Trufo™ <tech@trufo.ai> All Rights Reserved
SPDX-License-Identifier: UNLICENSED

Assessment of perceptibility.
"""

from typing import Dict, Union

import numpy as np
import cv2
from skimage.metrics import (
    peak_signal_noise_ratio as calc_psnr,
    structural_similarity as calc_ssim,
)

from benchmark.image import utils
from benchmark.image.tpcp import calc_tpcp


def assess_image(image_a: Union[bytes, np.ndarray], image_b: Union[bytes, np.ndarray]) -> Dict[str, float]:
    assessment = {
        'psnr' : 0., # peak signal-to-noise ratio
        'ssim' : 0., # structural similarity index
        'tpcp' : 0., # Trufo perceptibility measure
    }

    # try to read in w/ CV2; otherwise, no calculations
    if isinstance(image_a, bytes):
        image_a = utils.bytes_to_bgr(image_a)
    if isinstance(image_b, bytes):
        image_b = utils.bytes_to_bgr(image_b)

    # PSNR
    assessment['psnr'] = calc_psnr(image_a, image_b)
    # SSIM
    assessment['ssim'] = calc_ssim(image_a, image_b, channel_axis=2)
    # TPCP
    assessment['tpcp'] = calc_tpcp(image_a, image_b)

    return assessment
