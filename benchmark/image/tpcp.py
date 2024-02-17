"""
SPDX-FileCopyrightText: © 2023 Trufo™ <tech@trufo.ai> All Rights Reserved
SPDX-License-Identifier: UNLICENSED

Trufo's perceptibility measure.
"""

from typing import Dict

import numpy as np
import cv2

from benchmark.image import utils


FIXED_SIZE = 512

EPS = 0.0001
CCC = 10.
_leps = np.log2(1. / EPS)
MAX_TPCP = np.abs(_leps) * _leps / (CCC + _leps * _leps)


def blur(grid: np.ndarray, size: int) -> np.ndarray:
    return cv2.GaussianBlur(grid, (2 * size - 1, 2 * size - 1), size)


def sigmoid(values: np.ndarray, limit: float=1.) -> np.ndarray:
    EXP_MAX = 10
    values = np.clip(values / limit, -EXP_MAX, EXP_MAX)
    
    s_values = 1 / (1 + np.exp(-values * 2))
    s_values = 2 * s_values - 1
    return limit * s_values


def calc_tpcp_channel(grid_a: np.ndarray, grid_b: np.ndarray, is_color: True) -> float:
    """
    Calculate TPCP data for a single color channel.
    """
    # with some inspiration from both PSNR and SSIM
    blend_size = 6 if is_color else 2
    base_range = 2. if is_color else 6.

    window_size = 16
    
    diff = grid_a - grid_b
    diff = blur(diff - 0.9 * blur(diff, window_size), blend_size)

    sig_a = grid_a - blur(grid_a, blend_size)
    sig_b = grid_b - blur(grid_b, blend_size)
    sig_a = blur(sig_a * sig_a, window_size)
    sig_b = blur(sig_b * sig_b, window_size)

    pcp_grid = diff * diff / (base_range * base_range + sig_a + sig_b)
    return pcp_grid


def calc_tpcp_single(ycc_a: np.ndarray, ycc_b: np.ndarray) -> float:
    """
    Calculate TPCP data for a single view.
    """
    # human vision resolution is ~3x lower in Cr/Cb vs. Y
    pcp_y = calc_tpcp_channel(ycc_a[:, :, 0], ycc_b[:, :, 0], is_color=False)
    pcp_cb = calc_tpcp_channel(ycc_a[:, :, 1], ycc_b[:, :, 1], is_color=True)
    pcp_cr = calc_tpcp_channel(ycc_a[:, :, 2], ycc_b[:, :, 2], is_color=True)

    # adding up the various color channels
    pcp = pcp_y + pcp_cb + pcp_cr

    # overall measure across the full grid
    tpcp = np.mean(np.power(pcp, 1.5))

    return tpcp


def calc_tpcp(image_a_bgr: np.ndarray, image_b_bgr: np.ndarray) -> float:
    """
    Calculate TPCP, Trufo's watermark perceptibility measure.
    """
    # original size
    image_a_ycc = utils.bgr_to_ycc(image_a_bgr)
    image_b_ycc = utils.bgr_to_ycc(image_b_bgr)

    tpcp_a = calc_tpcp_single(image_a_ycc, image_b_ycc)

    # fixed size
    image_a_bgr = utils.resize_frame(image_a_bgr, FIXED_SIZE, FIXED_SIZE)
    image_b_bgr = utils.resize_frame(image_b_bgr, FIXED_SIZE, FIXED_SIZE)

    image_a_ycc = utils.bgr_to_ycc(image_a_bgr)
    image_b_ycc = utils.bgr_to_ycc(image_b_bgr)

    tpcp_b = calc_tpcp_single(image_a_ycc, image_b_ycc)

    # blending
    tpcp = np.sqrt(tpcp_a) * 0.5 + np.sqrt(tpcp_b) * 0.5

    # scaling: scale from 0 to 100 (higher is better)
    tpcp = np.log2(1. / (EPS + tpcp))
    tpcp = np.clip(np.abs(tpcp) * tpcp / (CCC + tpcp * tpcp), 0., None)
    tpcp *= 100. / MAX_TPCP

    return tpcp
