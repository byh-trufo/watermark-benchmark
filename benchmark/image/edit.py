"""
SPDX-FileCopyrightText: © 2023 Trufo™ <tech@trufo.ai> All Rights Reserved
SPDX-License-Identifier: UNLICENSED

Image edits.
"""

import abc
import enum
from typing import List

import numpy as np


class ImageEditParams(enum.Enum):
    """
    Parameters for downstream processing.
    """
    JPEG_Q = 'jpeg_quality'


class ImageEdit(abc.ABC):
    @property
    @abc.abstractmethod
    def NUM(self):
        ...
    
    def __init__(self, random_seed: int=0, indices: List[int] = None):
        """
        Pass in a random seed for replicable pseudorandomness.
        Pass in a indices subset for granular test selection.
        """
        self.rng = np.random.default_rng(random_seed)

        if indices is None:
            indices = range(self.NUM)
        assert all([ind < self.NUM for ind in indices]), \
            f"Some of the indices that were passed were larger than the maximum {self.NUM}: {indices}."
        self.indices = indices

    @abc.abstractmethod
    def generate(self, image_bgr: np.ndarray):
        """
        Generator for image edits.
        """
        ...


class IEBase(ImageEdit):
    """
    Basic PNG + JPEG.
    """
    NUM = 2

    def generate(self, image_bgr):
        if 0 in self.indices:
            yield ({}, image_bgr)
        if 1 in self.indices:
            yield ({ImageEditParams.JPEG_Q : 95}, image_bgr)


class IECompressJPEG(ImageEdit):
    """
    Various levels of JPEG compression.
    """
    NUM = 11
    JPEG_QUALITY_LEVELS = [99, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10]

    def generate(self, image_bgr):
        for ind in self.indices:
            jpeg_quality = self.JPEG_QUALITY_LEVELS[ind]
            yield ({ImageEditParams.JPEG_Q.value : jpeg_quality}, image_bgr)
