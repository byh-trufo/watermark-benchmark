"""
SPDX-FileCopyrightText: © 2023 Trufo™ <tech@trufo.ai> All Rights Reserved
SPDX-License-Identifier: UNLICENSED

Image edits, filters.
"""

import numpy as np
import cv2

from benchmark.image import utils
from benchmark.image.edit import ImageEdit


class IEFilterA(ImageEdit):
    """
    Global filters.
    """
    NUM = 6

    @staticmethod
    def gamma_correction(data, gamma):
        table = np.empty((1, 256), np.uint8)
        for i in range(256):
            table[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        return cv2.LUT(data, table)

    def generate(self, image_bgr):
        # grayscale
        if 0 in self.indices:
            fil_image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            yield ({'filter' : "grayscale"}, fil_image_bgr)

        # negative
        if 1 in self.indices:
            fil_image_bgr = np.invert(image_bgr)
            yield ({'filter' : "negative"}, fil_image_bgr)

        # brightness
        if 2 in self.indices or 3 in self.indices:
            is_bright = np.mean(image_bgr.astype(np.float64)) > utils.RANGE_MIDPOINT
            factors = [0.8, 1.5] if is_bright else [0.5, 1.2]

            for i in [2, 3]:
                if i in self.indices:
                    factor = factors[i - 2]
                    fil_image_bgr = self.gamma_correction(image_bgr, factor)
                    yield ({'filter' : "brightness-gamma", 'factor' : factor}, fil_image_bgr)

        # saturation
        if 4 in self.indices or 5 in self.indices:
            image_hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)
            factors = [0.5, 1.5]

            for i in [4, 5]:
                if i in self.indices:
                    factor = factors[i - 4]
                    fil_image_hls = image_hls.copy()
                    fil_image_hls[:, :, 2] = self.gamma_correction(image_hls[:, :, 2], factor)
                    fil_image_bgr = cv2.cvtColor(fil_image_hls, cv2.COLOR_HLS2BGR)
                    yield ({'filter' : "saturation-gamma", 'factor' : factor}, fil_image_bgr)


class IEFilterB(ImageEdit):
    """
    Local filters.
    """
    NUM = 3

    def generate(self, image_bgr):
        # blur / sharpen
        if 0 in self.indices or 1 in self.indices:
            fil_image_bgr = cv2.GaussianBlur(image_bgr, (7, 7), 4)

            if 0 in self.indices:
                yield ({'filter' : "blur"}, fil_image_bgr)
            if 1 in self.indices:
                fil_image_bgr = cv2.addWeighted(image_bgr, 1.5, fil_image_bgr, -0.5, 0.)
                yield ({'filter' : "sharpen"}, fil_image_bgr)

        # posterize
        if 2 in self.indices:
            fil_image_bgr = (np.round(image_bgr.astype(np.float32) / 4) * 4).astype(np.uint8)
            yield ({'filter' : "posterize"}, fil_image_bgr)
