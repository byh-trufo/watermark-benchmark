"""
SPDX-FileCopyrightText: © 2023 Trufo™ <tech@trufo.ai> All Rights Reserved
SPDX-License-Identifier: UNLICENSED

Image edits, alterations.
"""

import numpy as np
import cv2

from benchmark.image import utils
from benchmark.image.edit import ImageEdit


class IEAlterA(ImageEdit):
    """
    Simple alteration.
    """
    NUM = 9

    @staticmethod
    def overlay_box(image_bgr, ha, hb, wa, wb):
        mask = image_bgr[ha:hb, wa:wb, :].astype(np.float64) - utils.RANGE_MIDPOINT
        mask = mask / 4. + 64. * (-1) ** (np.mean(mask) > 0)
        image_bgr[ha:hb, wa:wb, :] = mask.astype(np.uint8)
        
        return image_bgr

    @staticmethod
    def overlay_lines(image_bgr, ha, hb, wa, wb):
        mask = image_bgr[ha:hb, :, :].astype(np.float64) - utils.RANGE_MIDPOINT
        mask = mask / 4. + 64. * (-1) ** (np.mean(mask) > 0)
        image_bgr[ha:hb, :, :] = mask.astype(np.uint8)
        mask = image_bgr[:, wa:wb, :].astype(np.float64) - utils.RANGE_MIDPOINT
        mask = mask / 4. + 64. * (-1) ** (np.mean(mask) > 0)
        image_bgr[:, wa:wb, :] = mask.astype(np.uint8)

        return image_bgr

    @staticmethod
    def remove_lines(image_bgr, ha, hb, wa, wb):
        image_bgr = np.delete(image_bgr, slice(ha, hb), 0)
        image_bgr = np.delete(image_bgr, slice(wa, wb), 1)

        return image_bgr

    @staticmethod
    def add_text(image_bgr, text, text_size, text_thickness):
        h, w = image_bgr.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        th, tw = cv2.getTextSize(text, font, text_size, text_thickness)[0]

        image_bgr = cv2.putText(
            image_bgr,
            text,
            ((h - th) // 2, (w - tw) // 2),
            font,
            text_size,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )

        return image_bgr
    
    def generate(self, image_bgr):
        h, w = image_bgr.shape[:2]

        # overlay, rectangles
        ha, hb = (8, 40) if self.rng.integers(2) else (h - 40, h - 8)
        wa, wb = (8, 40) if self.rng.integers(2) else (w - 40, w - 8)
        if 0 in self.indices:
            alt_image_bgr = self.overlay_box(image_bgr.copy(), ha, hb, wa, wb)
            yield ({'alteration' : "mask-corner-square"}, alt_image_bgr)

        hlim, wlim = (32, h // 2), (32, w // 2)
        hx, wx = self.rng.integers(*hlim), self.rng.integers(*wlim)
        ha, wa = self.rng.integers([h - hx, w - wx])
        if 1 in self.indices:
            alt_image_bgr = self.overlay_box(image_bgr.copy(), ha, ha + hx, wa, wa + wx)
            yield ({'alteration' : "mask-random-rectangle"}, alt_image_bgr)

        # overlay, pattern
        n = int(np.sqrt(h * w / 1024))
        has, was = self.rng.integers(h - 32, size=n), self.rng.integers(w - 32, size=n)
        if 2 in self.indices:
            alt_image_bgr = image_bgr.copy()
            for i in range(n):
                alt_image_bgr = self.overlay_box(alt_image_bgr, has[i], has[i] + 32, was[i], was[i] + 32)
            yield ({'alteration' : "mask-many-squares"}, alt_image_bgr)

        # overlay, lines
        ha, wa = self.rng.integers([h - 8, w - 8])
        if 3 in self.indices:
            alt_image_bgr = self.overlay_lines(image_bgr.copy(), ha, ha + 8, wa, wa + 8)
            yield ({'alteration' : "mask-small-lines"}, alt_image_bgr)

        hlim, wlim = (16, h // 4), (16, w // 4)
        hx, wx = self.rng.integers(*hlim), self.rng.integers(*wlim)
        ha, wa = self.rng.integers([h - hx, w - wx])
        if 4 in self.indices:
            alt_image_bgr = self.overlay_lines(image_bgr.copy(), ha, ha + hx, wa, wa + wx)
            yield ({'alteration' : "mask-random-lines"}, alt_image_bgr)

        # remove, lines
        ha, wa = self.rng.integers([h - 8, w - 8])
        if 5 in self.indices:
            alt_image_bgr = self.remove_lines(image_bgr.copy(), ha, ha + 8, wa, wa + 8)
            yield ({'alteration' : "remove-small-lines"}, alt_image_bgr)

        hlim, wlim = (8, h // 8), (8, w // 8)
        hx, wx = self.rng.integers(*hlim), self.rng.integers(*wlim)
        ha, wa = self.rng.integers([h - hx, w - wx])
        if 6 in self.indices:
            alt_image_bgr = self.remove_lines(image_bgr.copy(), ha, ha + hx, wa, wa + wx)
            yield ({'alteration' : "remove-random-lines"}, alt_image_bgr)

        # text
        text = "Trufo! (watermark benchmark)"

        if 7 in self.indices:
            alt_image_bgr = self.add_text(image_bgr.copy(), text, 0.5, 1)
            yield ({'alteration' : "text-small"}, alt_image_bgr)

        if 8 in self.indices:
            n = int(np.sqrt(h * w / 1024))
            alt_image_bgr = self.add_text(image_bgr.copy(), text, n // 16, n // 2)
            yield ({'alteration' : "text-large"}, alt_image_bgr)
