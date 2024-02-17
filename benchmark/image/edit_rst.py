"""
SPDX-FileCopyrightText: © 2024 Trufo™ <tech@trufo.ai>
SPDX-License-Identifier: MIT

Image edits, RST.
"""

import numpy as np

from benchmark.image import utils
from benchmark.image.edit import ImageEdit


class IECrop(ImageEdit):
    """
    Cropping.
    """
    NUM = 8

    def gen_croppings(self, image_bgr):
        """
        Generate cropping parameters.
        """
        h, w = image_bgr.shape[:2]
        ch, cw = int(h // 10), int(w // 10)

        # all sides
        croppings = [((8, 8), (8, 8)), ((ch, ch), (cw, cw))]

        # one side
        cropping = [[0, 0], [0, 0]]
        ra, rb = self.rng.integers(2, size=2)
        cropping[ra][rb] = 8
        croppings.append(tuple(map(tuple, cropping)))

        cropping = [[0, 0], [0, 0]]
        ra, rb = self.rng.integers(2, size=2)
        cropping[ra][rb] = ch if ra == 0 else cw
        croppings.append(tuple(map(tuple, cropping)))

        # arbitrary
        for _ in range(4):
            cropping = [[0, 0], [0, 0]]
            for ra in range(2):
                for rb in range(2):
                    if self.rng.integers(2):
                        cropping[ra][rb] = self.rng.integers(5 * (ch if ra == 0 else cw))
            croppings.append(tuple(map(tuple, cropping)))
        
        return croppings

    def generate(self, image_bgr):
        h, w = image_bgr.shape[:2]
        croppings = self.gen_croppings(image_bgr)

        for ind in self.indices:
            cropping = croppings[ind]
            cropped_image_bgr = image_bgr[cropping[0][0]:h - cropping[0][1], cropping[1][0]:w - cropping[1][1], :]
            yield ({'cropping' : cropping}, cropped_image_bgr)


class IERescale(ImageEdit):
    """
    Rescaling.
    """
    NUM = 8
    SCALES = [(0.5, 0.5), (0.95, 0.95), (1.05, 1.05), (1.5, 1.5), (0.8, 1.2), (1.2, 0.8)]

    def generate(self, image_bgr):
        h, w = image_bgr.shape[:2]

        # relative rescaling
        for i in range(len(self.SCALES)):
            if i in self.indices:
                scale = self.SCALES[i]
                nh, nw = int(scale[0] * h), int(scale[1] * w)
                if nh * nw > max(h * w, 3840 * 2160):
                    continue
                if nh * nw < min(h * w, 256 * 256):
                    continue
                yield ({'scale' : scale}, utils.resize_frame(image_bgr, nh, nw))
    
        # fixed rescaling
        if 6 in self.indices:
            nh, nw = 512, 512
            yield ({'scale' : "fixed size"}, utils.resize_frame(image_bgr, nh, nw))
        if 7 in self.indices:
            mult = np.sqrt(1024 ** 2 / (h * w))
            nh, nw = int(h * mult), int(w * mult)
            yield ({'scale' : "fixed area"}, utils.resize_frame(image_bgr, nh, nw))


class IERotateA(ImageEdit):
    """
    D4 rotations + reflections.
    """
    NUM = 3

    def generate(self, image_bgr):
        ra = self.rng.integers(1, 4)
        rb = self.rng.integers(1, 4)
    
        if 0 in self.indices:
            rr_image_bgr = np.rot90(image_bgr, ra)
            yield ({'rotations' : ra, 'flip' : False}, rr_image_bgr)
            
        if 1 in self.indices:
            rr_image_bgr = np.fliplr(image_bgr)
            yield ({'rotations' : 0, 'flip' : True}, rr_image_bgr)
            
        if 2 in self.indices:
            rr_image_bgr = np.rot90(np.fliplr(image_bgr), rb)
            yield ({'rotations' : rb, 'flip' : True}, rr_image_bgr)


class IERotateB(ImageEdit):
    """
    Arbitrary rotations.
    """
    NUM = 5
    ANGLES = [1, 3, 9, 27]

    def generate(self, image_bgr):
        angles = [angle * (-1) ** self.rng.integers(2) for angle in self.ANGLES]
        angles.append(self.rng.integers(360))

        for ind in self.indices:
            yield ({'angle' : angles[ind]}, utils.rotate_frame(image_bgr, angles[ind]))
