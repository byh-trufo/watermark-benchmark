"""
SPDX-FileCopyrightText: © 2024 Trufo™ <engineering@trufo.ai>
SPDX-License-Identifier: MIT

Wrapper class for the open-source invisible-watermark library.
"""

import abc
import logging

import numpy as np
import cv2

from wrappers.wrapper import ImageWrapper

from imwatermark import WatermarkEncoder, WatermarkDecoder


class IMWrapper(ImageWrapper):
    payload_size = 32

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...
    
    def encode(self, image_bytes: bytes, payload_bits: np.ndarray) -> bytes:
        assert len(payload_bits) == self.payload_size

        image_bgr = cv2.imdecode(np.asarray(bytearray(image_bytes)), cv2.IMREAD_COLOR)
        payload_bytes = np.packbits(payload_bits).tobytes()

        encoder = WatermarkEncoder()
        encoder.set_watermark('bytes', payload_bytes)
        if self.mode == 'rivaGan':
            if np.prod(image_bgr.shape) > 10 ** 7:
                raise Exception((
                    "Image size is rather large for the RivaGan watermark."
                    "If you are confident in your CPU/GPU, feel free to disable this size check."
                ))
            encoder.loadModel()

        enc_image_bgr = encoder.encode(image_bgr, self.mode)
        enc_image_bytes = cv2.imencode('.png', enc_image_bgr)[1].tobytes()

        return enc_image_bytes
    
    def decode(self, image_bytes: bytes) -> np.ndarray:
        image_bgr = cv2.imdecode(np.asarray(bytearray(image_bytes)), cv2.IMREAD_COLOR)

        decoder = WatermarkDecoder('bits', self.payload_size)
        if self.mode == 'rivaGan':
            if np.prod(image_bgr.shape) > 10 ** 7:
                raise Exception((
                    "Image size is rather large for the RivaGan watermark."
                    "If you are confident in your CPU/GPU, feel free to disable this size check."
                ))
            decoder.loadModel()
            
        try:
            payload_bits = decoder.decode(image_bgr, self.mode)
        except:
            return None

        return payload_bits


class DDWrapper(IMWrapper):
    name = "REF_DWTDCT"
    mode = 'dwtDct'

class DDSWrapper(IMWrapper):
    name = "REF_DWTDCTSVD"
    mode = 'dwtDctSvd'

class RGWrapper(IMWrapper):
    name = "REF_RIVAGAN"
    mode = 'rivaGan'

    def encode(self, *args, **kwargs):
        try:
            import imp
            imp.find_module('onnxruntime')
        except ImportError:
            logging.error("To run the RivaGan watermark from invisible-watermark, the module 'onnxruntime' is required.")
            raise Exception()
        return super().encode(*args, **kwargs)
