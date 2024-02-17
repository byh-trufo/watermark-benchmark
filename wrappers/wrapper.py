"""
SPDX-FileCopyrightText: © 2024 Trufo™ <tech@trufo.ai>
SPDX-License-Identifier: MIT

Image wrapper.
"""

import abc
from typing import Union

import numpy as np


class ImageWrapper(abc.ABC):
    """
    Abstract interface class for a watermark wrapper object.
    """
    TYPE = 'IMAGE'

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def payload_size(self) -> int:
        ...

    @abc.abstractmethod
    def encode(self, image_bytes: bytes, payload_bits: np.ndarray) -> bytes:
        ...

    @abc.abstractmethod
    def decode(self, image_bytes: bytes) -> Union[None, np.ndarray]:
        ...
