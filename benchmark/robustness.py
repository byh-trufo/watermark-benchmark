"""
SPDX-FileCopyrightText: © 2024 Trufo™ <tech@trufo.ai>
SPDX-License-Identifier: MIT

Assessment of robustness.
"""

import enum
from typing import List

from benchmark.image.edit import ImageEdit, IEBase, IECompressJPEG
from benchmark.image.edit_rst import IECrop, IERescale, IERotateA, IERotateB
from benchmark.image.edit_fil import IEFilterA, IEFilterB
from benchmark.image.edit_alt import IEAlterA
from benchmark.image.edit_cmp import IEComposeA


class ImageEvaluation(enum.Enum):
    V1_BASIC = 'BASIC'
    V1_FULL = 'FULL'


def image_edits(image_edits: ImageEvaluation, random_seed: int=0) -> List[ImageEdit]:
    """
    Get the list of image edits corresponding to an image evaluation mode.
    """
    if image_edits is ImageEvaluation.V1_BASIC:
        return [
            IEBase(random_seed)
        ]
    if image_edits is ImageEvaluation.V1_FULL:
        return [
            IEBase(random_seed),
            IECompressJPEG(random_seed),
            IECrop(random_seed),
            IERescale(random_seed),
            IERotateA(random_seed),
            IERotateB(random_seed),
            IEFilterA(random_seed),
            IEFilterB(random_seed),
            IEAlterA(random_seed),
            IEComposeA(random_seed),
        ]
    raise NotImplementedError
