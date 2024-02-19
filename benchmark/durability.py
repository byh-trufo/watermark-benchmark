"""
SPDX-FileCopyrightText: © 2024 Trufo™ <engineering@trufo.ai>
SPDX-License-Identifier: MIT

Assessment of durability.
"""

import enum
from typing import List

from benchmark.image.edit import ImageEdit, IEBase, IECompressJPEG
from benchmark.image.edit_rst import IECrop, IERescale, IERotateA, IERotateB
from benchmark.image.edit_fil import IEFilterA, IEFilterB
from benchmark.image.edit_alt import IEAlterA
from benchmark.image.edit_cmp import IEComposeA


class ImageRobustnessTests(enum.Enum):
    V1_BASIC = 'BASIC'
    V1_QUICK = 'QUICK'
    V1_FULL = 'FULL'


def image_edits(image_edits: ImageRobustnessTests, random_seed: int=0) -> List[ImageEdit]:
    """
    Get the list of image edits corresponding to an image evaluation mode.
    """
    if image_edits is ImageRobustnessTests.V1_BASIC:
        # 2 tests
        return [
            IEBase(random_seed)
        ]
    if image_edits is ImageRobustnessTests.V1_QUICK:
        # 12 tests
        return [
            IEBase(random_seed),
            IECompressJPEG(random_seed, [2, 5]),
            IECrop(random_seed, [5]),
            IERescale(random_seed, [7]),
            IERotateA(random_seed, [2]),
            IERotateB(random_seed, [2]),
            IEFilterA(random_seed, [0]),
            IEAlterA(random_seed, [4]),
            IEComposeA(random_seed, [2, 5]),
        ]
    if image_edits is ImageRobustnessTests.V1_FULL:
        # 60 tests
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
