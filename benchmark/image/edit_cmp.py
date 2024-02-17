"""
SPDX-FileCopyrightText: © 2024 Trufo™ <tech@trufo.ai>
SPDX-License-Identifier: MIT

Image edits, compositions.
"""

from benchmark.image.edit import ImageEdit, IECompressJPEG
from benchmark.image.edit_rst import IECrop, IERescale
from benchmark.image.edit_fil import IEFilterA
from benchmark.image.edit_alt import IEAlterA


class IEComposeA(ImageEdit):
    """
    Fixed composition.
    """
    NUM = 6

    def generate(self, image_bgr):
        # post: filters + alteration + compression
        random_seed = self.rng.integers(2**32)
        if 0 in self.indices:
            params = {'composite' : 'fixed/post'}
            
            m_gen = IEFilterA(random_seed, [2]).generate(image_bgr) # increase brightness
            m_params, mod_image_bgr = next(m_gen)
            params.update(m_params)
            m_gen = IEAlterA(random_seed, [0]).generate(mod_image_bgr) # corner square mask
            m_params, mod_image_bgr = next(m_gen)
            params.update(m_params)
            m_gen = IECompressJPEG(random_seed, [1]).generate(mod_image_bgr) # Q95 JPEG
            m_params, mod_image_bgr = next(m_gen)
            params.update(m_params)
            yield (params, mod_image_bgr)
        
        random_seed = self.rng.integers(2**32)
        if 1 in self.indices:
            params = {'composite' : 'fixed/post'}
            m_gen = IEFilterA(random_seed, [5]).generate(image_bgr) # decrease saturation
            m_params, mod_image_bgr = next(m_gen)
            params.update(m_params)
            m_gen = IEAlterA(random_seed, [2]).generate(mod_image_bgr) # many squares mask
            m_params, mod_image_bgr = next(m_gen)
            params.update(m_params)
            m_gen = IECompressJPEG(random_seed, [3]).generate(mod_image_bgr) # Q80 JPEG
            m_params, mod_image_bgr = next(m_gen)
            params.update(m_params)
            yield (params, mod_image_bgr)

        # repost: resizing + alteration + compression
        random_seed = self.rng.integers(2**32)
        if 2 in self.indices:
            params = {'composite' : 'fixed/repost'}
            m_gen = IERescale(random_seed, [1, 2]).generate(image_bgr) # 95%/105% size
            m_params, mod_image_bgr = next(m_gen)
            params.update(m_params)
            m_gen = IEAlterA(random_seed, [7]).generate(mod_image_bgr) # small text
            m_params, mod_image_bgr = next(m_gen)
            params.update(m_params)
            m_gen = IECompressJPEG(random_seed, [3]).generate(mod_image_bgr) # Q85 JPEG
            m_params, mod_image_bgr = next(m_gen)
            params.update(m_params)
            yield (params, mod_image_bgr)

        random_seed = self.rng.integers(2**32)
        if 3 in self.indices:
            params = {'composite' : 'fixed/repost'}
            m_gen = IERescale(random_seed, [6]).generate(image_bgr) # fixed size
            m_params, mod_image_bgr = next(m_gen)
            params.update(m_params)
            m_gen = IEAlterA(random_seed, [8]).generate(mod_image_bgr) # large text
            m_params, mod_image_bgr = next(m_gen)
            params.update(m_params)
            m_gen =  IECompressJPEG(random_seed, [6]).generate(mod_image_bgr) # Q70 JPEG
            m_params, mod_image_bgr = next(m_gen)
            params.update(m_params)
            yield (params, mod_image_bgr)

        # screenshot: resizing + cropping
        random_seed = self.rng.integers(2**32)
        if 4 in self.indices:
            params = {'composite' : 'fixed/screenshot'}
            m_gen = IERescale(random_seed, [1, 2]).generate(image_bgr) # 95%/105% size
            m_params, mod_image_bgr = next(m_gen)
            params.update(m_params)
            m_gen = IECrop(random_seed, [2]).generate(mod_image_bgr) # one side, 8px
            m_params, mod_image_bgr = next(m_gen)
            params.update(m_params)
            yield (params, mod_image_bgr)

        random_seed = self.rng.integers(2**32)
        if 5 in self.indices:
            params = {'composite' : 'fixed/screenshot'}
            m_gen = IERescale(random_seed, [7]).generate(image_bgr) # fixed area
            m_params, mod_image_bgr = next(m_gen)
            params.update(m_params)
            m_gen = IECrop(random_seed, [1]).generate(mod_image_bgr) # all sides, 10%
            m_params, mod_image_bgr = next(m_gen)
            params.update(m_params)
            yield (params, mod_image_bgr)


# [TODO] random composition
