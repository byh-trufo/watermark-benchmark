"""
SPDX-FileCopyrightText: © 2023 Trufo™ <tech@trufo.ai> All Rights Reserved
SPDX-License-Identifier: UNLICENSED

Evaluation module.
"""

import time
import logging
import mimetypes
from typing import List

import numpy as np
import cv2

from benchmark import perceptibility, robustness
from benchmark.image import utils
from benchmark.image.edit import ImageEdit, ImageEditParams

from wrappers.wrapper import ImageWrapper


IMAGE_RESULTS = {
    'watermark' : "",
    'dataset' : "",
    'evaluation' : "",
    'operation' : "",
    'content_id' : "",
    'content_format' : "",
    'content_dimensions' : (),
    'time_taken_ms' : 0.,
    'error' : False,
    'psnr' : 0.,
    'ssim' : 0.,
    'tpcp' : 0.,
    'edit_type' : "",
    'edit_parameters' : {},
    'detected' : False,
    'decoded' : False,
}


def evaluate_image(
    filepath: str,
    wrapper: ImageWrapper,
    evaluation: robustness.ImageEvaluation,
    encode: bool=True,
    debug_mode: bool=False,
):
    """
    Run a specified set of tests on an image.
    """
    results = []

    with open(filepath, 'rb') as image_file:
        image_name = filepath.split('/')[-1].split('.')[0]
        logging.info(f"Processing image {image_name}.")

        random_seed = hash(image_name) % 2 ** 32
        image_bytes = image_file.read()
        image_bgr = utils.bytes_to_bgr(image_bytes)

        rng = np.random.default_rng(random_seed)
        payload_bits = rng.integers(2, size=wrapper.payload_size).astype(bool)

        ### encoding
            
        if encode:

            # preprocessing
            if debug_mode:
                logging.info("Dispalying the original image.")
                utils.display_frame(image_bgr)

            enc_result = IMAGE_RESULTS.copy()
            enc_result['operation'] = "encode"
            enc_result['content_id'] = image_name
            enc_result['content_format'] = mimetypes.guess_type(filepath)[0]
            if enc_result['content_format'] is None:
                enc_result['content_format'] = "image/unknown"
            enc_result['content_dimensions'] = image_bgr.shape

            # encoding
            t = time.time()
            try:
                enc_image_bytes = wrapper.encode(image_bytes, payload_bits)
            except:
                enc_result['error'] = True
                logging.error("Encoding error:", exc_info=True)
            enc_result['time_taken_ms'] = int((time.time() - t) * 1000)
            if enc_result['error']:
                results.append(enc_result)
                return results

            # postprocessing
            enc_image_bgr = utils.bytes_to_bgr(enc_image_bytes)
            if debug_mode:
                logging.info("Dispalying the watermarked image.")
                utils.display_frame(enc_image_bgr)

            enc_result.update(perceptibility.assess_image(image_bgr, enc_image_bgr))
                
            results.append(enc_result)

        ### decoding

        if not encode:
            enc_image_bytes = image_bytes
            enc_image_bgr = image_bgr
        
        edits = robustness.image_edits(evaluation)
            
        for edit in edits:
            edit_generator = edit.generate(enc_image_bgr)
            for (edit_parameters, mod_image_bgr) in edit_generator:

                # preprocessing
                if debug_mode:
                    print(f"{type(edit).__name__}:{edit_parameters}.")
                    utils.display_frame(mod_image_bgr)

                dec_result = IMAGE_RESULTS.copy()
                dec_result['operation'] = "decode"
                dec_result['content_id'] = image_name
                dec_result['content_format'] = "image"
                dec_result['content_dimensions'] = mod_image_bgr.shape
                dec_result['edit_type'] = type(edit).__name__
                dec_result['edit_parameters'] = edit_parameters

                if ImageEditParams.JPEG_Q.value in edit_parameters:
                    mod_image_bytes = utils.bgr_to_bytes(mod_image_bgr, jpeg_quality=edit_parameters[ImageEditParams.JPEG_Q.value])
                else:
                    mod_image_bytes = utils.bgr_to_bytes(mod_image_bgr)

                # decoding
                t = time.time()
                try:
                    dec_payload_bits = wrapper.decode(mod_image_bytes)
                except:
                    dec_result['error'] = True
                    logging.error(f"Decoding error ({edit_parameters}):", exc_info=True)
                dec_result['time_taken_ms'] = int((time.time() - t) * 1000)
                if dec_result['error']:
                    results.append(dec_result)
                    continue

                # postprocessing
                if dec_payload_bits is not None:
                    dec_result['detected'] = True
                    dec_result['decoded'] = np.array_equal(dec_payload_bits, payload_bits)
            
                results.append(dec_result)

    return results
