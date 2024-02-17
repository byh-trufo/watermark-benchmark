"""
SPDX-FileCopyrightText: © 2023 Trufo™ <tech@trufo.ai> All Rights Reserved
SPDX-License-Identifier: UNLICENSED

Main file for running benchmark tests.
"""

import os
import glob
import enum
import logging

import pandas as pd
import cv2

from testing.benchmark.wrappers.wrapper import ImageWrapper
from benchmark import robustness
from benchmark import evaluate


class BenchmarkDataset(enum.Enum):
    """
    Configure the dataset used and the evaluations done.
    """
    NONE = ''
    IMG_0 = 'IMG_0'
    IMG_1 = 'IMG_1'
    IMG_2 = 'IMG_2'

DEFAULT_DATASET = BenchmarkDataset.IMG_0

DATASET_FILES = {
    BenchmarkDataset.IMG_0: "img_0/*",
    BenchmarkDataset.IMG_1: "img_1/*",
    BenchmarkDataset.IMG_2: "img_2/*",
}


class BenchmarkEvaluation(enum.Enum):
    """
    Preset benchmark evaluation modes.
    """
    NONE = ''
    # simple PNG + 95-Q JPEG (CV2 default)
    IMG_SIMPLE = 'IMG_SIMPLE'
    # for false positive testing
    IMG_NEGATIVE = 'IMG_NEGATIVE'
    # full suite of ~50 robustness tests per image
    IMG_ROBUSTNESS = 'IMG_ROBUSTNESS'

DEFAULT_EVALUATION = BenchmarkEvaluation.IMG_SIMPLE

EVALUATION_MODES = {
    BenchmarkEvaluation.IMG_SIMPLE: robustness.ImageEvaluation.V1_BASIC,
    BenchmarkEvaluation.IMG_NEGATIVE: robustness.ImageEvaluation.V1_BASIC,
    BenchmarkEvaluation.IMG_ROBUSTNESS: robustness.ImageEvaluation.V1_FULL,
}


def simple_logging_setup():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().handlers[0].setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s.%(levelname)s [%(filename)s:%(lineno)d] - %(message)s"
    ))


def benchmark(
    wrapper_class,
    dataset: BenchmarkDataset=DEFAULT_DATASET,
    evaluation: BenchmarkEvaluation=DEFAULT_EVALUATION,
    debug_mode: bool=False,
):
    """
    Run the benchmark evaluation on a single watermark. 
    """
    if wrapper_class.TYPE == ImageWrapper.TYPE:
        image_wrapper = wrapper_class()

        results = []

        image_filepaths = glob.glob(f"{os.getcwd()}/dataset/{DATASET_FILES[dataset]}")
        logging.info(f"Running {image_wrapper.name}.{dataset.value}.{evaluation.value} on {len(image_filepaths)} images.")

        if 'IMG' in evaluation.value:
            # [TODO] multiprocessing speedup
            for image_filepath in image_filepaths:
                results.extend(evaluate.evaluate_image(
                    image_filepath,
                    image_wrapper,
                    EVALUATION_MODES[evaluation],
                    encode=('NEG' not in evaluation.value),
                    debug_mode=debug_mode,
                ))

        if len(results) > 0:
            results = pd.DataFrame.from_dict(results)
            results['watermark'] = image_wrapper.name
            results['dataset'] = dataset.value
            results['evaluation'] = evaluation.value

            results.to_json(f"{os.getcwd()}/results/{image_wrapper.name}.{dataset.value}.{evaluation.value}.json")
    
    # [TODO] currently, only image functionality is supported
    else:
        raise NotImplementedError


def main():
    # [TODO] command line support: sys.argv -> (wrapper, mode)

    simple_logging_setup()
    from wrappers.ref_wrapper import DDWrapper
    benchmark(DDWrapper)

if __name__ == "__main__":
    main()