"""
SPDX-FileCopyrightText: © 2024 Trufo™ <engineering@trufo.ai>
SPDX-License-Identifier: MIT

Main file for running benchmark tests.
"""

import os
import glob
import enum
import logging

import pandas as pd
import cv2

from wrappers.wrapper import ImageWrapper
from benchmark import durability
from benchmark import evaluate


class BenchmarkDataset(enum.Enum):
    """
    Configure the dataset used and the evaluations done.
    """
    NONE = ''
    IMG_0 = 'IMG_0'
    IMG_1 = 'IMG_1'
    IMG_VOC = 'IMG_VOC'
    IMG_BIG = 'IMG_BIG'

DEFAULT_DATASET = BenchmarkDataset.IMG_0

DATASET_FILES = {
    BenchmarkDataset.IMG_0: "img_0/*",
    BenchmarkDataset.IMG_1: "img_1/*",
    BenchmarkDataset.IMG_VOC: "img_voc/*",
    BenchmarkDataset.IMG_BIG: "img_big/*",
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
    # full suite of 60 robustness tests per image
    IMG_ROBUSTNESS = 'IMG_ROBUSTNESS'
    # abridged suite of 12 robustness tests per image
    IMG_ROBUSTNESS_Q = 'IMG_ROBUSTNESS_Q'

DEFAULT_EVALUATION = BenchmarkEvaluation.IMG_SIMPLE

EVALUATION_MODES = {
    BenchmarkEvaluation.IMG_SIMPLE: durability.ImageRobustnessTests.V1_BASIC,
    BenchmarkEvaluation.IMG_NEGATIVE: durability.ImageRobustnessTests.V1_BASIC,
    BenchmarkEvaluation.IMG_ROBUSTNESS: durability.ImageRobustnessTests.V1_FULL,
    BenchmarkEvaluation.IMG_ROBUSTNESS_Q: durability.ImageRobustnessTests.V1_QUICK,
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
    override: bool=False,
    debug_mode: bool=False,
):
    """
    Run the benchmark evaluation on a single watermark. 
    """
    if wrapper_class.TYPE == ImageWrapper.TYPE:
        assert 'IMG' in dataset.value
        assert 'IMG' in evaluation.value
        image_wrapper = wrapper_class()

        out_filepath = f"{os.getcwd()}/results/{image_wrapper.name}.{dataset.value}.{evaluation.value}.json"
        if not override and os.path.exists(out_filepath):
            logging.info((
                f"The results file {image_wrapper.name}.{dataset.value}.{evaluation.value} already exists. "
                "If you would like to override the file, please set the override argument to True."
            ))
            return

        image_filepaths = glob.glob(f"{os.getcwd()}/dataset/{DATASET_FILES[dataset]}")
        logging.info(f"Running {image_wrapper.name}.{dataset.value}.{evaluation.value} on {len(image_filepaths)} images.")

        # [TODO] multiprocessing speedup
        results = []
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
            results.to_json(out_filepath)
    
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
