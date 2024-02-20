This is an open-source library for benchmarking watermarking, developed by Trufo Inc. An [overview](https://medium.com/p/436cbf1bd9ca) of the library is on Trufo's blog.

### Get Started ###

To begin, clone the repository and download the [datasets](https://drive.google.com/drive/folders/1P3X_-_Ug8fewCxd-a_66Pumr9BsHmsqf?usp=sharing) into the `dataset/` folder.

This is an analysis library, so you will need to load a Python environment and add the repository root to your (Python) path. A Pipfile is provided; to use it, first [install pipenv](https://pypi.org/project/pipenv/) and then from the repository root, run:

```
pipenv install --python 3.11
```

To test the benchmark functionality, you can run:

```
pipenv run python bench.py
```

This will run a basic bench (encoding + decoding) using the (invisible watermark library)[https://pypi.org/project/invisible-watermark/] on the standard Lena test image.

Next, the `notebooks/example.ipynb` [notebook](https://github.com/byh-trufo/watermark-benchmark/blob/main/notebooks/example.ipynb) provides a more complete (and recommended) usage of the library. The notebook will run multiple benches and then use the analysis module to summarize the results.

### Structure ###

The `bench.py` file is the main benchmark function. It takes in a watermark wrapper (from `wrappers/`), an image dataset (see `dataset/`), and a testing mode (primarily, see `benchmark/durability.py`). The raw results are outputted in `.json` format to the `results/` folder, which the analysis module (at `analysis/`) will read in and summarize.

A wrapper for the (invisible watermark library)[https://pypi.org/project/invisible-watermark/] is included. There are three watermark modes: DwtDct, DwtDctSvd, and RivaGAN. Note that if you would like to run the RivaGAN implementation from the , you will need to install the `onnxruntime` and `torch` packages as well.

There are a number of [datasets](https://drive.google.com/drive/folders/1P3X_-_Ug8fewCxd-a_66Pumr9BsHmsqf?usp=sharing) included. `IMG_0` is just standard Lena test image. `IMG_1` is a set of 10 images that vary in size, style, formats. `IMG_VOC` and `IMG_BIG` and `IMG_ART` are test sets assembled by a student researcher, consisting of 132 and 17 and 47 images of medium and large and artistic types, respectively.

### Become a Contributor ###

If you are interested in contributing to this project, or just have comments and/or questions, contact us at (engineering@trufo.ai)!
