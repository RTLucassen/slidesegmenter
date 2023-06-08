# SlideSegmenter
*SlideSegmenter* is a Python package for tissue and pen marking segmentation 
on low-magnification (1.25x) whole slide images (WSIs). 

> **Note**
> Model parameter files are not yet included at the moment.

## Installing *SlideSegmenter*
*SlideSegmenter* can be installed from GitHub:
```console
$ pip install git+https://github.com/RTLucassen/slidesegmenter
```

## Example
A minimal example of how *SlideSegmenter* can be used for loading WSIs.
```
from slidesegmenter import SlideSegmenter

# initialize SlideSegmenter instance
segmenter = SlideSegmenter(device='cpu')

# segment the tissue and pen markings
cross_sections, pen_markings = segmenter.segment(low_magnification_image)
```

## Details
*SlideSegmenter* uses a convolutional neural network for segmentation.
To divide the tissue segmentation in separate cross-sections, 
a horizontal and vertical distance map are additionally predicted for post-processing, 
similar to HoVer-Net.
