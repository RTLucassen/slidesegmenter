# SlideSegmenter
*SlideSegmenter* is a Python 3.9+ package for tissue and pen marking segmentation 
on low-magnification (1.25x) whole slide images (WSIs). 

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
