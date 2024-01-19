# SlideSegmenter
*SlideSegmenter* is a Python 3.9+ package for tissue cross-section and pen marking segmentation 
in low-magnification (1.25x) whole slide images (WSIs). For annotation, we developed and used a custom bitmap [annotation tool](https://github.com/RTLucassen/annotation_tool). 

<div align="center">
  <img width="100%" alt="Method" src=".github\method.png">
</div>

## Installing *SlideSegmenter*
*SlideSegmenter* can be installed from GitHub:

```console
$ pip install git+https://github.com/RTLucassen/slidesegmenter
```

## Example
A minimal example of how *SlideSegmenter* can be used for segmenting WSIs.

```python
from slidesegmenter import SlideSegmenter

# initialize SlideSegmenter instance
segmenter = SlideSegmenter(device='cpu', model_folder='latest')

# segment the tissue and pen markings
cross_sections, pen_markings = segmenter.segment(low_magnification_image)
```

## Versions
Multiple versions of the model are available, which can be selected by specifying 
the name of the model folder name (see options below) as argument when initializing
a *SlideSegmenter* instance. By default, the latest version of the model is used.
- `'2024-01-10'`: Model from the corresponding paper, trained on 140 WSIs of H&E stained skin biopsies and excisions.
- `'2023-08-13'`: Model from a prior version of the corresponding paper, trained on 100 WSIs of H&E stained skin biopsies and excisions.

## Output
The output depends on the configuration of the *SlideSegmenter* instance (see cases below).
At initialization, there is the option to disable: **(1)** tissue segmentation, **(2)** pen marking segmentation, and **(3)** cross-section separation. 
If any of these are disabled, the corresponding decoders of the model will not be initialized, making inference faster. By default, all are enabled.

<details>
<summary>
<b>Case 1. All enabled</b>
</summary>
  
```python
segmenter = SlideSegmenter(tissue_segmentation=True, pen_marking_segmentation=True, separate_cross_sections=True)
cross_sections, pen_markings = segmenter.segment(low_magnification_image)
# cross_sections: [H, W, S], pen_markings: [H, W, 1], where H = image height, W = image width, S = number of cross-sections
```
</details>

<details>
<summary>
<b>Case 2. Cross-section separation disabled</b>
</summary>
  
```python
segmenter = SlideSegmenter(tissue_segmentation=True, pen_marking_segmentation=True, separate_cross_sections=False)
tissue, pen_markings = segmenter.segment(low_magnification_image)
# tissue: [H, W, 1], pen_markings: [H, W, 1], where H = image height, W = image width
```
</details>

<details>
<summary>
<b>Case 3. Pen marking segmentation disabled</b>
</summary>
  
```python
segmenter = SlideSegmenter(tissue_segmentation=True, pen_marking_segmentation=False, separate_cross_sections=True)
cross_sections = segmenter.segment(low_magnification_image)
# cross_sections: [H, W, S], where H = image height, W = image width, S = number of cross-sections
```
</details>

<details>
<summary>
<b>Case 4. Cross-section separation and pen marking segmentation disabled</b>
</summary>
  
```python
segmenter = SlideSegmenter(tissue_segmentation=True, pen_marking_segmentation=False, separate_cross_sections=False)
tissue = segmenter.segment(low_magnification_image)
# tissue: [H, W, 1], where H = image height, W = image width
```
</details>

<details>
<summary>
<b>Case 5. Tissue segmentation disabled</b>
</summary>
  
```python
segmenter = SlideSegmenter(tissue_segmentation=False, pen_marking_segmentation=True, separate_cross_sections=False)
pen_markings = segmenter.segment(low_magnification_image)
# pen_markings: [H, W, 1], where H = image height, W = image width
```
</details>
