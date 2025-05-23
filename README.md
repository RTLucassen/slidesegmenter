# SlideSegmenter
*SlideSegmenter* is a Python 3.9+ package for tissue and pen marking segmentation in low-magnification (1.25x) whole slide images (WSIs). 
We also developed a custom bitmap [annotation tool](https://github.com/RTLucassen/annotation_tool), which was used in the development of the method. More details can be found in the corresponding paper. 

[[`SPIE`](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12933/129330B/Tissue-cross-section-and-pen-marking-segmentation-in-whole-slide/10.1117/12.3004683.short)] [[`arXiv`](https://arxiv.org/abs/2401.13511)]

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
segmentation = segmenter.segment(low_magnification_image)
```

## Versions
Multiple versions of the model are available, which can be selected by specifying 
the name of the model folder name (see options below) as argument when initializing
a *SlideSegmenter* instance. By default, the latest version of the model is used.
- `'2025-03-10'`: Model after the corresponding paper, trained on 840 WSIs of H&E stained skin biopsies and excisions. In comparison to the previous model version, 700 additional challenging WSIs were included. Annotations for these WSIs were obtained by manually correcting the segmentation predictions of the previous model version.
- `'2024-01-10'`: Model from the corresponding paper, trained on 140 WSIs of H&E stained skin biopsies and excisions.
- `'2023-08-13'`: Model from a prior version of the corresponding paper, trained on 100 WSIs of H&E stained skin biopsies and excisions.

## Configuration and Output
The output depends on the configuration of the *SlideSegmenter* instance.
At initialization, there is the option to enable or disable: **(1)** tissue segmentation, **(2)** pen marking segmentation, and **(3)** cross-section separation. 
If any of these are disabled, the corresponding decoders of the model will not be initialized, making inference faster. 
By default, tissue and pen marking segmentation are enabled and cross-section separation is disabled.

<details>
<summary>
<b>Case 1. All enabled</b>
</summary>
  
```python
segmenter = SlideSegmenter(tissue_segmentation=True, pen_marking_segmentation=True, separate_cross_sections=True)
segmentation = segmenter.segment(low_magnification_image)
# H = image height, W = image width, S = number of cross-sections
cross_sections = segmentation['tissue'] # [H, W, S]
pen_markings = segmentation['pen']      # [H, W, 1]
```
</details>

<details>
<summary>
<b>Case 2. All enabled (+ return distance maps)</b>
</summary>
  
```python
segmenter = SlideSegmenter(tissue_segmentation=True, pen_marking_segmentation=True, separate_cross_sections=True)
segmentation = segmenter.segment(low_magnification_image, return_distance_maps=True)
# H = image height, W = image width, S = number of cross-sections
cross_sections = segmentation['tissue']  # [H, W, S]
pen_markings = segmentation['pen']       # [H, W, 1]
distance_maps = segmentation['distance'] # [H, W, 2]
```
</details>

<details>
<summary>
<b>Case 3. Cross-section separation disabled</b>
</summary>
  
```python
segmenter = SlideSegmenter(tissue_segmentation=True, pen_marking_segmentation=True, separate_cross_sections=False)
segmentation = segmenter.segment(low_magnification_image)
# H = image height, W = image width
tissue = segmentation['tissue']    # [H, W, 1]
pen_markings = segmentation['pen'] # [H, W, 1]
```
</details>

<details>
<summary>
<b>Case 4. Pen marking segmentation disabled</b>
</summary>
  
```python
segmenter = SlideSegmenter(tissue_segmentation=True, pen_marking_segmentation=False, separate_cross_sections=True)
segmentation = segmenter.segment(low_magnification_image)
# H = image height, W = image width, S = number of cross-sections
cross_sections = segmentation['tissue'] # [H, W, S]
```
</details>

<details>
<summary>
<b>Case 5. Pen marking segmentation disabled (+ return distance maps)</b>
</summary>
  
```python
segmenter = SlideSegmenter(tissue_segmentation=True, pen_marking_segmentation=False, separate_cross_sections=True)
segmentation = segmenter.segment(low_magnification_image, return_distance_maps=True)
# H = image height, W = image width, S = number of cross-sections
cross_sections = segmentation['tissue']  # [H, W, S]
distance_maps = segmentation['distance'] # [H, W, 2]
```
</details>

<details>
<summary>
<b>Case 6. Cross-section separation and pen marking segmentation disabled</b>
</summary>
  
```python
segmenter = SlideSegmenter(tissue_segmentation=True, pen_marking_segmentation=False, separate_cross_sections=False)
segmentation = segmenter.segment(low_magnification_image)
# H = image height, W = image width
tissue = segmentation['tissue'] # [H, W, 1]
```
</details>

<details>
<summary>
<b>Case 7. Tissue segmentation disabled</b>
</summary>
  
```python
segmenter = SlideSegmenter(tissue_segmentation=False, pen_marking_segmentation=True, separate_cross_sections=False)
segmentation = segmenter.segment(low_magnification_image)
# H = image height, W = image width
pen_markings = segmentation['pen'] # [H, W, 1]
```
</details>

## Citing
If you found *SlideSegmenter* useful in your research, please consider citing our paper:
```
@inproceedings{lucassen2024tissue,
  author = {Ruben T Lucassen and Willeke A M Blokx and Mitko Veta},
  title = {{Tissue cross-section and pen marking segmentation in whole slide images}},
  volume = {12933},
  booktitle = {Medical Imaging 2024: Digital and Computational Pathology},
  organization = {International Society for Optics and Photonics},
  publisher = {SPIE},
  pages = {129330B},
  year = {2024},
  doi = {10.1117/12.3004683},
  URL = {https://doi.org/10.1117/12.3004683}
}
```
