#    Copyright 2023 Ruben T Lucassen, UMC Utrecht, The Netherlands 
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Select the images and corresponding annotations to include in the dataset
and randomly assign them to the training or validation set.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from skimage.filters import gaussian

from config import annotations_folder, images_folder, sheets_folder


def get_bounding_boxes(binary_image: np.ndarray, sigma: float,
    ) -> list[tuple[int, int, int, int]]:
    """
    Return the bounding box coordinates for all isolated components in a binary 2D image.

    Args:
        binary_image:  Binary image to determine bounding boxes for.
        sigma:  Standard deviation for Gaussian filter to join neighboring 
            isolated components and to add padding.

    Returns:
        bounding_boxes:  Bounding box coordinates as (min_y, min_x, max_y, max_x).
    """
    # filter the binary image to join neighboring isolated components and to add padding
    filtered_binary_image = gaussian(binary_image, sigma)

    # Label connected components in the binary image
    labeled_array, num_features = ndimage.label(filtered_binary_image)

    bounding_boxes = []
    for label in range(1, num_features + 1):
        # Find coordinates of labeled region
        coordinates = np.column_stack(np.where(labeled_array == label))

        # Calculate bounding box coordinates
        min_y, min_x = np.min(coordinates, axis=0)
        max_y, max_x = np.max(coordinates, axis=0)
        bounding_boxes.append((min_y, min_x, max_y, max_x))

    return bounding_boxes

# define settings
subfolder = 'pen_markings'
output_extension = 'png'
sets = ['train']
dataset_sheet = 'dataset.xlsx'
border_sigma = 10
joining_sigma = 10
size_threshold = 250 # px
intensity_threshold = 242


if __name__ == '__main__':

    # check if pen dataset sheet is available
    if not (sheets_folder/dataset_sheet).exists():
        raise FileNotFoundError('Dataset sheet file does not exists.')

    # load the dataset information
    df = pd.read_excel(sheets_folder/dataset_sheet)

    # check if information about pen marking presence if available
    if 'pen_marking_present' not in df.columns:
        raise ValueError('Information about pen marking presence not available.')

    # create a new folder to store the pen marking images
    pen_images_subfolder = images_folder / subfolder
    if not pen_images_subfolder.exists():
        pen_images_subfolder.mkdir()
    else:
        raise FileExistsError
    
    # create a new folder to store the pen marking annotations
    pen_annotations_subfolder = annotations_folder / subfolder
    if not pen_annotations_subfolder.exists():
        pen_annotations_subfolder.mkdir()
    else:
        raise FileExistsError

    # create lists to store pen
    pen_image_paths = []
    pen_annotation_paths = []
    pen_set = []

    # select all images from the sets with pen markings present
    df_selection = df[(df['set'].isin(sets)) & (df['pen_marking_present'] == True)]
    for i, row in df_selection.iterrows():
        # define image and annotation paths
        image_path = images_folder/row['image_paths']
        annotation_path = annotations_folder/row['annotation_paths']

        # read the image and annotation
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))   
        annotation = sitk.GetArrayFromImage(sitk.ReadImage(annotation_path))
        
        # get the tissue and pen marking annotations
        annotation = (annotation/255).astype(float)
        tissue = annotation[0, ...]
        pen_marking = annotation[1, ...]

        # create a pen marking mask by removing pen marking regions superimposed on tissue
        mask = gaussian(np.clip(pen_marking-tissue, 0, 1), border_sigma)
        mask = np.clip(mask-gaussian(tissue, border_sigma)*(2*np.pi*border_sigma*2)**0.5, 0, 1)[..., None]
        corrected_pen_marking = np.clip(pen_marking-gaussian(tissue, border_sigma)*(2*np.pi*border_sigma*2)**0.5, 0, 1)

        # create an image with the pen markings and a white background
        pen_marking_image = (image*mask+np.ones_like(image)*255*(1-mask)).astype(np.uint8)
        
        # get the bounding boxes
        boxes = get_bounding_boxes(mask[..., 0], joining_sigma)
        for i, (min_y, min_x, max_y, max_x) in enumerate(boxes):
            pen_size = np.sum(corrected_pen_marking[min_y:max_y, min_x:max_x])
            mean_pen_intensity = (
                np.sum(corrected_pen_marking[min_y:max_y, min_x:max_x, None]
                * pen_marking_image[min_y:max_y, min_x:max_x])
                / (pen_size*3)+1e-5
            )          
            # skip if the pen marking is too small            
            if pen_size < size_threshold:
                continue   
            # skip if the pen marking is too faint     
            elif mean_pen_intensity > intensity_threshold:
                continue 

            # save crop of pen marking image
            pen_image_path = pen_images_subfolder/f'{image_path.stem}_pen_marking_{i}.{output_extension}'
            pen_image_crop = pen_marking_image[min_y:max_y, min_x:max_x][None, ...]
            sitk.WriteImage(sitk.GetImageFromArray(pen_image_crop), pen_image_path)

            # save corresponding crop of pen marking annotation
            pen_annotation_path = pen_annotations_subfolder/f'{annotation_path.stem}_pen_marking_{i}.{output_extension}'
            pen_annotation_crop = corrected_pen_marking[min_y:max_y, min_x:max_x][None, ..., None]
            pen_annotation_crop = (pen_annotation_crop*255).astype(np.uint8)
            sitk.WriteImage(sitk.GetImageFromArray(pen_annotation_crop), pen_annotation_path)
            
            pen_image_paths.append(Path(subfolder, pen_image_path.name).as_posix())            
            pen_annotation_paths.append(Path(subfolder, pen_annotation_path.name).as_posix())
            pen_set.append(row['set'])

    # create dataframe
    df_pen = pd.DataFrame.from_dict({
        'set': pen_set,
        'image_paths': pen_image_paths,            
        'annotation_paths': pen_annotation_paths,
    })
    # add sheet to dataset file
    writer = pd.ExcelWriter(sheets_folder/dataset_sheet)
    df.to_excel(writer, sheet_name='images', index=False)
    df_pen.to_excel(writer, sheet_name='pen_markings', index=False)
    writer.close()