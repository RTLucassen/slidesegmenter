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
Annotate low magnification whole slide images in the dataset.
"""

import json
import os
from math import ceil, floor

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy.ndimage import gaussian_filter, maximum_filter
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from config import images_folder, annotations_folder, models_folder
from utils.dataset_utils import InferenceDataset
from utils.models import AdaptedUNet
from utils.visualization_utils import image_viewer


# define post-processing routine
def separate_cross_sections(
    segmentation: torch.Tensor, 
    horizontal_offset: torch.Tensor,
    vertical_offset: torch.Tensor, 
) -> torch.Tensor:
    """
    Separate cross-sections in the predicted segmentation map,
    based on the predicted horizontal and vertical distance maps.
    """
    # define parameter settings for segmentation and distance map correction
    offset_factor = 100
    # define parameter values for separating cross sections
    bins = 200
    sigma = 5
    filter_size = 9
    percentile = 99
    
    # initialize a variable with the image shape
    image_shape = segmentation.shape

    # create a vector with the binarized segmentation result for masking    
    mask = np.where(segmentation >= threshold, True, False).reshape((-1,))

    # create horizontal and vertical grid
    vertical_map, horizontal_map = np.meshgrid(
        np.linspace(0, image_shape[0]-1, image_shape[0]),
        np.linspace(0, image_shape[1]-1, image_shape[1]),
        indexing="ij",
    )
    # create the centroid maps
    x_centroid_map = (horizontal_map - (horizontal_offset*offset_factor))
    y_centroid_map = (vertical_map - (vertical_offset*offset_factor))

    # flatten the centroid map and select only the tissue regions
    x_centroid_flat = x_centroid_map.reshape((-1,))[mask]
    y_centroid_flat = y_centroid_map.reshape((-1,))[mask]

    # create 2D histogram
    histogram, y_edges, x_edges = np.histogram2d(
        y_centroid_flat, 
        x_centroid_flat, 
        bins=bins,
    )
    # apply Gaussian filtering to decrease local peaks
    histogram = gaussian_filter(histogram, sigma=sigma)
    histogram_mask = np.where(histogram > np.percentile(histogram, percentile), 1, 0)
    max_filtered_histogram = maximum_filter(histogram, filter_size)
    maxima = np.where(histogram == max_filtered_histogram, 1, 0)*histogram_mask

    # convert the edges from ranges to the center value
    x_bins = np.array([sum(x_edges[i:i+2])/2 for i in range(bins)])
    y_bins = np.array([sum(y_edges[i:i+2])/2 for i in range(bins)])

    # get the centroid coordinates
    indices = np.argwhere(maxima)
    centroids = np.concatenate(
        [x_bins[indices[:, 1], None], y_bins[indices[:, 0], None]], 
        axis=1,
    )
    centroid_coords = list(zip((centroids[:, 0]), centroids[:, 1]))

    # combine the x and y centroid maps into one array
    predicted_centroids = [
        x_centroid_map[..., None, None], 
        y_centroid_map[..., None, None],
    ]
    predicted_centroid_array = np.concatenate(predicted_centroids, axis=-1)
    
    # flatten the array and select only the tissue regions
    predicted_centroid_flat = predicted_centroid_array.reshape((-1,1,2))[mask, ...]

    # for each pixel, determine the distance between the predicted centroid
    # and all extracted centroids. Broadcasting is used for efficiency:
    # - predicted_centroid_array: [x*y, 1, 2] -> [x*y, N_centroids, 2]
    # - centroid_array: [1, N_centroids, 2] ->  [x*y, N_centroids, 2]
    distance_flat = np.sum((predicted_centroid_flat-centroids[None, ...])**2, axis=-1)
    
    # determine for each pixel what the nearest centroid is
    nearest_centroid_flat = np.argmin(distance_flat, axis=-1)+1

    # get the x and y coordinates for the pixels in the segmentation for indexing
    horizontal_flat = horizontal_map.reshape((-1,)).astype(np.uint16)[mask]
    vertical_flat = vertical_map.reshape((-1,)).astype(np.uint16)[mask]
    
    # convert back from the nearest centroid vector to the image
    nearest_centroid_map = np.zeros(image_shape)
    nearest_centroid_map[vertical_flat, horizontal_flat] = nearest_centroid_flat

    return nearest_centroid_map, centroid_coords


# define settings
batch = 'batch-2'
model_subfolder = 'Adapted_U-Net_2023-05-14_19h05m58s'
model_settings = 'settings.json'
model_checkpoint = 'checkpoint_I14500.tar'
threshold = 0.90
show_results = True
save_prediction = False
device = 'cpu'

if __name__ == '__main__':

    # define paths
    images_subfolder = images_folder / batch
    annotations_subfolder = annotations_folder / batch

    # get all filenames from images in the specified batch
    image_names = os.listdir(images_subfolder)

    # check if an annotation folder for the batch exists
    # if the folder already exists, get all filenames from annotations
    if not annotations_subfolder.exists():
        annotations_subfolder.mkdir()

    # prepare paths to images (and optionally annotations)
    paths = [images_subfolder / image_name for image_name in image_names]
    df = pd.DataFrame.from_dict({'image_paths': paths})

    # initialize inference dataset instance and dataloader
    dataset = InferenceDataset(
        df=df, 
    )
    dataloader = DataLoader(
        dataset=dataset, 
        sampler= SequentialSampler(dataset),
        batch_size=1,
        num_workers=1,
        pin_memory=True,
    )

    # load model settings
    with open(models_folder / model_subfolder / model_settings, 'r') as f:
        settings = json.load(f)

    # load the model
    checkpoint = torch.load(
        models_folder / model_subfolder /  model_checkpoint,
        map_location=torch.device(device)
    )
    model = AdaptedUNet(**settings['model']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    divisor = np.prod(settings['model']['downsample_factors'])

    i = 0
    with torch.no_grad():
        for X in tqdm(dataloader):
            
            _, _, height, width = X.shape

            # determine the total amount of padding necessary
            width_pad = (ceil(width / divisor)*divisor)-width
            height_pad = (ceil(height / divisor)*divisor)-height
            
            # determine the amount of padding on each side of the image
            padding = [
                (0,0),
                (0,0),
                (floor(height_pad/2), ceil(height_pad/2)), 
                (floor(width_pad/2), ceil(width_pad/2)), 
            ]

            # add padding to image
            X = np.pad(
                array=X.numpy(), 
                pad_width=padding, 
                mode='constant', 
                constant_values=1,
            )
            # convert the image to a torch Tensor
            X = torch.from_numpy(X)
            
            # get model prediction
            y_pred = model(X.to(device)).to('cpu')

            # show the result
            top = padding[2][0]
            left = padding[3][0]
            
            # remove the padding
            tissue = torch.sigmoid(y_pred[:, 0:1, top:top+height, left:left+width])
            #pen = torch.zeros_like(tissue)
            pen = torch.sigmoid(y_pred[:, 1:2, top:top+height, left:left+width])
            horizontal = y_pred[:, -2:-1, top:top+height, left:left+width]
            vertical = y_pred[:, -1:, top:top+height, left:left+width]

            # get cross-sections
            cross_sections, _ = separate_cross_sections(
                 tissue.numpy()[0, 0, ...], 
                 horizontal.numpy()[0, 0, ...], 
                 vertical.numpy()[0, 0,...],
            )

            cross_sections = cross_sections[None, None, ...]

            centroids = int(np.max(cross_sections))
            cross_sections = np.tile(cross_sections, (1, centroids, 1, 1))
            layers = np.ones_like(cross_sections)*np.tile(np.arange(1, centroids+1)[None, :, None, None], (1, 1, height, width))
            cross_sections = torch.from_numpy(np.where(cross_sections == layers, 1, 0))

            if show_results:
                image_viewer(
                    torch.concat([tissue, pen], dim=0), 
                    vmin=0, 
                    vmax=1,
                )
                image_viewer(cross_sections)
                image_viewer(
                    torch.concat([vertical, horizontal], dim=0), 
                    vmin=-15, 
                    vmax=15,
                )

            # save the prediction
            annotation_name = image_names[i].replace('.png', '_annotation-000.tiff')
            output_path = annotations_folder / batch / annotation_name
            if not output_path.exists() and save_prediction:
                binary_segmentation = torch.concatenate([tissue, pen, cross_sections], dim=1)
                binary_segmentation = torch.where(binary_segmentation > threshold, 255, 0)
                binary_segmentation = binary_segmentation.to(torch.uint8)
                binary_segmentation = binary_segmentation.permute((1,2,3,0))
                sitk.WriteImage(sitk.GetImageFromArray(binary_segmentation), output_path)

            i += 1