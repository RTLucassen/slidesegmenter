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
Utility class for instance segmentation of tissue cross-sections 
and semantic segmentation of pen markings.
"""

import json
from math import ceil, floor
from pathlib import Path
from typing import Union

import torch
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from ._model_utils import AdaptedUNet
from . import model


# define settings
SETTINGS_FILE = 'settings.json'
CHECKPOINT_FILE = 'model_checkpoint.tar'

class SlideSegmenter:
    """
    Class for segmenting tissue and pen markings on low resolution (1.25x)
    whole slide images. The class is responsible for preprocessing 
    (i.e., padded to a valid size), running model inference to get the segmentations, 
    and post-processing (i.e., cropping to the original size and optionally 
    separating tissue cross-sections).
    """

    # define parameter settings for segmentation and distance map correction
    padding_mode = 'constant'
    padding_value = 0  # in case of 'constant' padding mode
    offset_factor = 100
    
    # define parameter values for separating cross sections
    bins = 200
    sigma = 2.5
    filter_size = 5
    percentile = 99
    
    def __init__(
        self, 
        threshold: float = 0.9,
        channels_last: bool = True,
        return_offset_maps: bool = False,
        device: str = 'cpu', 
    ) -> None:
        """
        Initialize SlideSegmenter instance.

        Args:
            threshold: threshold value for binarizing the predicted segmentation.
            channels_last: indicates whether the input is expected to have 
                           the channels dimension after the spatial dimension.
                           If False, channels first is assumed.
            return_offset_maps: indicates whether predicted horizontal and vertical 
                                offset maps are additionally returned.
            device: specifies whether the pytorch model is executed on cpu or gpu.
        """
        # load model settings
        with open(Path(model.__file__).parent / SETTINGS_FILE, 'r') as f:
            settings = json.load(f)

        # load the model parameters
        self.device = torch.device(device)
        checkpoint = torch.load(
            Path(model.__file__).parent / CHECKPOINT_FILE, 
            map_location=self.device,
        )
        # configure model
        self.model = AdaptedUNet(**settings['model'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # create instance attributes
        self.threshold = threshold
        self.channels_last = channels_last
        self.return_offset_maps = return_offset_maps

        # determine by what value the image height and width must be divisible
        self.divisor = np.prod(settings['model']['downsample_factors'])

    def segment(
        self, 
        image: Union[np.ndarray, torch.Tensor],
        separate_cross_sections: bool = True,
    ) -> Union[np.ndarray, tuple]:
        """
        Steps in segmentation pipeline:
        (1) Preprocess the image by adding padding to make the length of the 
        height and width valid.
        (2) Predict the tissue and pen marking segmentation for the image. 
        (3) Post-process the segmentation by cropping it to the original size.
        (4) Optionally divide the tissue segmentations into separate cross-sections.

        Args:
            image: whole slide image (at 1.25x) [uint8] as (height, width, channel)
                   for channels last or (channel, height, width) for channels first.
            separate_cross_sections: indicates whether the tissue segmentation
                                     should be returned as separate cross-sections.
        Returns:
            segmentation: segmentation for whole slide image [float32] (at 1.25x) 
                          as (height, width, channel) for channels last or 
                          (channel, height, width) for channels first.
            horizontal_offset: image with predicted horizontal offset [float32]
                               with respect to corresponding centroid.
            vertical_offset: image with predicted vertical offset [float32]
                             with respect to corresponding centroid.
        """
        # check image object type, convert to numpy array if necessary
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        elif not isinstance(image, np.ndarray):
            raise TypeError('Invalid type of input argument for image.')
        
        # check if the images intensities are in the range of 0.0-1.0
        if np.min(image) < 0 or np.max(image) > 1:
            raise ValueError('Invalid image intensities (must be in the range 0.0-1.0)')

        # check if the image input argument is valid
        if len(image.shape) != 3:
            raise ValueError('Invalid number of dimensions for input argument.')

        # change the channels dimension to be the first dimension if necessary
        if self.channels_last:
            image = image.transpose((2, 0, 1))

        # determine the height and width of the input image       
        channels, height, width = image.shape
        
        # check if the number of channels is valid:
        if channels != 3:
            raise ValueError('Invalid number of channels for input argument.') 
        
        # determine the total amount of padding necessary
        width_pad = (ceil(width / self.divisor)*self.divisor)-width
        height_pad = (ceil(height / self.divisor)*self.divisor)-height
        
        # determine the amount of padding on each side of the image
        padding = [
            (0,0),
            (floor(height_pad/2), ceil(height_pad/2)), 
            (floor(width_pad/2), ceil(width_pad/2)), 
        ]
        # add padding to image
        image = np.pad(
            array=image,
            pad_width=padding, 
            mode=self.padding_mode, 
            constant_values=self.padding_value,
        )
        # convert the image to a torch Tensor
        image = torch.from_numpy(image).float()
        # get the model prediction
        with torch.no_grad():
            prediction = self.model(image[None, ...].to(self.device))
        
        # independent of the device, bring the prediction to the cpu and remove
        # the batch dimension
        prediction = prediction.to('cpu')[0, ...]

        # crop the padding from the prediction and separate the channels
        top = padding[1][0]
        left = padding[2][0]
        prediction = prediction[:, top:top+height, left:left+width]
        # separate the channels and apply the final activation functions
        segmentation = torch.sigmoid(prediction[0, ...]).numpy()
        horizontal_offset = prediction[1, ...].numpy()
        vertical_offset = prediction[2, ...].numpy()

        # create cross-sections
        if separate_cross_sections:
            segmentation, _ = self._separate_cross_sections(
                segmentation, 
                horizontal_offset, 
                vertical_offset,
            )

        # change channels first / last

        if self.return_offset_maps:
            return segmentation, horizontal_offset, vertical_offset
        else:
            return segmentation

    def _separate_cross_sections(
        self,
        segmentation: torch.Tensor, 
        horizontal_offset: torch.Tensor,
        vertical_offset: torch.Tensor, 
    ) -> torch.Tensor:
        """
        Separate cross-sections in the predicted segmentation map,
        based on the predicted horizontal and vertical distance maps.

        Args:
            segmentation: segmentation for whole slide image [float32] (at 1.25x) 
                          as (height, width, channel) for channels last or 
                          (channel, height, width) for channels first.
            horizontal_offset: image with predicted horizontal offset [float32]
                               with respect to corresponding centroid.
            vertical_offset: image with predicted vertical offset [float32]
                             with respect to corresponding centroid.
        Returns:
            nearest_centroid_map: segmentation for whole slide image [float32] 
                                  (at 1.25x)
                          as (height, width, channel) for channels last or 
                          (channel, height, width) for channels first.
            centroid_coords: coordinates of extracted centroids.
        """
        # initialize a variable with the image shape
        image_shape = segmentation.shape

        # create a vector with the binarized segmentation result for masking    
        mask = np.where(segmentation >= self.threshold, True, False).reshape((-1,))

        # create horizontal and vertical grid
        vertical_map, horizontal_map = np.meshgrid(
            np.linspace(0, image_shape[0]-1, image_shape[0]),
            np.linspace(0, image_shape[1]-1, image_shape[1]),
            indexing="ij",
        )
        # create the centroid maps
        x_centroid_map = (horizontal_map - (horizontal_offset*self.offset_factor))
        y_centroid_map = (vertical_map - (vertical_offset*self.offset_factor))

        # flatten the centroid map and select only the tissue regions
        x_centroid_flat = x_centroid_map.reshape((-1,))[mask]
        y_centroid_flat = y_centroid_map.reshape((-1,))[mask]

        # create 2D histogram
        histogram, y_edges, x_edges = np.histogram2d(
            y_centroid_flat, 
            x_centroid_flat, 
            bins=self.bins,
        )
        # apply Gaussian filtering to decrease local peaks
        histogram = gaussian_filter(histogram, sigma=self.sigma)
        histogram_mask = np.where(histogram > np.percentile(histogram, self.percentile), 1, 0)
        max_filtered_histogram = maximum_filter(histogram, self.filter_size)
        maxima = np.where(histogram == max_filtered_histogram, 1, 0)*histogram_mask

        # convert the edges from ranges to the center value
        x_bins = np.array([sum(x_edges[i:i+2])/2 for i in range(self.bins)])
        y_bins = np.array([sum(y_edges[i:i+2])/2 for i in range(self.bins)])
        
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