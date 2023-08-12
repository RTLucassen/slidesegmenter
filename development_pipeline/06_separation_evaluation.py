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
Loop for hyperparameter search and evaluation of cross-section separation performance.
"""

import itertools
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from config import images_folder, annotations_folder, predictions_folder
from config import sheets_folder, models_folder
from slidesegmenter import SlideSegmenter
from utils.dataset_utils import SupervisedTrainingDataset
from utils.evaluation_utils import mean_stdev
from utils.models import ModifiedUNet


# define settings
dataset_sheet = 'dataset.xlsx'
sets = ['val']
model_subfolder = 'Modified_U-Net_2023-08-08_16h26m26s'
model_settings = 'settings.json'
model_checkpoint = 'checkpoint_I48500.tar'
device = 'cpu'
save_results = True
save_predictions = True

# define hyperparameter values for thresholds selected based on previous step
tissue_threshold = 0.3
pen_threshold = 0.1
# define hyperparameter values for grid search
pixels_per_bin_values = [10, 15, 20]
sigmas = [None, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0]
filter_sizes = [5, 7, 9, 11, 15]
percentiles = [95, 98, 99, 99.5]


if __name__ == '__main__':

    # define output path
    output_path = predictions_folder / f'SEP-{"_".join(sets)}-{model_subfolder}'
    if output_path.exists():
        raise FileExistsError('Output directory already exists.')
    else:
        os.mkdir(output_path)

    # define path to dataset sheet
    sheet_path = sheets_folder / dataset_sheet

    # load sheet and select the sets of interest
    df = pd.read_excel(sheet_path)
    df = df[df['set'].isin(sets)]

    # get all filenames from images in the specified batch
    image_paths = [images_folder / path for path in list(df['image_paths'])]
    annotation_paths = [annotations_folder / path for path in list(df['annotation_paths'])]

    # initialize inference dataset instance and dataloader
    dataset = SupervisedTrainingDataset(
        df=pd.DataFrame.from_dict({
            'image_paths': image_paths, 
            'annotation_paths': annotation_paths,
        }), 
        return_image_name=True,
        return_N_cross_sections=True,
    )
    dataloader = DataLoader(
        dataset=dataset, 
        sampler= SequentialSampler(dataset),
        batch_size=1,
        num_workers=1,
        pin_memory=True,
    )

    # configure model
    settings_path = models_folder / model_subfolder / model_settings
    checkpoint_path = models_folder / model_subfolder /  model_checkpoint
    segmenter = SlideSegmenter(channels_last=False, return_pen_segmentation=False, 
                               return_offset_maps=True)
    segmenter._load_model(ModifiedUNet, checkpoint_path, settings_path)

    # initalize dictionary to store results
    results = {}
    keys = ['image_name', 'pixels_per_bin', 'sigma', 'filter_size', 'percentile',
            'annotated_cross-sections', 'predicted_cross-sections', 'absolute_difference']

    # start evaluation
    with torch.no_grad():
        for image_name, cross_sections, image, annotations in tqdm(dataloader):

            # get the tissue annotations
            tissue_annotation = annotations[:, 0:1, ...]
            annotated_cross_sections = int(cross_sections)

            # get the model predictions
            predictions = segmenter.segment(
                image=image[0, ...], 
                tissue_threshold=tissue_threshold,
                pen_marking_threshold=pen_threshold,
                separate_cross_sections=False,
            )
            tissue_segmentation, horizontal_offset, vertical_offset = predictions

            # loop over all combinations of hyperparameter values in the grid search
            hyperparameters_ranges = [pixels_per_bin_values, sigmas, filter_sizes, percentiles]
            combinations = itertools.product(*hyperparameters_ranges)
            for index, hyperparameters in tqdm(enumerate(combinations)):
                
                # check if the index for a particular threshold value already exists
                if index not in results:
                    results[index] = {key: [] for key in keys}

                # assign hyperparameter values 
                segmenter.pixels_per_bin = hyperparameters[0]
                segmenter.sigma = hyperparameters[1]
                segmenter.filter_size = hyperparameters[2]
                segmenter.percentile = hyperparameters[3]
                
                # separate cross-sections
                separated_cross_sections, centroids = segmenter._separate_cross_sections(
                    segmentation=tissue_segmentation[0, ...], 
                    horizontal_offset=horizontal_offset[0, ...], 
                    vertical_offset=vertical_offset[0, ...],
                )
                # determine the number of predicted cross-sections
                predicted_cross_sections = separated_cross_sections.shape[-1]
                
                # save the results
                results[index]['image_name'].append(image_name[0])
                results[index]['pixels_per_bin'].append(hyperparameters[0])
                results[index]['sigma'].append(hyperparameters[1])
                results[index]['filter_size'].append(hyperparameters[2])
                results[index]['percentile'].append(hyperparameters[3])
                results[index]['annotated_cross-sections'].append(annotated_cross_sections)
                results[index]['predicted_cross-sections'].append(predicted_cross_sections)
                results[index]['absolute_difference'].append(
                    abs(annotated_cross_sections-predicted_cross_sections),
                )

                if save_predictions:
                    # create image
                    if isinstance(image, torch.Tensor):
                        image = image.numpy()
                    image = (image*255).astype(np.uint8)
                    
                    # create image
                    _, _, height, width = image.shape
                    horizontal_gradient = [x/width*255 for x, _ in centroids]
                    vertical_gradient = [y/height*255 for _, y in centroids]

                    red_channel =  np.sum(separated_cross_sections*np.array([[vertical_gradient]]), axis=-1, keepdims=True) * 0.8
                    green_channel = np.sum(separated_cross_sections*np.array([[horizontal_gradient]]), axis=-1, keepdims=True)
                    blue_channel = np.sum(separated_cross_sections, axis=-1, keepdims=True)*255

                    combined = np.concatenate([red_channel, green_channel, blue_channel], axis=-1)
                    combined = combined.astype(np.uint8).transpose((2,0,1))[None, ...]

                    if True:
                        s = 5
                        for x, y in centroids:
                            combined[:, :, int(y)-s:int(y)+s+1, int(x)-s:int(x)+s+1] = 255

                    # create the figure
                    figure = np.concatenate((image, combined), axis=-1).transpose((0,2,3,1)).astype(np.uint8)
                    # save the figure
                    sitk.WriteImage(sitk.GetImageFromArray(figure), output_path / image_name[0])

    if save_results:
        # initialize dictionary to combine all results
        total_keys = keys[:-3]+['mean_absolute_difference', 'stdev_absolute_difference']
        total_results = {key: [] for key in total_keys}

        # calculate average dice scores for each threshold value
        for data in results.values():
            total_results['image_name'].append('total')
            total_results['pixels_per_bin'].append(data['pixels_per_bin'][0])
            total_results['sigma'].append(data['sigma'][0])
            total_results['filter_size'].append(data['filter_size'][0])
            total_results['percentile'].append(data['percentile'][0])
            # calculate mean and stdev for the absolute difference
            mean_abs_difference, stdev_abs_difference = mean_stdev(data['absolute_difference'])
            total_results['mean_absolute_difference'].append(mean_abs_difference)
            total_results['stdev_absolute_difference'].append(stdev_abs_difference)
        # convert to dataframe
        total_results_df = pd.DataFrame.from_dict(total_results)
        
        # add the intividual results
        all_results = {key: [] for key in keys}
        for data in results.values():
            for key in keys:
                all_results[key].extend(data[key])
        # convert to dataframe
        all_results_df = pd.DataFrame.from_dict(all_results)

        # create a excel writer object
        with pd.ExcelWriter(output_path / 'results.xlsx') as writer:
            total_results_df.to_excel(writer, sheet_name='Total', index=False)
            all_results_df.to_excel(writer, sheet_name="All", index=False)