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
Loop for hyperparameter search and evaluation of segmentation performance.
"""

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
from utils.evaluation_utils import mean_stdev, dice_score
from utils.models import ModifiedUNet


# define settings
dataset_sheet = 'dataset.xlsx'
sets = ['val']
model_subfolder = 'Modified_U-Net_2024-01-10_13h41m39s'
model_settings = 'settings.json'
model_checkpoint = 'checkpoint_I97500.tar'
device = 'cpu'
save_results = True
save_predictions = True

# define hyperparameter values for grid search
segmentation_thresholds = np.arange(0, 1.01, 0.01)

# define threshold values for evaluation
tissue_threshold = None # 0.5
pen_threshold = None # 0.5


if __name__ == '__main__':

    # define output path
    output_path = predictions_folder / f'SEG-{"_".join(sets)}-{model_subfolder}-{"&".join(sets)}'
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
    segmenter = SlideSegmenter(channels_last=False, separate_cross_sections=False)
    segmenter._load_model(ModifiedUNet, checkpoint_path, settings_path)

    # initalize dictionary to store results
    results = {}
    if segmentation_thresholds is not None:
        threshold_keys = ['segmentation_threshold']
    else:
        threshold_keys = ['tissue_threshold', 'pen_threshold']     
    keys = ['image_name'] + threshold_keys + ['tissue_dice', 'pen_dice']

    # start evaluation
    with torch.no_grad():
        for image_name, image, annotations in tqdm(dataloader):

            # get the annotations
            tissue_annotation = annotations[:, 0:1, ...].numpy()
            pen_annotation = annotations[:, 1:2, ...].numpy()

            # get the model predictions
            predictions = segmenter.segment(image[0, ...], tissue_threshold=None, 
                                            pen_marking_threshold=None)
            tissue_prediction, pen_prediction = predictions

            # loop over all combinations of hyperparameter values in the grid search
            if segmentation_thresholds is not None:
                for index, threshold in enumerate(segmentation_thresholds):
                    # check if the index for a particular threshold value already exists
                    if index not in results:
                        results[index] = {key: [] for key in keys}

                    # binarize the segmentation predictions
                    binary_tissue_prediction = np.where(tissue_prediction>=threshold, 1, 0)[None, ...]
                    binary_pen_prediction = np.where(pen_prediction>=threshold, 1, 0)[None, ...]
             
                    # calculate dice scores
                    tissue_dice = dice_score(
                        y_hat=binary_tissue_prediction, 
                        y_true=tissue_annotation,
                    )
                    pen_dice = dice_score(
                        y_hat=binary_pen_prediction, 
                        y_true=pen_annotation,
                    )
                    # save the results
                    results[index]['image_name'].append(image_name[0])
                    results[index]['segmentation_threshold'].append(threshold)
                    results[index]['tissue_dice'].append(tissue_dice.item())
                    results[index]['pen_dice'].append(pen_dice.item())
            else:
                # check if the index for a particular threshold value already exists
                index = 0
                if index not in results:
                    results[index] = {key: [] for key in keys}

                # binarize the segmentation predictions
                binary_tissue_prediction = np.where(tissue_prediction>=tissue_threshold, 1, 0)[None, ...]
                binary_pen_prediction = np.where(pen_prediction>=pen_threshold, 1, 0)[None, ...]
   
                # calculate dice scoresq
                tissue_dice = dice_score(
                    y_hat=binary_tissue_prediction, 
                    y_true=tissue_annotation,
                )
                pen_dice = dice_score(
                    y_hat=binary_pen_prediction, 
                    y_true=pen_annotation,
                )
                # save the results
                results[0]['image_name'].append(image_name[0])
                results[0]['tissue_threshold'].append(tissue_threshold)
                results[0]['pen_threshold'].append(pen_threshold)
                results[0]['tissue_dice'].append(tissue_dice.item())
                results[0]['pen_dice'].append(pen_dice.item())

                if save_predictions:
                    # create image
                    if isinstance(image, torch.Tensor):
                        image = image.numpy()
                    image = (image*255).astype(np.uint8)
                    # create the tissue segmentation and prediction overlap image
                    tissue_channels = (
                        binary_tissue_prediction, 
                        tissue_annotation*binary_tissue_prediction, 
                        tissue_annotation,
                    )
                    tissue_overlap = np.concatenate(tissue_channels, axis=1).astype(np.uint8)*255
                    # create the pen marking segmentation and prediction overlap image
                    pen_channels = (
                        binary_pen_prediction, 
                        pen_annotation*binary_pen_prediction, 
                        pen_annotation,
                    )
                    pen_overlap = np.concatenate(pen_channels, axis=1).astype(np.uint8)*255
                    # create the figure
                    figure = np.concatenate((image, tissue_overlap, pen_overlap), axis=-1).transpose((0,2,3,1))
                    # save the figure
                    sitk.WriteImage(sitk.GetImageFromArray(figure), output_path / image_name[0])

    if save_results:
        # initialize dictionary to store the total results
        total_keys = ['image_name'] + threshold_keys + ['mean_tissue_dice', 
                    'stdev_tissue_dice', 'mean_pen_dice', 'stdev_pen_dice']
        total_results = {key: [] for key in total_keys}

        # calculate average dice scores for each threshold value
        for data in results.values():
            total_results['image_name'].append('total')
            if segmentation_thresholds is not None:
                total_results['segmentation_threshold'].append(data['segmentation_threshold'][0])
            else:
                total_results['tissue_threshold'].append(data['tissue_threshold'][0])
                total_results['pen_threshold'].append(data['pen_threshold'][0])
            # calculate mean and stdev for tissue Dice scores
            mean_tissue_dice, stdev_tissue_dice = mean_stdev(data['tissue_dice'])
            total_results['mean_tissue_dice'].append(mean_tissue_dice)
            total_results['stdev_tissue_dice'].append(stdev_tissue_dice)
            # calculate mean and stdev for pen markings Dice scores
            mean_pen_dice, stdev_pen_dice = mean_stdev(data['pen_dice'])
            total_results['mean_pen_dice'].append(mean_pen_dice)
            total_results['stdev_pen_dice'].append(stdev_pen_dice)
        # convert to dataframe
        total_results_df = pd.DataFrame.from_dict(total_results)

        # initialize dictionary to store the total results
        all_results = {key: [] for key in keys}
        # add the individual results
        for data in results.values():
            for key in keys:
                all_results[key].extend(data[key])
        # convert to dataframe
        all_results_df = pd.DataFrame.from_dict(all_results)

        # create a excel writer object
        with pd.ExcelWriter(output_path / 'results.xlsx') as writer:
            total_results_df.to_excel(writer, sheet_name='Total', index=False)
            all_results_df.to_excel(writer, sheet_name="All", index=False)