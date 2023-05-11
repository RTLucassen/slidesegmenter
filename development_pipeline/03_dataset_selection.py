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

import os
import random
import natsort
import pandas as pd
from pathlib import Path

from config import PROJECT_SEED, images_folder, annotations_folder, sheets_folder

subfolders = ['remaining']
dataset_name = 'supervised_remaining'
include_annotations = True
validation_fraction = 0.05

if __name__ == '__main__':

    # seed randomness
    random.seed(PROJECT_SEED)

    # if annotations should not be included, save only the paths to the images
    if not include_annotations:
        image_paths = []
        for subfolder in subfolders:
            for filename in os.listdir(images_folder / subfolder):
                image_paths.append(Path(subfolder, filename))
    # else save both the paths to the images and corresponding annotations
    else:
        # initialize lists to store all paths 
        image_paths = []
        annotation_paths = []

        for subfolder in subfolders:
            # define the paths to the image and annotation subfolders
            images_subfolder = images_folder / subfolder
            annotations_subfolder = annotations_folder / subfolder

            # if the images subfolder does not exist, raise an error
            if not images_subfolder.exists():
                raise FileNotFoundError(f'{images_subfolder} does not exist.')
            # if the annotations subfolder does not exist, 
            # add None instead of the path to the annotation
            elif not annotations_subfolder.exists():
                for filename in os.listdir(images_subfolder):
                    image_paths.append(Path(subfolder, filename))
                    annotation_paths.append(None)
            else:
                for image_filename in os.listdir(images_subfolder):
                    # check if there are annotations available for the image
                    search_term = os.path.splitext(image_filename)[0]
                    hits = []
                    for annotation_filename in os.listdir(annotations_subfolder):
                        if search_term in annotation_filename:
                            hits.append(annotation_filename)

                    # add the image path
                    image_paths.append(Path(subfolder, image_filename))

                    # get the path to the latest version of the annotation 
                    # (if applicable) and add it
                    if len(hits) > 0:
                        annotation_filename = natsort.natsorted(hits)[-1]
                        annotation_paths.append(Path(subfolder, annotation_filename))
                    else:
                        annotation_paths.append(None)
    
    # check if the specified name for the dataset sheet already exists
    if (sheets_folder / f'{dataset_name}.xlsx').exists():
        raise FileExistsError('Dataset sheet name already exists.')

    # calculate the number of images in the training and validation set
    N_val = round(len(image_paths)*validation_fraction)
    N_train = len(image_paths)-N_val
    dataset_division = ['val']*N_val+['train']*N_train
    random.shuffle(dataset_division)
    
    # create data dict
    data_dict = {
        'set': dataset_division, 
        'image_paths': [path.as_posix() for path in image_paths]}
    if include_annotations:
        data_dict['annotation_paths'] = [path.as_posix() for path in annotation_paths]

    # create dataframe and save as spreadsheet
    df = pd.DataFrame(data_dict)
    df.to_excel(os.path.join(sheets_folder, f'{dataset_name}.xlsx'), index=False)