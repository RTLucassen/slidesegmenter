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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import natsort
import numpy as np
import pandas as pd
import SimpleITK as sitk

from config import PROJECT_SEED
from config import annotations_folder, images_folder, sheets_folder


def check_pen_marking(path: Optional[Path]) -> bool:
    """
    Load annotation image and determine whether pen markings are present.
    """
    if path is None:
        return False

    annotation = sitk.GetArrayFromImage(sitk.ReadImage(path))
    pen_marking_present = bool(np.sum(annotation[1, ...]) > 0)

    return pen_marking_present


# define settings
subfolders = ['dataset']
all_cases = 'all_cases.xlsx'
dataset_sheet = 'dataset.xlsx'
include_annotations = True
include_pen_marking_presence = True
N_val = 20
N_test = 40

if __name__ == '__main__':

    # check settings condition
    if include_pen_marking_presence and not include_annotations:
        raise ValueError(
            'Pen marking information can only be included if annotations are available.'
        )

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
    if (sheets_folder / dataset_sheet).exists():
        raise FileExistsError('Dataset sheet name already exists.')

    # check if the specified name for the dataset sheet already exists
    if N_test+N_val > len(image_paths):
        raise ValueError('The number of images for validation and testing exceeds'
                        ' the total amount of images available.')
    N_train = len(image_paths)-N_test-N_val

    # load spreadsheet with all case names and patient pseudo IDs for patient-level split
    df_cases = pd.read_excel(sheets_folder / all_cases)

    # match all images to the corresponding patient pseudo IDs
    cases = {}
    for i, image_path in enumerate(image_paths):
        # get the first part of the image name
        case = image_path.name.split('_')[0]
        row = df_cases[df_cases['case'] == case]
        if len(row) != 1:
            raise ValueError('Selected number of cases must be 1.')
        # get the pseudo ID of the corresponding patient
        pseudo_id = list(row['pseudo_id'])[0]

        # add pseudo ID to dictionary
        if pseudo_id not in cases:
            cases[pseudo_id] = {'image_paths': [], 'annotation_paths': []}
    
        # add item to dictionary with pseudo ID as key
        cases[pseudo_id]['image_paths'].append(image_path.as_posix())
        if include_annotations:
            cases[pseudo_id]['annotation_paths'].append(annotation_paths[i].as_posix())

    # create data dict
    data_dict = {'set': [], 'pseudo_id': [], 'image_paths': []}
    if include_annotations:
        data_dict['annotation_paths'] = []
    
    # get all pseudo IDs
    pseudo_ids = list(cases.keys())
    # loop until all images are assigned to a set
    while len(pseudo_ids):
        # randomly select index
        i = random.randint(0, len(pseudo_ids)-1)
        # get cases from selected patient
        pseudo_id = pseudo_ids[i]
        selected_cases = cases[pseudo_id]
        N_images = len(selected_cases['image_paths'])
        
        # loop over sets to assign case
        for N_total, partition in zip([N_test, N_val, N_train], ['test', 'val', 'train']):
            N_assigned = data_dict['set'].count(partition)
            # check if there is still space in the set
            if N_total-N_assigned > 0:
                # check if the set size is not exceeded after assigning the images
                if (N_total-N_assigned)-N_images >= 0:
                    data_dict['set'].extend([partition]*N_images)
                    data_dict['pseudo_id'].extend([pseudo_id]*N_images)
                    data_dict['image_paths'].extend(selected_cases['image_paths'])
                    if include_annotations:
                        data_dict['annotation_paths'].extend(selected_cases['annotation_paths'])
                    # remove the pseudo ID from the list
                    pseudo_ids.remove(pseudo_id)
                    break

    # add information about pen marking presence
    if include_pen_marking_presence:
        paths = [annotations_folder/path for path in data_dict['annotation_paths']]
        with ThreadPoolExecutor() as executor:
            results = executor.map(check_pen_marking, paths)
            data_dict['pen_marking_present'] = results
        
    # save partitions
    df = pd.DataFrame.from_dict(data_dict)
    df.to_excel(sheets_folder/dataset_sheet, index=False)