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
Configure settings and create project subfolders.
"""
 
import platform
from pathlib import Path

# define project seed
PROJECT_SEED = 222

# define paths
if platform.system() == 'Windows':
    project_root = r'C:\Users\rlucasse\Documents\projects\tissue_segmentation'
    openslide_path = r'C:\Users\rlucasse\Documents\openslide-win64-20221111\bin'
elif platform.system() == 'Linux':
    project_root = r'/hpc/dla_patho/ruben/projects/tissue_segmentation'
    openslide_path = False
else:
    print(f'Platform {platform.system()} was not configured.')

if project_root is None:
    raise ValueError('First specify the project root.')
elif openslide_path is None:
    raise ValueError('First specify the path to the openslide binaries.')

# convert paths to path objects
if isinstance(project_root, str):
    project_root = Path(project_root)  
if isinstance(openslide_path, str):
    openslide_path = Path(openslide_path)

# create a list to collect the paths to the project subfolders
folders = []

# the raw folder is where the original clips which remain untouched are stored
raw_folder = project_root / 'raw'
folders.append(raw_folder)

# the intermediate folder is where the processed data and models are stored
intermediate_folder = project_root / 'intermediate'
folders.append(intermediate_folder)

# the data folder is where the images, arrays, and sheets are stored
data_folder = intermediate_folder / 'data'
folders.append(data_folder)

# the images folder is where (processed) images and label maps are stored
images_folder = data_folder / 'images'
folders.append(images_folder)

# the annotations folder is where the annotation data is stored
annotations_folder = data_folder / 'annotations'
folders.append(annotations_folder)

# the predictions folder is where the model predictions are saved
predictions_folder = data_folder / 'predictions'
folders.append(predictions_folder)

# the annotations folder is where the annotation data is stored
sheets_folder = data_folder / 'sheets'
folders.append(sheets_folder)

# the models folder is where the trained deep learning models are stored
models_folder = intermediate_folder / 'models'
folders.append(models_folder)

if __name__ == '__main__':
    
    # create the folders if they do not exist yet
    for folder in folders:
        if folder.exists():
            print(f'{folder} already exists.')
        else:
            folder.mkdir()    