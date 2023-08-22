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

import natsort

from annotation_tool import AnnotationTool
from config import annotations_folder, images_folder


# define names        
classes = ['Tissue', 'Pen marking']
subfolder = 'dataset2'

# define settings
rotate_portrait = True
add_layers = True
autosave = True

if __name__ == '__main__':

    # define the paths to the image and annotation subfolders
    images_subfolder = images_folder / subfolder
    annotations_subfolder = annotations_folder / subfolder

    # get all filenames from annotations if the annotation folder already exists
    if not annotations_subfolder.exists():
        annotations_subfolder.mkdir()
        annotation_paths = [] 
    else:
        annotation_paths = list(annotations_subfolder.iterdir())

    # prepare paths to images (and optionally annotations)
    paths = []
    for image_path in images_subfolder.iterdir():

        # check if there are annotations available for the image
        search_term = image_path.stem
        hits = []
        for path in annotation_paths:
            if search_term in path.name:
                hits.append(path)
      
        # get the latest annotation version
        hits = natsort.natsorted(hits)

        # add the path to the latest annotation or None
        # if there are no annotations present
        if not len(hits):
            paths.append((image_path, None))
        else:
            paths.append((image_path, hits[-1]))

    # prepare the annotation tool for annotation
    AnnotationTool(
        input_paths=paths,
        layers=classes,
        output_directory=annotations_subfolder,
        rotate_portrait=rotate_portrait,
        add_layers=add_layers,
        autosave=autosave, 
    )     