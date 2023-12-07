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
Load high magnification whole slide images and save low magnification versions.
"""

import SimpleITK as sitk
from tqdm import tqdm

from slideloader import SlideLoader
from config import images_folder, raw_folder


# define settings
subfolder = 'dataset3'
magnification = 1.25 # (low) magnification value
output_extension = 'png'

if __name__ == '__main__':

    # create a new folder to store the low magnification images
    images_subfolder = images_folder / subfolder
    if not images_subfolder.exists():
        images_subfolder.mkdir()
    else:
        raise FileExistsError

    # define slide handler instance for loading and saving WSIs
    loader = SlideLoader()
    loader.progress_bar = False

    # loop over all WSI paths
    path_dict = {}
    for path in list(raw_folder.iterdir()):
        # get the name
        name = path.name
        for extension in ['.1.dcm', '.2.dcm', '.3.dcm', '.4.dcm', '.ndpi']:
            name = name.replace(extension, '')
        
        # group paths with the same name
        if name not in path_dict:
            path_dict[name] = [str(path)]
        else:
            path_dict[name].append(str(path))

    # loop over all grouped WSI paths
    for name, paths in tqdm(path_dict.items()):
        # load high magnification WSI and return low magnification image
        loader.load_slide(paths)
        image = loader.get_image(magnification)[None, ...]

        # save the low magnification image
        path = images_subfolder / f'{name}.{output_extension}'
        if not path.exists():
            sitk.WriteImage(sitk.GetImageFromArray(image), str(path))
        else:
            raise FileExistsError