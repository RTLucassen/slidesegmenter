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
from config import raw_folder, images_folder

# define settings
subfolder = 'other'
magnification = 1.25 # (low) magnification value
extension = 'png'

#TODO: Remove str() with pathlib objects for slideloader once this is possible

if __name__ == '__main__':

    # create a new folder to store the low magnification images
    images_subfolder = images_folder / subfolder
    images_subfolder.mkdir()

    # define slide handler instance for loading and saving WSIs
    slide_handler = SlideLoader()
    slide_handler.progress_bar = False

    # loop over all WSIs
    for path in tqdm(list(raw_folder.iterdir())):
        # load high magnification WSI and return low magnification image
        slide_handler.load_slide(str(path))
        image = slide_handler.get_image(magnification)[None, ...]

        # save the low magnification image
        path = images_subfolder / f'{path.stem}.{extension}'
        sitk.WriteImage(sitk.GetImageFromArray(image), str(path))