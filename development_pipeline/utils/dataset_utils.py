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
Implementation of image generator for neural network training.
"""

import os
import random
from math import ceil
from typing import Any, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import scipy
import SimpleITK as sitk
import torch


class TrainingDataset(torch.utils.data.Dataset):
    """
    Parent dataset class for neural network training.
    """

    pen_transforms = A.Compose([
        A.VerticalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HueSaturationValue(
            p=0.5,
            hue_shift_limit=255, 
            sat_shift_limit=0,          
            val_shift_limit=0, 
        ),
        A.RandomBrightnessContrast(
            p=0.25,
            brightness_limit=0,
            contrast_limit=(0, 1.0),
        ),
        A.Affine(
            p=1.0,
            rotate=(-180, 180), 
            shear=(-15, 15),
            scale=(0.8, 1.25), 
            cval=(255,255,255), 
        ),
        A.GaussianBlur(
            p=0.1,
            sigma_limit=(0.00001, 2),
        )
    ])

    def __init__(
        self, 
        df: pd.core.frame.DataFrame,
        df_pen: Optional[pd.core.frame.DataFrame] = None,
        length: Optional[int] = None,
        shape: Optional[tuple[int, int]] = None,
        max_shape: Optional[tuple[int, int]] = None,
        divisor: Optional[int] = None,
        augmentations: dict[str, Any] = {},
        return_image_name: bool = False,
        return_N_cross_sections: bool = False,
    ) -> None:
        """
        Initialize dataset instance.

        Args:
            df:  Dataframe with dataset information.
            df_pen:  Dataframe with pen marking dataset information.
            length:  Number of items in the dataset (can be set arbitrarily for
                training without specifying epochs).
            shape:  Shape of random crops from the images as (height, width).
            max_shape:  Maximum shape of the images as (height, width).
            divisor:  Input images are padded to be divisible by this number.
            augmentations:  Settings for on-the-fly data augmentation.
            return_image_name:  Indicates whether the image names are returned.
            return_N_cross_sections:  Indicates whether the number of 
                cross-sections are returned.
        """
        # initialize instance attributes
        self.length = length
        self.shape = shape
        self.max_shape = max_shape
        self.divisor = divisor
        self.augmentations = augmentations
        self.return_image_name = return_image_name
        self.return_N_cross_sections = return_N_cross_sections

        # check if the specified maximum shape is not smaller than the specified shape
        if self.max_shape is not None:
            if self.shape is not None:
                if ((self.max_shape[0] < self.shape[0]) or 
                    (self.max_shape[1] < self.shape[1])):
                    raise ValueError(('A specified maximum shape must be larger '
                                     'than the specified shape.'))
            # check if the specified maximum shape is divisible by the specified value
            if self.divisor is not None:
                if ((self.max_shape[0] % self.divisor != 0) or 
                    (self.max_shape[1] % self.divisor != 0)):
                    raise ValueError(('A specified maximum shape must be exactly '
                                      'divisible by the specified divisor.'))

        # check if the specified shape is divisible by the specified value
        if self.shape is not None:
            if self.divisor is not None:
                if ((self.shape[0] % self.divisor != 0) or 
                    (self.shape[1] % self.divisor != 0)):
                    raise ValueError(('A specified shape must be exactly '
                                      'divisible by the specified divisor.'))
            
            # initialize random crop function
            self.random_crop = A.augmentations.crops.transforms.RandomCrop(                    
                width=self.shape[0],
                height=self.shape[1],
            )

        # retrieve paths to images and annotations
        self.image_paths = list(df['image_paths'])
        self.pen_image_paths = [] if df_pen is None else list(df_pen['image_paths'])
        self.pen_images = [sitk.GetArrayFromImage(sitk.ReadImage(p)) for p in self.pen_image_paths]

        # compose augmentation function
        self.transforms = self.get_aug_transforms()
        self.shape_altering_transforms = A.Compose(
            self.transforms['geometric']['shape altering'],
        )
        self.shape_preserving_transforms = A.Compose(
            self.transforms['geometric']['shape preserving'],
        )
        self.color_transforms = A.Compose(
            self.transforms['color']['all'],
        )

    def __len__(self) -> int:
        if self.length is None:
            return len(self.image_paths)
        else:
            return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        """ 
        Returns indexed image-label pair from dataset.

        Args:
            index:  Index for selecting image-annotation pair.

        Returns:
            image:  Image tensor with shape (channels=3, height, width), 
                rescaled in the 0.0-1.0 range and optionally augmented.
        """
        # correct index
        index = index % len(self.image_paths)

        # load image
        image_path = self.image_paths[index]
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))

        # (1) superimpose additional pen markings
        image = self.superimpose_pen_marking(image=image)

        # (2) apply the shape altering geometric augmentation transformations
        image = self.shape_altering_transforms(image=image)['image']

        # add padding to prevent stacked border effects
        padding, center_crop_coords = self.get_padding(image.shape)
        image = np.pad(image, padding, **self.get_padding_mode())
        
        # (3) apply the shape preserving geometric augmentation transformations
        image = self.shape_preserving_transforms(image=image)['image']

        # get the center crop (before the color augmentations for efficiency)
        (top, left), (bottom, right) = center_crop_coords
        image = image[top:bottom, left:right, :]
        
        # get a random crop with the shape that was specified
        if self.shape is not None:
            image = self.random_crop(image=image)['image']

        # (4) apply the color augmentation transformations
        image = self.color_transforms(image=image)['image']

        # change order of dimensions to be (channels, rows, columns),  
        # convert the image intensities to be in the range of 0.0-1.0,      
        # and from numpy ndarray to torch tensor   
        image = np.transpose(image, (2,0,1))/255
        image = torch.from_numpy(image).to(torch.float)

        return image

    def get_padding_mode(self) -> dict[str, Any]:
        """
        Parses specified augmentation settings for padding configuration and
        returns a padding mode based on the probabilities for selection.
        """
        if 'Padding' in self.augmentations:
            config = self.augmentations['Padding']
            # check if all probabilities add up to 1.
            if sum([mode['p'] for mode in config]) != 1:
                raise ValueError('The probabilities p for all padding modes '
                                 'must add up to 1.')
            # select a padding mode based on the probabilities for selection
            i = random.random()
            p = 0
            for mode in config:
                if i <= p+mode['p']:
                    if mode['mode'] in ['edge', 'reflect', 'symmetric', 'wrap']:
                        return {'mode': mode['mode']}
                    elif mode['mode'] == 'constant':
                        return {'mode': mode['mode'], 
                                'constant_values': mode['value']}
                    else:
                        name = mode['mode']
                        raise ValueError(f'Invalid padding mode: {name}.')
                else:
                    p += mode['p']
        else:
            return {'mode': 'reflect'} # default

    def get_padding(self, image_shape: tuple[int, int]) -> tuple:
        """
        Args:
            image_shape:  Shape of the image as (height, width).

        Returns:
            padding:  Tuple with amount of padding to add to each image dimension.
            center_crop_coords:  Top left and bottom right coordinate for 
                center cropping.
        """
        # determine the image shape for calculating the padding
        if self.shape is not None:
            shape = (
                max(self.shape[0], image_shape[0]),
                max(self.shape[1], image_shape[1]),
            )
        elif self.divisor is not None:
            shape = (
                ceil(image_shape[0]/self.divisor)*self.divisor,
                ceil(image_shape[1]/self.divisor)*self.divisor,
            )
        else:
            shape = tuple(image_shape[:2])

        if self.max_shape is not None:
            shape = (
                min(self.max_shape[0], shape[0]),
                min(self.max_shape[1], shape[1]),
            )

        # calculate the radius of the circle that can enclose the image shape
        radius = ((shape[0]/2)**2+(shape[1]/2)**2)**0.5
        if 'Affine' in self.augmentations:    
            if self.augmentations['Affine']['p'] > 0:
                # account for scale
                scale = self.augmentations['Affine']['scale']
                if isinstance(scale, (float, int)):
                    radius *= scale
                elif isinstance(scale, (list, tuple)):
                    radius *= 1/min(scale)
                elif scale is not None:
                    raise ValueError('Invalid argument for scale.')  

        # determine the correction factor for scaling
        max_translation = [0, 0]
        if 'Affine' in self.augmentations:
            if self.augmentations['Affine']['p'] > 0:
                # check input argument for translation
                translation = self.augmentations['Affine']['translate_px']
                if not isinstance(translation, dict):
                    if (not isinstance(translation, (int, float, list, tuple))
                        and translation is not None):
                        raise ValueError('Invalid argument for translation.') 
                    else:
                        translation = {'x': translation, 'y': translation}
                # determine the maximum translation
                for i, axis in enumerate(['y', 'x']):
                    if isinstance(translation[axis], (float, int)):
                        max_translation[i] = translation[axis]
                    elif isinstance(translation[axis], (list, tuple)):
                        max_translation[i] = max([abs(t) for t in translation[axis]])
                    else:
                        raise ValueError('Invalid argument for translation.') 
                         
        # get the total height and width of the image with padded 
        # to prevent border effects in the center cropped region
        total_height = (radius+max_translation[0])*2
        total_width = (radius+max_translation[1])*2

        # calculate the amount of padding for each dimension
        height_pad = max(0, round((total_height-image_shape[0])/2))
        width_pad = max(0, round((total_width-image_shape[1])/2))
        padding = ((height_pad, height_pad), (width_pad, width_pad), (0,0))

        # calculate the top left and bottom right coordinates for center cropping
        top = round((total_height-shape[0])/2)
        left = round((total_width-shape[1])/2)
        center_crop_coords = [(top, left), (top+shape[0], left+shape[1])]

        return padding, center_crop_coords

    def superimpose_pen_marking(self, image: np.ndarray) -> np.ndarray:
        """
        Read a pen marking image, augment it, and superimpose it on the image.

        Args:
            image: Image tensor with shape (channels=3, height, width).

        Returns:
            image:  Image tensor with shape (channels=3, height, width) 
                and superimposed pen marking.
        """
        # check if pen marking augmentation was specified
        if 'PenMarkings' in self.augmentations:
            config = self.augmentations['PenMarkings']
            # check if the probability is larger than zero
            if not (0.0 <= config['p'] <= 1.0):
                raise ValueError('The probability p must be between 0.0 and 1.0.')
            # check if there are any pen marking images available
            if not len(self.pen_image_paths):
                return image
        else:
            return image
        
        # determine if pen marking augmentation should be applied
        if random.random() > config['p']:
            return image
        
        # repeat loading and superimposing pen markings on the image N times
        if isinstance(config['N'], (tuple, list)):
            if len(config['N']) != 2:
                raise ValueError('Invalid number of items (expected 2).')
            times = random.randint(*config['N'])
        elif isinstance(config['N'], int):
            times = config['N']
        else:
            raise ValueError('The number of pen markings N must be a single '
                             'integer or a 2-tuple with the minimum and maximum '
                             'amount for random sampling.')
        for _ in range(times):
            # load pen marking image
            pen_image = self.pen_images[random.randint(0, len(self.pen_images)-1)]

            # add padding to make the pen marking image square
            pen_height, pen_width, _ = pen_image.shape
            diff = pen_height-pen_width
            if diff > 0:
                pen_image = np.pad(
                    array=pen_image, 
                    pad_width=((0,0), (int(diff/2), ceil(diff/2)), (0,0)), 
                    constant_values=255,
                )
                pen_width = pen_height
            elif diff < 0:
                pen_image = np.pad(
                    array=pen_image, 
                    pad_width=((int(abs(diff)/2), ceil(abs(diff)/2)), (0,0), (0,0)), 
                    constant_values=255,
                )
                pen_height = pen_width

            # apply augmentations to pen marking image
            pen_image = self.pen_transforms(image=pen_image)['image']

            # determine a random position
            height, width, _ = image.shape
            y = random.randint(0, height-1)
            x = random.randint(0, width-1)

            # determine the coordinates
            min_y = max(y-int(pen_height/2), 0)
            max_y = min(y+ceil(pen_height/2), height)
            min_x = max(x-int(pen_width/2), 0)
            max_x = min(x+ceil(pen_width/2), width)

            diff_y = pen_height-(max_y-min_y)
            diff_x = pen_width-(max_x-min_x)

            # multiply the image with the pen marking
            image[min_y:max_y, min_x:max_x, :] = (
                pen_image[int(diff_y/2):pen_height-ceil(diff_y/2), 
                        int(diff_x/2):pen_width-ceil(diff_x/2), :] / 255
                * image[min_y:max_y, min_x:max_x, :]
            )

        return image
    
    def get_aug_transforms(self) -> dict[str, dict]:
        """
        Parses specified augmentation settings and configures augmentation transforms.

        Returns:
            transforms:  Transform functions based on specified configuration.
        """
        transforms = {
            'geometric': {
                'shape altering': [],
                'shape preserving': [],
            },
            'color': {
                'all': [],
            },
        }
        # initialize transforms if specified
        if 'RandomRotate90' in self.augmentations:
            transforms['geometric']['shape altering'].append(
                A.RandomRotate90(
                    **self.augmentations['RandomRotate90'],
                )
            )
        if 'Affine' in self.augmentations:
            transforms['geometric']['shape preserving'].append(
                A.Affine(
                    **self.augmentations['Affine'],
                    keep_ratio=True, 
                    interpolation=1, 
                    mode=cv2.BORDER_REFLECT,
                )
            )
        if 'HorizontalFlip' in self.augmentations:
            transforms['geometric']['shape preserving'].append(
                A.HorizontalFlip(
                    **self.augmentations['HorizontalFlip'], 
                )
            )
        if 'VerticalFlip' in self.augmentations:
            transforms['geometric']['shape preserving'].append(
                A.VerticalFlip(
                    **self.augmentations['VerticalFlip'],
                )
            )  
        if 'HueSaturationValue' in self.augmentations:
            transforms['color']['all'].append(
                A.HueSaturationValue(
                    **self.augmentations['HueSaturationValue'],
                )
            )
        if 'RandomBrightnessContrast' in self.augmentations:
            transforms['color']['all'].append(
                A.RandomBrightnessContrast(
                    **self.augmentations['RandomBrightnessContrast'],
                )
            )
        if 'RandomGamma' in self.augmentations:
            transforms['color']['all'].append(
                A.RandomGamma(
                    **self.augmentations['RandomGamma'],
                )
            )
        if 'GaussNoise' in self.augmentations:
            transforms['color']['all'].append(
                A.GaussNoise(
                    **self.augmentations['GaussNoise'],
                )
            )
        if 'GaussianBlur' in self.augmentations:
            transforms['color']['all'].append(
                A.GaussianBlur(
                    **self.augmentations['GaussianBlur'],
                    blur_limit=(15, 15), 
                )
            )
        if 'JPEGCompression' in self.augmentations:
            transforms['color']['all'].append(
                JPEGCompression(
                    **self.augmentations['JPEGCompression'],
                )
            )  

        return transforms


def compress(image: np.ndarray, quality: int) -> np.ndarray:
    """
    Apply JPEG compression to image.

    Args:
        image:  Image to be compressed.
        quality:  Value between 0 and 100 to indicate the compression quality.
    
    Returns:
        compressed_image:  Image after compression.
    """
    compression_parameters = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_image = cv2.imencode('.jpg', image, compression_parameters)
    compressed_image = cv2.imdecode(encoded_image, 1)

    return compressed_image


class JPEGCompression(A.ImageOnlyTransform):
    """
    Apply JPEG compression to image.

    Args:
        quality_limit (int, int): Range of compression quality. 
            Default: (50, 100).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        any
    """
    def __init__(self, quality_limit: tuple[int, int] = (50, 100), 
                 always_apply: bool = False, p: float = 0.5) -> None:
        super(JPEGCompression, self).__init__(always_apply, p)
        # initialize instance attributes
        self.quality_limit = quality_limit

    def apply(self, image, quality=50, **params):
        return compress(image, quality)

    def get_params(self):
        return {"quality": random.randint(*self.quality_limit)}

    def get_transform_init_args_names(self):
        return ("quality",)


class SupervisedTrainingDataset(TrainingDataset):
    """
    Dataset class for supervised neural network training.
    """

    def __init__(
        self, 
        df: pd.core.frame.DataFrame,
        df_pen: Optional[pd.core.frame.DataFrame] = None,
        length: Optional[int] = None,
        shape: Optional[tuple[int, int]] = None,
        max_shape: Optional[tuple[int, int]] = None,
        divisor: Optional[int] = None,
        augmentations: dict[str, Any] = {},
        return_image_name: bool = False,
        return_N_cross_sections: bool = False,
    ) -> None:
        """
        Initialize dataset instance.

        Args:
            df:  Dataframe with dataset info.
            df_pen:  Dataframe with pen marking dataset information.
            length:  Number of items in the dataset (can be set arbitrarily for
                training without specifying epochs).
            shape:  Shape of random crops from images as (width, height).
            max_shape:  Maximum shape of the images as (height, width).
            divisor:  Input images are padded to be divisible by this number.
            augmentations:  Settings for on-the-fly data augmentation.
            return_image_name:  Indicates whether the image names are returned.
            return_N_cross_sections:  Indicates whether the number of 
                cross-sections are returned.
        """
        super().__init__(df, df_pen, length, shape, max_shape, divisor, augmentations, 
                         return_image_name, return_N_cross_sections)
        
        # retrieve paths to annotations
        self.annotation_paths = list(df['annotation_paths'])
        self.pen_annotation_paths = [] if df_pen is None else list(df_pen['annotation_paths'])
        self.pen_annotations = [sitk.GetArrayFromImage(sitk.ReadImage(p)) for p in self.pen_annotation_paths]

        # compose augmentation function for specific regions
        self.tissue_color_transforms = A.Compose(
            self.transforms['color']['tissue'],
        )
        self.non_tissue_color_transforms = A.Compose(
            self.transforms['color']['non-tissue'],
        )
        self.pen_color_transforms = A.Compose(
            self.transforms['color']['pen'],
        )
        self.background_color_transforms = A.Compose(
            self.transforms['color']['background'],
        )


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """ 
        Returns indexed image-label pair from dataset.

        Args:
            index:  Index for selecting image-annotation pair.

        Returns:
            image:  Image tensor with shape (channels=3, rows, columns), 
                rescaled in the 0.0-1.0 range and optionally augmented.
            label:  Label tensor with shape (channels, rows, columns),
                rescaled in the 0.0-1.0 range and optionally augmented.
        """
        # correct index
        index = index % len(self.image_paths)

        # load image
        image_path = self.image_paths[index]
        image_name = os.path.split(image_path)[-1]
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))

        # load annotation
        annotation_path = self.annotation_paths[index]
        annotation = sitk.GetArrayFromImage(sitk.ReadImage(annotation_path))
        annotation = np.transpose(annotation, (1,2,0))
        N_cross_sections = annotation.shape[-1]-2

        # (1) superimpose additional pen markings
        image, annotation = self.superimpose_pen_marking(image=image, mask=annotation)

        # (2) apply the shape altering geometric augmentation transformations
        transformed = self.shape_altering_transforms(image=image, mask=annotation)
        image = transformed['image']
        annotation = transformed['mask']

        # add padding to prevent stacked border effects
        padding, center_crop_coords = self.get_padding(image.shape)
        padding_mode = self.get_padding_mode()
        image = np.pad(image, padding, **padding_mode)
        # padding with constant values should always result in zeros in the mask
        if 'constant_values' in padding_mode:
            padding_mode['constant_values'] = 0
        annotation = np.pad(annotation, padding, **padding_mode)

        # (3) apply the shape preserving geometric augmentation transformations
        transformed = self.shape_preserving_transforms(image=image, mask=annotation)
        image = transformed['image']
        annotation = transformed['mask']
        
        # get the center crop (before the color augmentations for efficiency)
        (top, left), (bottom, right) = center_crop_coords
        image = image[top:bottom, left:right, :]
        annotation = annotation[top:bottom, left:right, :]

        # get a random crop with the shape that was specified
        if self.shape is not None:
            cropped = self.random_crop(image=image, mask=annotation)
            image = cropped['image']
            annotation = cropped['mask']

        # get region images
        tissue_region = np.where(annotation[..., 0:1] > 0, 1, 0)
        pen_region = np.where(annotation[..., 1:2] > 0, 1, 0)
        background = np.where(np.sum(annotation[..., 0:2], axis=2, keepdims=True) > 0, 0, 1)

        # (4) apply color augmentation transformations to all image regions
        image = self.color_transforms(image=image)['image']
        # (5) apply color augmentation transformations to the tissue regions
        transformed = self.tissue_color_transforms(image=image)
        image = (transformed['image']*tissue_region 
                 + image*(1-tissue_region)).astype(np.uint8)
        # (6) apply color augmentation transformations to the pen and background regions
        transformed = self.non_tissue_color_transforms(image=image)
        image = (transformed['image']*(pen_region+background) 
                 + image*(1-(pen_region+background))).astype(np.uint8)
        # (7) apply color augmentation transformations to the pen marking regions
        transformed = self.pen_color_transforms(image=image)
        image = (transformed['image']*pen_region 
                 + image*(1-pen_region)).astype(np.uint8)
        # (8) apply color augmentation transformations to the background regions
        transformed = self.background_color_transforms(image=image)
        image = (transformed['image']*background 
                 + image*(1-background)).astype(np.uint8)
        
        # change order of dimensions to be (channels, rows, columns) and 
        # convert the image intensities to be in the range of 0.0-1.0    
        image = np.transpose(image, (2,0,1))/255

        # convert annotation to label
        annotation = np.transpose(annotation, (2,0,1))
        label = create_label(annotation)

        # convert the image and label from numpy ndarray to torch tensor 
        image = torch.from_numpy(image).to(torch.float)
        label = torch.from_numpy(label).to(torch.float)

        output = [image, label]
        if self.return_N_cross_sections:
            output = [N_cross_sections]+output
        if self.return_image_name:
            output = [image_name]+output
        
        return tuple(output)

    def get_aug_transforms(self) -> list:
        """
        Parses specified augmentation settings and configures augmentation transforms.

        Returns:
            transforms:  Transform functions based on specified configuration.
        """
        transforms = super().get_aug_transforms()
        
        # check settings for region-specific augmentations
        for region in ['tissue', 'pen', 'background', 'non-tissue']:
            transforms['color'][region] = []
            if f'HueSaturationValue {region}' in self.augmentations:
                transforms['color'][region].append(
                    A.HueSaturationValue(
                        **self.augmentations[f'HueSaturationValue {region}'],
                    )
                )
            if f'RandomBrightnessContrast {region}' in self.augmentations:
                transforms['color'][region].append(
                    A.RandomBrightnessContrast(
                        **self.augmentations[f'RandomBrightnessContrast {region}'],
                    )
                )  
        
        return transforms
    
    def superimpose_pen_marking(self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Read a pen marking image, augment it, and superimpose it on the image.

        Args:
            image: Image tensor with shape (channels=3, height, width).
            mask: Mask tensor with shape (channels, rows, columns).

        Returns:
            image:  Image tensor with shape (channels=3, height, width) and 
                superimposed pen markings.
            mask: Mask tensor with shape (channels, rows, columns)
                and superimposed pen markings.
        """
        # check if pen marking augmentation was specified
        if 'PenMarkings' in self.augmentations:
            config = self.augmentations['PenMarkings']
            # check if the probability is larger than zero
            if not (0.0 <= config['p'] <= 1.0):
                raise ValueError('The probability p must be between 0.0 and 1.0.')
            # check if there are any pen marking images available
            if not len(self.pen_image_paths):
                return image, mask
        else:
            return image, mask
        
        # determine if pen marking augmentation should be applied
        if random.random() > config['p']:
            return image, mask
        
        # repeat loading and superimposing pen markings on the image N times
        if isinstance(config['N'], (tuple, list)):
            if len(config['N']) != 2:
                raise ValueError('Invalid number of items (expected 2).')
            times = random.randint(*config['N'])
        elif isinstance(config['N'], int):
            times = config['N']
        else:
            print(type(config['N']))
            raise ValueError('The number of pen markings N must be a single '
                             'integer or a 2-tuple with the minimum and maximum '
                             'amount for random sampling.')

        # repeat loading and superimposing pen markings on the image N times
        for _ in range(times):
            # load pen marking image and annotation
            index = random.randint(0, len(self.pen_image_paths)-1)
            pen_image = self.pen_images[index]
            pen_annotation = self.pen_annotations[index]

            # get the height, width, and the distance between them
            pen_height, pen_width, _ = pen_image.shape
            diff = pen_height-pen_width

            # add padding to make the pen marking image square            
            if diff != 0:
                if diff > 0:
                    padding = ((0,0), (int(diff/2), ceil(diff/2)), (0,0))
                    pen_width = pen_height
                elif diff < 0:
                    padding = ((int(abs(diff)/2), ceil(abs(diff)/2)), (0,0), (0,0))
                    pen_height = pen_width
            
                pen_image = np.pad(pen_image, padding, constant_values=255)
                pen_annotation = np.pad(pen_annotation, padding[:2], constant_values=0)
                               
            # apply augmentations to pen marking image and annotation
            transformed = self.pen_transforms(image=pen_image, mask=pen_annotation)
            pen_image = transformed['image']
            pen_annotation = transformed['mask']

            # determine a random position
            height, width, _ = image.shape
            y = random.randint(0, height-1)
            x = random.randint(0, width-1)

            # determine the coordinates
            min_y = max(y-int(pen_height/2), 0)
            max_y = min(y+ceil(pen_height/2), height)
            min_x = max(x-int(pen_width/2), 0)
            max_x = min(x+ceil(pen_width/2), width)

            diff_y = pen_height-(max_y-min_y)
            diff_x = pen_width-(max_x-min_x)

            # multiply the image with the pen marking
            image[min_y:max_y, min_x:max_x, :] = (
                pen_image[int(diff_y/2):pen_height-ceil(diff_y/2), 
                          int(diff_x/2):pen_width-ceil(diff_x/2), :] 
                / 255 * image[min_y:max_y, min_x:max_x, :]
            )
            # combine the pen marking annotation with the existing mask 
            mask[min_y:max_y, min_x:max_x, 1] = np.where(
                ((pen_annotation[int(diff_y/2):pen_height-ceil(diff_y/2), 
                                 int(diff_x/2):pen_width-ceil(diff_x/2)] /255)
                + (mask[min_y:max_y, min_x:max_x, 1]/255)) > 0, 255, 0,
            )
                 
        return image, mask


class InferenceDataset(torch.utils.data.Dataset):
    """
    Dataset class for neural network inference.
    """

    def __init__(
        self, 
        df: pd.core.frame.DataFrame,
    ) -> None:
        """
        Initialize dataset instance.

        Args:
            df:  Dataframe with dataset info.
        """
        # retrieve paths to images and annotations
        self.image_paths = list(df['image_paths'])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """ 
        Returns indexed image from dataset.

        Args:
            index:  Index for selecting image.

        Returns:
            image:  Image tensor with shape (channels=3, rows, columns), 
                rescaled in the 0.0-1.0 range and optionally augmented.
        """
        # load image
        image_path = self.image_paths[index]
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))

        # change order of dimensions to be (channels, rows, columns) and 
        # convert the image intensities to be in the range of 0.0-1.0    
        image = np.transpose(image, (2,0,1))/255

        # convert the image and label from numpy ndarray to torch tensor 
        image = torch.from_numpy(image).to(torch.float)

        return image


def seed_worker(worker_id):
    """ Seed randomness of worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_label(annotation: np.ndarray, divisor: float = 100) -> np.ndarray:
    """
    create label from the annotation array.

    Args:
        annotation:  Annotation array consisting of mask and separate regions.
        divisor:  Input images are cropped to be divisible by the specified number.
    
    returns:
        label:  Label array consisting of binary mask, horizontal map, 
            and vertical map.
    """
    # convert annotations to zero and one
    annotation = np.where(annotation > 0.5, 1, 0)
    
    # split the mask and the regions
    tissue_mask = annotation[0:1, ...]
    marking_mask = annotation[1:2, ...]
    regions = annotation[2:, ...]

    # initialize empty arrays to add horizontal 
    # and vertical maps for regions to
    horizontal_regions = np.zeros_like(tissue_mask, dtype=np.float32)
    vertical_regions = np.zeros_like(tissue_mask, dtype=np.float32)

    # create horizontal and vertical grid
    horizontal_map, vertical_map = np.meshgrid(
        np.linspace(0, tissue_mask.shape[2]-1, tissue_mask.shape[2]),
        np.linspace(0, tissue_mask.shape[1]-1, tissue_mask.shape[1]),
    )
    # loop over regions
    for i in range(regions.shape[0]):
        region = regions[i, ...]
        if np.sum(region) > 0:
            # find number of connected components 
            # (in case of mirroring or wrapping)
            separated, N_subregions = scipy.ndimage.label(region)
            # loop over subregions
            for j in range(1, N_subregions+1):
                # get the masked subregion
                subregion = np.where(separated == j, 1, 0)
                masked_subregion = (tissue_mask*subregion)[0, ...]           
                if np.sum(masked_subregion) > 0:
                    # create the absolute horizontal and vertical distance maps 
                    # for the region
                    horizontal_region = masked_subregion*horizontal_map
                    vertical_region = masked_subregion*vertical_map
                    
                    # calculate the centroid of the masked subregion
                    centroid = [
                        np.sum(horizontal_region)/np.sum(masked_subregion),
                        np.sum(vertical_region)/np.sum(masked_subregion)
                    ]
                    # convert to relative distance maps by subtracting the centroid
                    horizontal_region = (horizontal_region-centroid[0])*masked_subregion
                    vertical_region = (vertical_region-centroid[1])*masked_subregion

                    # add the distance maps to the global region
                    horizontal_regions += horizontal_region[None, ...]
                    vertical_regions += vertical_region[None, ...]
    
    # combine mask and maps
    label = np.concatenate(
        [tissue_mask, marking_mask, horizontal_regions/divisor, vertical_regions/divisor],
        axis=0,
    )
    
    return label