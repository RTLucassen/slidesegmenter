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
Utility classes for visualization of batches of images
"""

from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


class Index:
    """
    Simple class to keep track of index.
    """

    def __init__(self, initial: int, minimum: int, maximum: int) -> None:
        """
        Initialize index tracker class.

        Args:
            initial: initial index
            minimum: minimum index
            maximum: maximum index
        """
        self.__idx = initial
        self.__minimum = minimum
        self.__maximum = maximum

        # check if the current index is not outside of the range
        if self.__idx < self.__minimum or self.__idx > self.__maximum:
            raise ValueError('Initialized index outside of specified range.')
    
    def current(self) -> int:
        return self.__idx
    
    def minimum(self) -> int:
        return self.__minimum

    def maximum(self) -> int:
        return self.__maximum
    
    def add(self, step: int = 1) -> None:
        self.__idx = min(self.__idx+step, self.__maximum)

    def subtract(self, step: int = 1) -> None:
        self.__idx = max(self.__idx-step, self.__minimum)


class BasicEventTracker:
    """
    Tool for visualization of a batch of images.
    Use the scroll wheel to scroll through the images.
    Press 'SHIFT' to scroll at a higher scrolling speed.
    """
    # define the class attributes
    __initial_speed = 1
    __scroll_speed = 2

    def __init__(
        self, 
        ax: np.ndarray, 
        image_tensor: Union[torch.Tensor, np.ndarray], 
        idx: int = 0,
    ) -> None:
        """ 
        Initialize BasicEventTracker instance.

        Args:
            ax: matplotlib.pyplot figure axis.
            image_tensor: batch of images with the shape: 
                          (instance, channel, height, width).
            idx: slice that is displayed first when the figure is created.
        """
        # check and define instance attribute for image tensor
        if isinstance(image_tensor, np.ndarray):
            image_tensor = torch.from_numpy(image_tensor)
        if isinstance(image_tensor, torch.Tensor):
            self.__image_tensor = torch.permute(image_tensor, (0, 2, 3, 1))
        else:
            raise TypeError('Unexpected type for image_tensor.')
        
        # define instance attributes
        self.__ax = ax
        self.__speed = self.__initial_speed
        self.__image = self.__ax.imshow(self.__image_tensor[idx, ...])

        # initialize objects to track the class and instance index
        self.__idx = Index(idx, 0, self.__image_tensor.shape[0]-1)

        # set the first image
        self.__update()

    def onscroll(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """ 
        Add or subtract 'self.__speed' from the instance index 
        when scrolling up or down respectively.
        Update the image frame afterwards. Do not update the index when 
        scolling up for the last or down for the first image.
        """
        if event.button == 'up':
            # update the index after scolling
            self.__idx.add(self.__speed)
            self.__update()

        elif event.button == 'down': 
            # update the index after scolling
            self.__idx.subtract(self.__speed)
            self.__update()

    def keypress(self, event: matplotlib.backend_bases.KeyEvent) -> None:
        """ 
        Increase the scrolling speed while 'SHIFT' is pressed.
        """
        # for handling multiple keys when pressed at once
        for key in event.key.split('+'): 
            # increases the scrolling speed
            if key == 'shift' and self.__speed == 1:
                self.__speed = self.__scroll_speed

    def keyrelease(self, event: matplotlib.backend_bases.KeyEvent) -> None:
        """ 
        Reset the scrolling speed when 'SHIFT' is released.
        """
        # for handling multiple keys when pressed at once
        for key in event.key.split('+'):
            # decreases scrolling speed
            if key == 'shift' and self.__speed == self.__scroll_speed:
                self.__speed = 1

    def __update(self) -> None:
        """ 
        Update the image.
        """
        # load the new image
        image = self.__image_tensor[self.__idx.current(), ...]
        self.__ax.set_title(
            f'Instance: {self.__idx.current()}/{self.__idx.maximum()}',
        )
        self.__image.set_data(image)
        # update the canvas
        self.__ax.figure.canvas.draw()


class EventTracker:
    """
    Tool for visualization of a batch of images.
    Use the scroll wheel to scroll through the images.
    Press 'SHIFT' to scroll at a higher scrolling speed.
    Press 'CONTROL' to scroll through the channels.
    """
    # define the class attributes
    __initial_speed = 1
    __scroll_speed = 2

    def __init__(
        self, 
        ax: np.ndarray, 
        image_tensor: Union[torch.Tensor, np.ndarray], 
        idx: int = 0,
        class_idx: int = 0, 
        cmap: str = 'gray',
        vmin: float = 0.0,
        vmax: float = 1.0,
    ) -> None:
        """ 
        Initialize EventTracker instance.

        Args:
            ax: matplotlib.pyplot figure axis.
            image_tensor: batch of images with the shape: 
                          (instance, channel, height, width).
            idx: slice that is displayed first when the figure is created.
            class_idx: index for corresponding channel to show.
            cmap: colormap recognized by matplotlib.pyplot module.
            vmin: image intensity used as minimum for the colormap.
            vmax: image intensity used as maximum for the colormap.
        """
        # check and define instance attribute for image tensor
        if isinstance(image_tensor, np.ndarray):
            image_tensor = torch.from_numpy(image_tensor)
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError('Unexpected type for image_tensor.')

        # define instance attributes
        self.__ax = ax
        self.__speed = self.__initial_speed
        self.__image_tensor = image_tensor

        self.__image = self.__ax.imshow(
            self.__image_tensor[idx, class_idx, :, :], 
            vmin=vmin, 
            vmax=vmax, 
            cmap=cmap,
        )
        # initialize objects to track the class and instance index
        self.__idx = Index(idx, 0, self.__image_tensor.shape[0]-1)
        self.__class_idx = Index(class_idx, 0, self.__image_tensor.shape[1]-1)
        self.__selected_idx = self.__idx

        # set the first image
        self.__update()

    def onscroll(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """ 
        Add or subtract 'self.__speed' from the selected index 
        (either instance or class) when scrolling up or down respectively.
        Update the image frame afterwards. Do not update the index when 
        scolling up for the last or down for the first image.
        """
        if event.button == 'up':
            # update the index after scolling
            self.__selected_idx.add(self.__speed)
            self.__update()

        elif event.button == 'down': 
            # update the index after scolling
            self.__selected_idx.subtract(self.__speed)
            self.__update()

    def keypress(self, event: matplotlib.backend_bases.KeyEvent) -> None:
        """ 
        Increase the scrolling speed when 'SHIFT' is pressed.
        """
        # for handling multiple keys when pressed at once
        for key in event.key.split('+'): 
            # increases the scrolling speed
            if key == 'shift' and self.__speed == 1:
                self.__speed = self.__scroll_speed
            # scrolling now influences class index
            if key == 'control' and self.__selected_idx == self.__idx:
                self.__selected_idx = self.__class_idx

    def keyrelease(self, event: matplotlib.backend_bases.KeyEvent) -> None:
        """ 
        Reset the scrolling speed when 'SHIFT' is released.
        """
        # for handling multiple keys when pressed at once
        for key in event.key.split('+'):
            # decreases scrolling speed
            if key == 'shift' and self.__speed == self.__scroll_speed:
                self.__speed = 1
            # scrolling now influences instance index
            if key == 'control' and self.__selected_idx == self.__class_idx:
                self.__selected_idx = self.__idx

    def __update(self) -> None:
        """ 
        Change the image or channel.
        """
        # load the new image
        image = self.__image_tensor[self.__idx.current(), self.__class_idx.current(), :, :]
        self.__ax.set_title(
            (f'Instance: {self.__idx.current()}/{self.__idx.maximum()}, '
            f'Channel: {self.__class_idx.current()}/{self.__class_idx.maximum()}'),
        )
        self.__image.set_data(image)

        # update the canvas
        self.__ax.figure.canvas.draw()


def rgb_image_viewer(
    tensor: Union[torch.Tensor, np.ndarray],
    idx: int = 0,
) -> None:
    """ 
    Tool to view a set of RGB images by scrolling through the slices.

    Args:
        tensor: batch of images with the shape: (instance, channel, height, width).
        idx: slice that is displayed first when the figure is created.
    """
    # create a figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')

    # create tracker object and connect it to the figure
    tracker = BasicEventTracker(ax, tensor, idx)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('key_press_event', tracker.keypress)
    fig.canvas.mpl_connect('key_release_event', tracker.keyrelease)

    plt.show()


def image_viewer(
    tensor: Union[torch.Tensor, np.ndarray],
    idx: int = 0,
    class_idx: int = 0,
    cmap: str = 'gray',
    vmin: float = 0,
    vmax: float = 1,
) -> None:
    """ 
    Tool to view a set of images by scrolling through the slices or channels.

    Args:
        tensor: batch of images with the shape: (instance, channel, height, width).
        idx: slice that is displayed first when the figure is created.
        class_idx: index for corresponding channel to show.
        cmap: colormap recognized by matplotlib.pyplot module.
        vmin: image intensity used as minimum for the colormap.
        vmax: image intensity used as maximum for the colormap.
    """
    # create a figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')

    # create tracker object and connect it to the figure
    tracker = EventTracker(ax, tensor, idx, class_idx, cmap, vmin, vmax)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('key_press_event', tracker.keypress)
    fig.canvas.mpl_connect('key_release_event', tracker.keyrelease)

    plt.show()