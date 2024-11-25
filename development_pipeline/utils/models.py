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
Implementation of network architecture (modified U-Net) in Pytorch.
"""

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torchinfo import summary


class Block(nn.Module):

    def __init__(
        self, 
        input_channels: int, 
        output_channels: int, 
        activation: str,
        normalization: str, 
        residual_connection: bool,
    ) -> None:
        """
        Implementation of a block of layers.

        Args:
            input_channels:  Number of channels of the input tensor.
            output_channels:  Number of channels of the output tensor.
            activation:  Activation function to non-linearly transform feature maps.
            normalization:  Type of normalization layer for feature maps.
            residual_connection:  Indicates whether a residual connection is added.
        """
        super().__init__()

        # define activation function
        if activation == 'relu':
            self.activation = nn.ReLU
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU
        else:
            raise ValueError('Invalid argument for activation function.')
        
        # define normalization layer
        if normalization is None:
            self.normalization = None
        elif 'batch' in normalization:
            self.normalization = nn.BatchNorm2d
        elif 'instance' in normalization:
            self.normalization = nn.InstanceNorm2d
        else:
            raise ValueError('Invalid argument for normalization layer.')
        
        # define residual connections attribute
        self.residual_connection = residual_connection
        
        # define layers
        self.conv1 = nn.Conv2d(
            input_channels, 
            output_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            padding_mode='zeros',
            bias = False if self.normalization is not None else True,
        )
        if self.normalization is not None:
            self.norm1 = self.normalization(output_channels)
        self.act1 = self.activation(inplace=True)

        self.conv2 = nn.Conv2d(
            output_channels, 
            output_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            padding_mode='zeros',
            bias = False if self.normalization is not None else True,
        )
        if self.normalization is not None:
            self.norm2 = self.normalization(output_channels)
        self.act2 = self.activation(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  Input tensor
        Returns:
            x:  Tensor after operations
        """
        x1 = self.conv1(x)
        x1 = self.norm1(x1) if self.normalization is not None else x1
        x1 = self.act1(x1)
        
        x2 = self.conv2(x1)
        x2 = self.norm2(x2) if self.normalization is not None else x2
        x2 = self.act2(x2)
        x = x1+x2 if self.residual_connection else x2

        return x


class Down(Block):

    def __init__(self, 
        input_channels: int, 
        output_channels: int, 
        activation: str, 
        normalization: str,
        downsample_factor: int,
        downsample_method: str,         
        residual_connection: bool,
    ) -> None:
        """
        Implementation of a block of layers starting with a downsampling operation.

        Args:
            input_channels:  Number of channels of the input tensor.
            output_channels:  Number of channels of the output tensor.
            activation:  Activation function to non-linearly transform feature maps.
            normalization:  Type of normalization layer for feature maps.
            downsample_factor:  Downsampling factor used to reduce the spatial size.
            downsample_method:  Operation used for downsampling the feature maps.
            residual_connection:  Indicates whether a residual connection is added.
        """
        super().__init__(input_channels, output_channels, activation, 
                         normalization, residual_connection)

        # define downsampling layer as strided convolution
        if downsample_method == 'max_pool':
            self.downsample = nn.MaxPool2d(
                kernel_size=downsample_factor, 
                stride=downsample_factor,
            )
        elif downsample_method == 'strided_conv':
            self.downsample = nn.Conv2d(
                input_channels, 
                input_channels, 
                kernel_size=downsample_factor, 
                stride=downsample_factor, 
                padding=0,
            )
        elif downsample_method == 'interpolate':
            self.downsample = lambda x: nn.functional.interpolate(
                x, 
                scale_factor=1/downsample_factor,
                mode='nearest',
            )
        else:
            raise ValueError('Invalid argument for downsample method.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  Input tensor.
        Returns:
            x:  Tensor after operations.
        """
        x = self.downsample(x)
        x1 = self.conv1(x)
        x1 = self.norm1(x1) if self.normalization is not None else x1
        x1 = self.act1(x1)
        
        x2 = self.conv2(x1)
        x2 = self.norm2(x2) if self.normalization is not None else x2
        x2 = self.act2(x2)
        x = x1+x2 if self.residual_connection else x2

        return x


class Up(Block):

    def __init__(self, 
        input_channels: int, 
        shortcut_channels: int,
        output_channels: int, 
        activation: str, 
        normalization: str,
        upsample_factor: int,
        upsample_method: str, 
        residual_connection: bool,
    ) -> None:
        """
        Implementation of a block of layers starting with a upsampling operation.

        Args:
            input_channels:  Number of channels of the input tensor.
            shortcut_channels:  Number of channels of the shortcut tensor.
            output_channels:  Number of channels of the output tensor.
            activation:  Activation function to non-linearly transform feature maps.
            normalization:  Type of normalization layer for feature maps.
            upsample_factor:  Upsampling factor used for increasing the spatial size.
            upsample_method:  Operation used for upsampling the feature maps.
            residual_connection:  Indicates whether a residual connection is added.
        """
        super().__init__(input_channels+shortcut_channels, output_channels, 
                         activation, normalization, residual_connection)

        # define additional layers
        if upsample_method == 'transposed_conv':
            self.upsample = nn.ConvTranspose2d(
                input_channels, 
                input_channels, 
                kernel_size=upsample_factor, 
                stride=upsample_factor,
            )
        elif upsample_method == 'interpolate':
            self.upsample = nn.Upsample(
                scale_factor=upsample_factor, 
                mode='nearest',
            )    
        else:
            raise ValueError('Invalid argument for upsample method.')       

    def forward(self, x_down: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_down:  Input tensor from downsampling path though shortcut.
            x:  Input tensor from upsampling path.
        Returns:
            x:  Tensor after operations.
        """
        x = self.upsample(x)
        x = torch.cat([x_down, x], dim=1)
        x1 = self.conv1(x)
        x1 = self.norm1(x1) if self.normalization is not None else x1
        x1 = self.act1(x1)
        
        x2 = self.conv2(x1)
        x2 = self.norm2(x2) if self.normalization is not None else x2
        x2 = self.act2(x2)
        x = x1+x2 if self.residual_connection else x2

        return x


class UNet(nn.Module):

    def __init__(
        self,
        input_channels: int,
        N_classes: int,
        filters: int = 8,
        activation: str = 'relu',
        normalization: str = 'instance',
        downsample_method: str = 'max_pool',
        downsample_factors: list = [2, 2, 2, 2, 2],
        upsample_method: str = 'interpolate',
        residual_connection: bool = False,
        weight_init: str = 'kaiming_normal',
    ) -> None:
        """
        Implementation of U-Net.

        Args:
            input_channels:  Number of channels of the input tensor.
            N_classes:  Number of channels of the output tensor.
            filters:  Number of filters used in the first convolutional layer. 
                Each consecutive layer in the encoder path uses twice as many filters.
                Each consecutive layer in the decoder path uses half as many filters.
            activation:  Activation function to non-linearly transform feature maps.
            normalization:  Type of normalization layer for feature maps.
            downsample_method:  Method to downsample feature maps.  
            downsample_factors:  Factors for downsampling the feature maps.
            upsample_method:  Method to upsample feature maps.
            residual_connection:  Indicates whether a residual connection is added.
            weight_init:  Indicates which weight initialization method should be used.
        """
        super().__init__()

        # define hyperparameters as instance attributes
        self.filters = filters
        self.activation = activation
        self.normalization = normalization
        self.downsample_method = downsample_method
        self.downsample_factors = downsample_factors
        self.upsample_method = upsample_method
        self.residual_connection = residual_connection
        self.weight_init = weight_init

        #define network layers
        self.layers = nn.ModuleDict({
            'block': Block(input_channels, int(self.filters), self.activation, self.normalization, self.residual_connection),
            'down1': Down(int(  1 * self.filters), int(  2 * self.filters),                          self.activation, self.normalization, self.downsample_factors[0], self.downsample_method, self.residual_connection),
            'down2': Down(int(  2 * self.filters), int(  4 * self.filters),                          self.activation, self.normalization, self.downsample_factors[1], self.downsample_method, self.residual_connection),
            'down3': Down(int(  4 * self.filters), int(  8 * self.filters),                          self.activation, self.normalization, self.downsample_factors[2], self.downsample_method, self.residual_connection),
            'down4': Down(int(  8 * self.filters), int( 16 * self.filters),                          self.activation, self.normalization, self.downsample_factors[3], self.downsample_method, self.residual_connection),
            'down5': Down(int( 16 * self.filters), int( 16 * self.filters),                          self.activation, None,               self.downsample_factors[4], self.downsample_method, self.residual_connection),
            'up1'  : Up(  int( 16 * self.filters), int( 16 * self.filters), int( 16 * self.filters), self.activation, self.normalization, self.downsample_factors[4], self.upsample_method,   self.residual_connection),
            'up2'  : Up(  int( 16 * self.filters), int(  8 * self.filters), int(  8 * self.filters), self.activation, self.normalization, self.downsample_factors[3], self.upsample_method,   self.residual_connection),
            'up3'  : Up(  int(  8 * self.filters), int(  4 * self.filters), int(  4 * self.filters), self.activation, self.normalization, self.downsample_factors[2], self.upsample_method,   self.residual_connection),
            'up4'  : Up(  int(  4 * self.filters), int(  2 * self.filters), int(  2 * self.filters), self.activation, self.normalization, self.downsample_factors[1], self.upsample_method,   self.residual_connection),
            'up5'  : Up(  int(  2 * self.filters), int(  1 * self.filters), int(  1 * self.filters), self.activation, self.normalization, self.downsample_factors[0], self.upsample_method,   self.residual_connection),
            'final_conv': nn.Conv2d(self.filters, N_classes, kernel_size=1, padding=0, stride=1)
        })

        # recursively apply the initialize_weights method to all 
        # convolutional layers to initialize weights
        self.layers.apply(self.initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  Input tensor.
        
        Returns:
            logit:  Tensor after operations.
        """
        x1 = self.layers['block'](x)
        x2 = self.layers['down1'](x1)
        x3 = self.layers['down2'](x2)
        x4 = self.layers['down3'](x3)
        x5 = self.layers['down4'](x4)
        x  = self.layers['down5'](x5)
        x  = self.layers['up1'](x5, x)
        x  = self.layers['up2'](x4, x)
        x  = self.layers['up3'](x3, x)
        x  = self.layers['up4'](x2, x)
        x  = self.layers['up5'](x1, x)
        logit = self.layers['final_conv'](x)
        return logit

    def initialize_weights(self, layer: torch.nn) -> None:
        """
        Initialize the weights using the specified initialization method
        if it is a 2D convolutional layer.
        
        Args:
            layer:  Torch network layer.
        """
        # define dictionary with initialization function and names
        init_methods = {
            'xavier_uniform' : nn.init.xavier_uniform_,
            'xavier_normal'  : nn.init.xavier_normal_,
            'kaiming_uniform': lambda x: nn.init.kaiming_uniform_(
                    x, mode='fan_out', nonlinearity='relu',
                ),
            'kaiming_normal' : lambda x: nn.init.kaiming_normal_(
                    x, mode='fan_out', nonlinearity='relu',
                ),
            'zeros' : nn.init.zeros_,
        }

        if isinstance(layer, nn.Conv2d) == True:
            # select the specified weight initialization function and initialize 
            # the layer weights and biases
            if self.weight_init in init_methods.keys():
                init_methods[self.weight_init](layer.weight)
                if layer.bias != None:
                    nn.init.zeros_(layer.bias)
            else:
                raise ValueError('Invalid argument for initialization method.')
    
    def __repr__(self):
        """
        Returns total and trainable number of parameters of the model. 
        """
        parameters = 0
        trainable_parameters = 0
        # count the total and trainable number of parameters of the model
        for parameter in self.parameters():
            parameters += parameter.numel()
            if parameter.requires_grad:
                trainable_parameters += parameter.numel()   
        # create sentence with information about number of parameters
        info = (f"Total number of parameters is {parameters:,}, " 
                f"of which {trainable_parameters:,} are trainable.\n")
        
        return info


class ModifiedUNet(nn.Module):

    def __init__(
        self,
        input_channels: int,
        filters: int = 8,
        activation: str = 'relu',
        normalization: str = 'instance',
        downsample_method: str = 'max_pool',
        downsample_factors: list = [2, 2, 2, 2, 2],
        upsample_method: str = 'interpolate',
        residual_connection: bool = False,
        weight_init: str = 'kaiming_normal',
        attach_tissue_decoder: bool = True,
        attach_pen_decoder: bool = True,
        attach_distance_decoder: bool = True,
    ) -> None:
        """
        Implementation of modified U-Net with a single encoder connected to
        three decoders for tissue and pen marking segmentation, as well as
        predicting the distance to the centroid for each tissue cross-section.

        Args:
            input_channels:  Number of channels of the input tensor.
            filters:  Number of filters used in the first convolutional layer. 
                Each consecutive layer in the encoder path uses twice as many filters.
                Each consecutive layer in the decoder path uses half as many filters.
            activation:  Activation function to non-linearly transform feature maps.
            normalization:  Type of normalization layer for feature maps.
            downsample_method:  Method to downsample feature maps.  
            downsample_factors:  Factors for downsampling the feature maps.
            upsample_method:  Method to upsample feature maps.
            residual_connection:  Indicates whether a residual connection is added.
            weight_init:  Indicates which weight initialization method should be used.
            attach_tissue_decoder:  Indicates whether the tissue decoder is attached.
            attach_pen_decoder:  Indicates whether the pen decoder is attached.
            attach_distance_decoder:  Indicates whether the distance decoder is attached.
        """
        super().__init__()

        # define hyperparameters as instance attributes
        self.filters = filters
        self.activation = activation
        self.normalization = normalization
        self.downsample_method = downsample_method
        self.downsample_factors = downsample_factors
        self.upsample_method = upsample_method
        self.residual_connection = residual_connection
        self.weight_init = weight_init
        self.attach_tissue_decoder = attach_tissue_decoder
        self.attach_pen_decoder = attach_pen_decoder
        self.attach_distance_decoder = attach_distance_decoder

        # check if the sepcified combination of attached decoders is valid
        if (not (self.attach_tissue_decoder or self.attach_pen_decoder 
            or self.attach_distance_decoder)):
            raise ValueError('Atleast one decoder must be attached.')
        if self.attach_distance_decoder and not self.attach_tissue_decoder:
            raise ValueError('The tissue segmentation decoder must be attached'
                             'if the distance map decoder is attached.')

        # define the network layers
        layers = {
            'block':    Block(input_channels, int(self.filters), self.activation, self.normalization, self.residual_connection),
            'down1':    Down(int(  1 * self.filters), int(  2 * self.filters), self.activation, self.normalization, self.downsample_factors[0], self.downsample_method, self.residual_connection),
            'down2':    Down(int(  2 * self.filters), int(  4 * self.filters), self.activation, self.normalization, self.downsample_factors[1], self.downsample_method, self.residual_connection),
            'down3':    Down(int(  4 * self.filters), int(  8 * self.filters), self.activation, self.normalization, self.downsample_factors[2], self.downsample_method, self.residual_connection),
            'down4':    Down(int(  8 * self.filters), int( 16 * self.filters), self.activation, self.normalization, self.downsample_factors[3], self.downsample_method, self.residual_connection),
            'down5':    Down(int( 16 * self.filters), int( 16 * self.filters), self.activation, None,               self.downsample_factors[4], self.downsample_method, self.residual_connection),
        }
        if self.attach_tissue_decoder:
            layers = {
                **layers,
                'up_tissue1':  Up(  int( 16 * self.filters), int( 16 * self.filters), int( 16 * self.filters), self.activation, self.normalization, self.downsample_factors[4], self.upsample_method, self.residual_connection),
                'up_tissue2':  Up(  int( 16 * self.filters), int(  8 * self.filters), int(  8 * self.filters), self.activation, self.normalization, self.downsample_factors[3], self.upsample_method, self.residual_connection),
                'up_tissue3':  Up(  int(  8 * self.filters), int(  4 * self.filters), int(  4 * self.filters), self.activation, self.normalization, self.downsample_factors[2], self.upsample_method, self.residual_connection),
                'up_tissue4':  Up(  int(  4 * self.filters), int(  2 * self.filters), int(  2 * self.filters), self.activation, self.normalization, self.downsample_factors[1], self.upsample_method, self.residual_connection),
                'up_tissue5':  Up(  int(  2 * self.filters), int(  1 * self.filters), int(  1 * self.filters), self.activation, self.normalization, self.downsample_factors[0], self.upsample_method, self.residual_connection),
                'final_conv_tissue': nn.Conv2d(self.filters, 1, kernel_size=3, padding=1, padding_mode='zeros', stride=1),
            }
        if self.attach_pen_decoder:
            layers = {
                **layers,
                'up_pen1': Up(  int( 16 * self.filters), int( 16 * self.filters), int( 16 * self.filters), self.activation, self.normalization, self.downsample_factors[4], self.upsample_method, self.residual_connection),
                'up_pen2': Up(  int( 16 * self.filters), int(  8 * self.filters), int(  8 * self.filters), self.activation, self.normalization, self.downsample_factors[3], self.upsample_method, self.residual_connection),
                'up_pen3': Up(  int(  8 * self.filters), int(  4 * self.filters), int(  4 * self.filters), self.activation, self.normalization, self.downsample_factors[2], self.upsample_method, self.residual_connection),
                'up_pen4': Up(  int(  4 * self.filters), int(  2 * self.filters), int(  2 * self.filters), self.activation, self.normalization, self.downsample_factors[1], self.upsample_method, self.residual_connection),
                'up_pen5': Up(  int(  2 * self.filters), int(  1 * self.filters), int(  1 * self.filters), self.activation, self.normalization, self.downsample_factors[0], self.upsample_method, self.residual_connection),
                'final_conv_pen': nn.Conv2d(self.filters, 1, kernel_size=3, padding=1, padding_mode='zeros', stride=1),
            }
        if self.attach_distance_decoder:
            layers = {
                **layers,
                'up_distance1':  Up(  int( 16 * self.filters), int( 16 * self.filters), int( 16 * self.filters), self.activation, self.normalization, self.downsample_factors[4], self.upsample_method, self.residual_connection),
                'up_distance2':  Up(  int( 16 * self.filters), int(  8 * self.filters), int(  8 * self.filters), self.activation, self.normalization, self.downsample_factors[3], self.upsample_method, self.residual_connection),
                'up_distance3':  Up(  int(  8 * self.filters), int(  4 * self.filters), int(  4 * self.filters), self.activation, self.normalization, self.downsample_factors[2], self.upsample_method, self.residual_connection),
                'up_distance4':  Up(  int(  4 * self.filters), int(  2 * self.filters), int(  2 * self.filters), self.activation, self.normalization, self.downsample_factors[1], self.upsample_method, self.residual_connection),
                'up_distance5':  Up(  int(  2 * self.filters), int(  1 * self.filters), int(  1 * self.filters), self.activation, self.normalization, self.downsample_factors[0], self.upsample_method, self.residual_connection),
                'final_conv_distance': nn.Conv2d(self.filters, 2, kernel_size=3, padding=1, padding_mode='zeros', stride=1),
            }
        self.layers = nn.ModuleDict(layers)
        # recursively apply the initialize_weights method 
        # to all convolutional layers to initialize weights
        self.layers.apply(self.initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  Input tensor.
        
        Returns:
            out:  Tensor after operations.
        """
        outputs = {}

        # encoder
        x1 = self.layers['block'](x)
        x2 = self.layers['down1'](x1)
        x3 = self.layers['down2'](x2)
        x4 = self.layers['down3'](x3)
        x5 = self.layers['down4'](x4)
        x = self.layers['down5'](x5)
        
        # tissue segmentation and distance map decoder
        if self.attach_tissue_decoder:
            x_tissue = self.layers['up_tissue1'](x5, x)
            x_tissue = self.layers['up_tissue2'](x4, x_tissue)
            x_tissue = self.layers['up_tissue3'](x3, x_tissue)
            x_tissue = self.layers['up_tissue4'](x2, x_tissue)
            x_tissue = self.layers['up_tissue5'](x1, x_tissue)
            out_tissue = self.layers['final_conv_tissue'](x_tissue)
            outputs['tissue'] = out_tissue
        
        # pen segmentation decoder
        if self.attach_pen_decoder:
            x_pen = self.layers['up_pen1'](x5, x)
            x_pen = self.layers['up_pen2'](x4, x_pen)
            x_pen = self.layers['up_pen3'](x3, x_pen)
            x_pen = self.layers['up_pen4'](x2, x_pen)
            x_pen = self.layers['up_pen5'](x1, x_pen)
            out_pen = self.layers['final_conv_pen'](x_pen)
            outputs['pen'] = out_pen
        
        # tissue distance map decoder
        if self.attach_distance_decoder:
            x_distance = self.layers['up_distance1'](x5, x)
            x_distance = self.layers['up_distance2'](x4, x_distance)
            x_distance = self.layers['up_distance3'](x3, x_distance)
            x_distance = self.layers['up_distance4'](x2, x_distance)
            x_distance = self.layers['up_distance5'](x1, x_distance)
            out_distance = self.layers['final_conv_distance'](x_distance)
            outputs['distance'] = torch.sigmoid(out_tissue)*out_distance
        
        return outputs

    def initialize_weights(self, layer: torch.nn) -> None:
        """
        Initialize the weights using the specified initialization method
        if it is a 2D convolutional layer.
        
        Args:
            layer:  Torch network layer.
        """
        # define dictionary with initialization function and names
        init_methods = {
            'xavier_uniform' : nn.init.xavier_uniform_,
            'xavier_normal'  : nn.init.xavier_normal_,
            'kaiming_uniform': lambda x: nn.init.kaiming_uniform_(
                    x, mode='fan_out', nonlinearity='relu',
                ),
            'kaiming_normal' : lambda x: nn.init.kaiming_normal_(
                    x, mode='fan_out', nonlinearity='relu',
                ),
            'zeros' : nn.init.zeros_,
        }

        if isinstance(layer, nn.Conv2d) == True:
            # select the specified weight initialization function and initialize 
            # the layer weights and biases
            if self.weight_init in init_methods.keys():
                init_methods[self.weight_init](layer.weight)
                if layer.bias != None:
                    nn.init.zeros_(layer.bias)
            else:
                raise ValueError('Invalid argument for initialization method.')

    def __repr__(self):
        """
        Returns total and trainable number of parameters of the model. 
        """
        parameters = 0
        trainable_parameters = 0
        # count the total and trainable number of parameters of the model
        for parameter in self.parameters():
            parameters += parameter.numel()
            if parameter.requires_grad:
                trainable_parameters += parameter.numel()   
        # create sentence with information about number of parameters
        info = (f"Total number of parameters is {parameters:,}, " 
                f"of which {trainable_parameters:,} are trainable.\n")
        
        return info


def partition_model_state_dict(checkpoint_path: Union[Path, str], parts: int = 1) -> None:
    """
    Load a saved model checkpoint (.pth), select only the 'model_state_dict',
    optionally split it up into multiple parts, and save the state dictionary 
    of the model.

    Args:
        checkpoint_path: path to saved model checkpoint.
        parts: number of parts in which the state dictionary of the model is split.
    """
    # convert to Path object in case it is a string
    checkpoint_path = Path(checkpoint_path)

    # load the checkpoint and select only the state dictionary of the model
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint['model_state_dict']
    if parts < 1:
        raise ValueError('The number of parts must be equal to or larger than 1.')
    elif parts == 1:
        model_state_dict_path = checkpoint_path.parent / 'model_state_dict.pth'
        if model_state_dict_path.exists():
            raise FileExistsError('"model_state_dict.pth" already exists.')
        torch.save(model_state_dict, model_state_dict_path)
    else:
        # sort the layer names based on the number of parameters
        layers = []
        for layer_name, parameters in model_state_dict.items():
            layers.append((parameters.numel(), layer_name))
        layers = sorted(layers)
        # decrease the number of parts to the number of layers if less layers
        # are available than part.
        parts = min(len(layers), parts)
        # divide layers over parts
        model_state_dicts = [{} for _ in range(parts)]
        for i, (_, layer) in enumerate(layers):
            model_state_dicts[i%parts][layer] = model_state_dict[layer]
        # save all parts
        for i, state_dict in enumerate(model_state_dicts):
            part_name = f'model_state_dict_pt{str(i+1).zfill(3)}.pth'
            model_state_dict_path = checkpoint_path.parent / part_name
            if model_state_dict_path.exists():
                raise FileExistsError(f'{part_name} already exists.')
            torch.save(state_dict, model_state_dict_path)


if __name__ == '__main__':
    # define U-Net network instance
    net = UNet(3, 2).to('cuda')
    input_shape = (1, 3, 512, 512)
    summary(net, input_shape, depth=3, col_names=["input_size", "output_size", "num_params"])