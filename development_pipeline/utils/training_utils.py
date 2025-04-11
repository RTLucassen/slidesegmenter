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
Utility classes for training neural networks.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(
        self, 
        sigmoid: bool = False, 
        class_weights: Optional[list[float]] = None,
        smooth_nom: float = 1, 
        smooth_denom: float = 1,
    ) -> None:
        """
        Initialize Dice loss.

        Args:
            sigmoid:  Specify if a sigmoid instead of a softmax function is applied.
                If there is only a single class, the sigmoid is automatically used.
            class_weights:  If not None, compute a weighted average of the loss 
                for the classes. 
            smooth_nom:  Small value added to the nominator to better handle 
                negative cases.
            smooth_denom:  Small value added to the denominator to prevent division 
                by zero errors and to better handle negative cases.
        """
        super().__init__()
        # define the instance attributes
        self.sigmoid = sigmoid
        self.smooth_nom = smooth_nom
        self.smooth_denom = smooth_denom
        self.class_weights = class_weights

    def forward(self, logit: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            logit:  Logit predictions of shape: (batch, class, X, Y, ...).
            y_true:  True labels of shape: (batch, class, X, Y, ...).
        
        Returns:
            loss:  Dice loss averaged over all images in the batch.
        """
        # check if the logit prediction and true labels are of equal shape
        if logit.shape != y_true.shape:
            raise ValueError('Shape of predictions and true labels do not match.')
        
        # check if the values of y_true range between 0.0-1.0
        if torch.min(y_true) < 0.0 or torch.max(y_true) > 1.0:
            raise ValueError('Invalid values for y_true (outside the range 0.0-1.0).')
        
        # check if the number of classes matches the number of class weights if provided
        if self.class_weights is not None:
            if len(self.class_weights) != logit.shape[1]:
                raise ValueError('The number of class weights and classes do not match.')
        else:
            self.class_weights = [1]*logit.shape[1]

        # get the pixel-wise predicted probabilities by applying
        # a sigmoid or softmax function to the logit returned by the network
        if self.sigmoid or logit.shape[1] == 1:
            y_pred = torch.sigmoid(logit)
        else:
            y_pred = torch.softmax(logit, dim=1)

        # flatten the image (but keep the dimension of the batch and channels)
        y_true_flat = y_true.contiguous().view(*y_true.shape[0:2], -1)
        y_pred_flat = y_pred.contiguous().view(*y_pred.shape[0:2], -1)
        
        # compute the dice loss per channel for each image in the batch
        intersection = torch.sum(y_true_flat * y_pred_flat, dim=-1)
        union = torch.sum(y_true_flat, dim=-1) + torch.sum(y_pred_flat, dim=-1)
        class_separated_loss = 1 - ((2. * intersection + self.smooth_nom) / (union + self.smooth_denom))
        
        # multiply the dice score per class by the class weight
        for i, class_weight in enumerate(self.class_weights):
            class_separated_loss[:, i] *= class_weight
        
        # compute the batch loss
        loss = torch.mean(class_separated_loss)

        return loss


class TverskyLoss(nn.Module):

    def __init__(
        self, 
        sigmoid: bool = False, 
        class_weights: Optional[list[float]] = None,
        fp_weight: float = 0.5,
        fn_weight: float = 0.5,
        smooth_nom: float = 1, 
        smooth_denom: float = 1,
    ) -> None:
        """
        Initialize Tversky loss.

        Args:
            sigmoid:  Specify if a sigmoid instead of a softmax function is applied.
                If there is only a single class, the sigmoid is automatically used.
            class_weights:  If not None, compute a weighted average of the loss 
                for the classes. 
            fp_weight:  Weighting factor for false positives (alpha in paper).
            fn_weight:  Weighting factor for false negatives (beta in paper).
            smooth_nom:  Small value added to the nominator to better handle 
                negative cases.
            smooth_denom:  Small value added to the denominator to prevent division 
                by zero errors and to better handle negative cases.
        """
        super().__init__()
        # define the instance attributes
        self.sigmoid = sigmoid
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight
        self.smooth_nom = smooth_nom
        self.smooth_denom = smooth_denom
        self.class_weights = class_weights

    def forward(self, logit: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            logit:  Logit predictions of shape: (batch, class, X, Y, ...).
            y_true:  True labels of shape: (batch, class, X, Y, ...).
        
        Returns:
            loss:  Tversky loss averaged over all images in the batch.
        """
        # check if the logit prediction and true labels are of equal shape
        if logit.shape != y_true.shape:
            raise ValueError('Shape of predictions and true labels do not match.')
        
        # check if the values of y_true range between 0.0-1.0
        if torch.min(y_true) < 0.0 or torch.max(y_true) > 1.0:
            raise ValueError('Invalid values for y_true (outside the range 0.0-1.0).')
        
        # check if the number of classes matches the number of class weights if provided
        if self.class_weights is not None:
            if len(self.class_weights) != logit.shape[1]:
                raise ValueError('The number of class weights and classes do not match.')
        else:
            self.class_weights = [1]*logit.shape[1]

        # get the pixel-wise predicted probabilities by applying
        # a sigmoid or softmax function to the logit returned by the network
        if self.sigmoid or logit.shape[1] == 1:
            y_pred = torch.sigmoid(logit)
        else:
            y_pred = torch.softmax(logit, dim=1)

        # flatten the image (but keep the dimension of the batch and channels)
        y_true_flat = y_true.contiguous().view(*y_true.shape[0:2], -1)
        y_pred_flat = y_pred.contiguous().view(*y_pred.shape[0:2], -1)
        
        # compute the dice loss per channel for each image in the batch
        intersection = torch.sum(y_true_flat * y_pred_flat, dim=-1)
        fps = torch.sum(y_pred_flat*(1-y_true_flat), dim=-1)
        fns = torch.sum((1-y_pred_flat)*y_true_flat, dim=-1)
        class_separated_loss = 1 - ((intersection + self.smooth_nom) / 
            (intersection + self.fp_weight*fps + self.fn_weight*fns + self.smooth_denom))
        
        # multiply the dice score per class by the class weight
        for i, class_weight in enumerate(self.class_weights):
            class_separated_loss[:, i] *= class_weight
        
        # compute the batch loss
        loss = torch.mean(class_separated_loss)

        return loss


class FocalLoss(nn.Module):

    def __init__(
        self,
        sigmoid: bool = False, 
        gamma: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        aggregation='mean',
    ) -> None:
        """
        Initialize focal loss.

        Args:
            sigmoid:  Specify if a sigmoid instead of a softmax function is applied.
                If there is only a single class, the sigmoid is automatically used.
            gamma:  Parameter that governs the relative importance of incorrect 
                predictions. If gamma equals 0.0, the focal loss is equal to the 
                cross-entropy loss.
            class_weights:  If not None, compute a weighted average of the loss 
                for the classes. 
        """
        super().__init__()
        self.sigmoid = sigmoid
        self.gamma = gamma
        self.class_weights = class_weights
        if aggregation == 'mean':
            self.aggregation = torch.mean
        elif aggregation == 'sum':
            self.aggregation = torch.sum
        else:
            ValueError('Invalid aggregation function')

    def forward(self, logit: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            logit:  Logit predictions volumes of shape: (batch, class, X, Y, ...).
            y_true:  True label volumes of matching shape: (batch, class, X, Y, ...).
        
        Returns:
            loss:  Focal loss averaged over all images in the batch.
        """
        # check if the logit prediction and true labels are of equal shape
        if logit.shape != y_true.shape:
            raise ValueError('Shape of predicted and true labels do not match.')
        
        # check if the values of y_true range between 0.0-1.0
        if torch.min(y_true) < 0.0 or torch.max(y_true) > 1.0:
            raise ValueError('Invalid values for y_true (outside the range 0.0-1.0).')
        
        # check if the number of classes matches the number of class weights if provided
        if self.class_weights is not None:
            if len(self.class_weights) != logit.shape[1]:
                raise ValueError('The number of class weights and classes do not match.')
        else:
            self.class_weights = [1]*logit.shape[1]

        # get the pixel-wise predicted probabilities by taking
        # the sigmoid or softmax of the logit returned by the network
        if self.sigmoid or logit.shape[1] == 1:
            y_pred = torch.sigmoid(logit)
            log_y_pred = F.logsigmoid(logit)
        else:
            y_pred = torch.softmax(logit, dim=1)
            log_y_pred = F.log_softmax(logit, dim=1)

        # flatten the images and labels (but keep the dimension of the batch and channels)
        y_true_flat = y_true.contiguous().view(*y_true.shape[0:2], -1)
        y_pred_flat = y_pred.contiguous().view(*y_pred.shape[0:2], -1)
        log_y_pred_flat = log_y_pred.contiguous().view(*y_pred.shape[0:2], -1)

        # calculate the pixelwise cross-entropy, focal weight, and pixelwise focal loss
        pixelwise_CE = -(log_y_pred_flat * y_true_flat)
        focal_weight = (1-(y_true_flat * y_pred_flat))**self.gamma
        pixelwise_focal_loss = focal_weight * pixelwise_CE

        # calculate the class-separated focal loss
        class_separated_focal_loss = self.aggregation(pixelwise_focal_loss, dim=-1)
        
        # multiply the dice score per class by the class weight
        for i, class_weight in enumerate(self.class_weights):
            class_separated_focal_loss[:, i] *= class_weight
        instance_loss = torch.sum(class_separated_focal_loss, dim=1)

        # compute the mean loss over the batch
        loss = torch.mean(instance_loss)

        return loss
    

class MSELoss(nn.Module):

    def __init__(self, aggregation='mean') -> None:
        super().__init__()
        if aggregation == 'mean':
            self.aggregation = torch.mean
        elif aggregation == 'sum':
            self.aggregation = torch.sum
        else:
            ValueError('Invalid aggregation function')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            y_pred:  Predictions volumes of shape: (batch, class, X, Y, ...).
            y_true:  True label volumes of matching shape: (batch, class, X, Y, ...).
        
        Returns:
            loss:  MSE loss averaged over all images in the batch.
        """
        # check if the prediction and true labels are of equal shape
        if y_pred.shape != y_true.shape:
            raise ValueError('Shape of predicted and true labels do not match.')
                
        # flatten the images and labels (but keep the dimension of the batch and channels)
        y_true_flat = y_true.contiguous().view(*y_true.shape[0:2], -1)
        y_pred_flat = y_pred.contiguous().view(*y_pred.shape[0:2], -1)

        # calculate the mean squared error
        MSE = self.aggregation((y_true_flat-y_pred_flat)**2, dim=-1)

        # compute the mean loss over the batch
        instance_loss = torch.sum(MSE, dim=1)
        loss = torch.mean(instance_loss)

        return loss


class MAELoss(nn.Module):

    def __init__(self, aggregation='mean') -> None:
        super().__init__()
        if aggregation == 'mean':
            self.aggregation = torch.mean
        elif aggregation == 'sum':
            self.aggregation = torch.sum
        else:
            ValueError('Invalid aggregation function')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            y_pred:  Predictions volumes of shape: (batch, class, X, Y, ...).
            y_true:  True label volumes of matching shape: (batch, class, X, Y, ...).
        
        Returns:
            loss:  MAE loss averaged over all images in the batch.
        """
        # check if the prediction and true labels are of equal shape
        if y_pred.shape != y_true.shape:
            raise ValueError('Shape of predicted and true labels do not match.')
                
        # flatten the images and labels (but keep the dimension of the batch and channels)
        y_true_flat = y_true.contiguous().view(*y_true.shape[0:2], -1)
        y_pred_flat = y_pred.contiguous().view(*y_pred.shape[0:2], -1)

        # calculate the mean absolute error
        MAE = self.aggregation(torch.abs(y_true_flat-y_pred_flat), dim=-1)

        # compute the mean loss over the batch
        instance_loss = torch.sum(MAE, dim=1)
        loss = torch.mean(instance_loss)

        return loss


class CombinedLoss(nn.Module):

    def __init__(
        self, 
        device: str, 
        loss_weights: dict = {
            'tversky': 1.0, 
            'focal': 1.0, 
            'MSE dist': 1.0, 
            'MSE grad dist': 1.0,
        }, 
        class_weights: dict = {
            'tissue': 1.0, 
            'pen': 1.0, 
        }, 
        fp_weight: float = 0.5, 
        fn_weight: float = 0.5, 
        gamma: float = 0.0,
        smooth_nom: float = 1.0,
        smooth_denom: float = 1.0,
    ) -> None:
        """
        Initialize combination of Tversky and focal loss for segmentation, 
        as well as the combination of MSE loss with respect to the predicted 
        distance map and the gradient of the predicted distance map.

        Args:
            device:  Torch device specification.    
            loss_weights:  If not None, compute a weighted average of the losses. 
            class_weights:  If not None, compute a weighted average of the loss
                for the classes. 
            fp_weight:  Weighting factor for false positives (alpha in paper).
            fn_weight:  Weighting factor for false negatives (beta in paper).
            gamma:  Parameter that governs the relative importance of incorrect 
                predictions. If gamma equals 0.0, the focal loss is equal to 
                the cross-entropy loss.
            smooth_nom:  Small value added to the nominator to better handle 
                negative cases.
            smooth_denom:  Small value added to the denominator to prevent division 
                by zero errors and to better handle negative cases.
        """
        super().__init__()

        # define loss names
        self.names = ['tversky', 'focal', 'MSE dist', 'MSE grad dist']

        # initialize weights
        self.loss_weights = loss_weights
        self.class_weights = class_weights

        # initialize loss components
        self.tversky_loss = TverskyLoss(
            sigmoid=True,
            class_weights=None,
            fp_weight=fp_weight, 
            fn_weight=fn_weight, 
            smooth_nom=smooth_nom,
            smooth_denom=smooth_denom,
        )
        self.focal_loss = FocalLoss(
            sigmoid=True,
            gamma=gamma, 
            class_weights=None,
        )
        self.MSE_loss = MSELoss()

        # initialize gradient kernel 
        grad_kernel = torch.zeros((1, 2, 3, 3))
        grad_kernel[0, 0, 1, 0] = -1 
        grad_kernel[0, 0, 1, 2] = 1 
        grad_kernel[0, 1, 0, 1] = -1 
        grad_kernel[0, 1, 2, 1] = 1 
        self.grad_kernel = grad_kernel.to(device)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            y_pred:  Predictions volumes of shape: (batch, class, X, Y, ...).
            y_true:  True label volumes of matching shape: (batch, class, X, Y, ...).
        
        Returns:
            loss:  Combined loss averaged over all images in the batch.
        """
        # calculate the combined loss
        losses = {'tversky': [], 'focal': [], 'MSE dist': [], 'MSE grad dist': []}
        for decoder, output in y_pred.items():
            if decoder == 'tissue':
                losses['tversky'].append(self.tversky_loss(output, y_true[:, 0:1, ...])
                                         * self.loss_weights['tversky']
                                         * self.class_weights['tissue'])
                losses['focal'].append((self.focal_loss(output, y_true[:, 0:1, ...])
                                        + self.focal_loss(-output, 1-y_true[:, 0:1, ...]))
                                        * self.loss_weights['focal']
                                        * self.class_weights['tissue'])
            elif decoder == 'pen':
                losses['tversky'].append(self.tversky_loss(output, y_true[:, 1:2, ...])
                                         * self.loss_weights['tversky']
                                         * self.class_weights['pen'])
                losses['focal'].append((self.focal_loss(output, y_true[:, 1:2, ...])
                                        + self.focal_loss(-output, 1-y_true[:, 1:2, ...]))
                                        * self.loss_weights['focal']
                                        * self.class_weights['pen'])
            elif decoder == 'distance':
                losses['MSE dist'].append(self.MSE_loss(output, y_true[:, 2:, ...])
                                          * self.loss_weights['MSE dist']),
                losses['MSE grad dist'].append(self.MSE_loss(F.conv2d(output, self.grad_kernel), 
                                                             F.conv2d(y_true[:, 2:, ...], self.grad_kernel),)
                                               * self.loss_weights['MSE grad dist'])

        losses = {name: sum(loss_values)/len(loss_values) 
                  for name, loss_values in losses.items() if len(loss_values)}
        
        return losses