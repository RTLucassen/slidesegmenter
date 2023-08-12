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
Function for evaluating predictive performance.
"""

from typing import Optional, Sequence, Union

import numpy as np
import torch


def mean_stdev(values: Sequence) -> float:
    # remove None, np.nan, and torch.nan cases
    numeric_values = []
    for value in values:
        if value is None:
            pass
        elif np.isnan(value):
            pass
        else:
            numeric_values.append(value)
    
    # check if there are at least two values for computing the standard deviation
    if len(numeric_values) > 1:
        mean = sum(numeric_values)/len(numeric_values)
        squared_diff_sum = sum((x - mean) ** 2 for x in numeric_values)
        variance = squared_diff_sum / (len(numeric_values) - 1)  # Bessel's correction
        stdev = variance ** 0.5
        return mean, stdev
    # check if there is at least one value for returning the mean
    elif len(numeric_values) == 1:
        mean = numeric_values[0]
        return mean, None
    else:
        return None, None

def dice_score(
    y_hat: Union[torch.Tensor, np.ndarray], 
    y_true: Union[torch.Tensor, np.ndarray], 
    average_classes: bool = False,
    class_weights: Optional[list[float]] = None,
    smooth_denom: float = 1e-8,
) -> Optional[torch.Tensor]:
    """
    Calculates the Dice score.

    Args:
        y_hat: predicted probabilities of shape: (batch, class, X, Y, ...).
        y_true: true label of shape: (batch, class, X, Y, ...).
        class_weights: if not None, compute a weighted average score for the classes. 
        smooth_denom: small value added to the denominator to prevent division 
                      by zero errors and to better handle negative cases.
    Returns:
        score: Dice score for all items in the batch.
    """
    # convert to numpy arrays if necessary
    if isinstance(y_hat, np.ndarray):
        y_hat = torch.from_numpy(y_hat)
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)

    # check if the logit prediction and true labels are of equal shape
    if y_hat.shape != y_true.shape:
        raise ValueError('Shape of predicions and true labels do not match.')

    # check if the values of y_hat and y_true range between 0.0-1.0
    if torch.min(y_hat) < 0.0 or torch.max(y_hat) > 1.0:
        raise ValueError('Invalid values for y_hat (outside the range 0.0-1.0).')
    if torch.min(y_true) < 0.0 or torch.max(y_true) > 1.0:
        raise ValueError('Invalid values for y_true (outside the range 0.0-1.0).')
    
    # check if the number of classes matches the number of class weights if provided
    if class_weights is not None:
        if len(class_weights) != y_hat.shape[1]:
            raise ValueError('The number of class weights and classes do not match.')
    else:
        class_weights = [1]*y_hat.shape[1]

    # flatten the image (but keep the dimension of the batch and channels)
    y_true_flat = y_true.contiguous().view(*y_true.shape[0:2], -1)
    y_hat_flat = y_hat.contiguous().view(*y_hat.shape[0:2], -1)
    
    # compute the dice score
    intersection = torch.sum(y_true_flat * y_hat_flat, dim=-1)
    union = torch.sum(y_true_flat, dim=-1) + torch.sum(y_hat_flat, dim=-1)
    class_separated_score = (2. * intersection) / (union + smooth_denom)
    
    # multiply the dice score per class by the class weight
    for i, class_weight in enumerate(class_weights):
        class_separated_score[:, i] *= class_weight
    
    # determine whether there are empty classes
    y_true_sum = torch.sum(y_true_flat, dim=-1)
    class_separated_score = torch.where(y_true_sum >= 1, class_separated_score, torch.nan)

    if not average_classes:
        score = torch.mean(class_separated_score)
        return score
    else:
        return class_separated_score