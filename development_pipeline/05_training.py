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
Training loop for model.
"""

# suppress irrelevant pytorch warning until fix
import warnings
warnings.filterwarnings(
    'ignore', 
    category=UserWarning, 
    message='TypedStorage is deprecated',
)

import io
import json
import logging
import os
import platform
import random
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchinfo import summary
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from config import PROJECT_SEED
from config import annotations_folder, images_folder, models_folder, sheets_folder
from utils.models import ModifiedUNet
from utils.training_utils import CombinedLoss
from utils.dataset_utils import seed_worker, SupervisedTrainingDataset
from utils.visualization_utils import image_viewer, rgb_image_viewer 


# define settings class
class Settings():

    def __init__(self, settings: dict[str, Any]) -> None:
        """
        Initialize settings

        Args:
            settings: settings configuration loaded from JSON file.
        """
        # initialize variables
        self.experiment_name = settings['experiment_name']
        self.dataset_filename = settings['dataset_filename']
        self.seed = settings['seed']
        self.model_name = settings['model_name']
        self.compile_model = settings['compile_model']
        self.checkpoint_path = settings['checkpoint_path']

        self.model = settings['model']
        self.training = settings['training']
        self.augmentation = settings['augmentation']
        self.dataloader = {
            **settings['dataloader'],
            'worker_init_fn': seed_worker,
            'generator': torch.Generator(),
        }

# force the use of the CPU even if a GPU is available
USE_CPU = False

# specify experiment and dataset settings
configuration = {
    "experiment_name": f'Modified_U-Net_{datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")}',
    "dataset_filename": 'dataset.xlsx',
    "seed": PROJECT_SEED,
    "model_name": "ModifiedUNet",
    "compile_model": False,
    "checkpoint_path": False,

    # specify model hyperparameters
    "model": {
        "input_channels": 3,
        "filters": 32,
        "downsample_factors": [2, 4, 4, 4, 4],
        "residual_connection": False,
    },

    # specify training hyperparameters
    "training": {
        "learning_rate": 3e-4,
        "iterations": 100000,
        "iterations_per_decay": 20000,
        "decay_factor": 0.5,
        "iterations_per_checkpoint": 500,
        "iterations_per_update": 5,
        "loss": {
            "weights": [1, 10, 1, 1],
            "class_weights": [1, 1],
            "fp_weight": 0.5,
            "fn_weight": 0.5,
            "gamma": 0,
            "smooth_nom": 1,
            "smooth_denom": 1,
        },
    },
    "dataloader": {
        "num_workers": 4,
        "pin_memory": True,
        "batch_size": 1,
        "image_shape": None,
        "max_image_shape": (3072, 3072),
    },

    # specify augmentation hyperparameters
    "augmentation": {
        "RandomRotate90": {
            "p": 0.25,
        },
        "Affine": {
            "p": 0.5,
            "translate_px": {
               "x": (-256, 256), 
               "y": (-256, 256),
            },
            "scale": (0.95, 1.05),
            "rotate": (-180, 180),
        },
        "HorizontalFlip": {
            "p": 0.5,
        },
        "VerticalFlip": {
            "p": 0.5,
        },
        "HueSaturationValue": {
            "p": 0.5,
            "hue_shift_limit": 25,
            "sat_shift_limit": 5,
            "val_shift_limit": 0,
        },
        "HueSaturationValue pen": {
            "p": 0.25,
            "hue_shift_limit": 10,
            "sat_shift_limit": 5,
            "val_shift_limit": 0,
        },
        "HueSaturationValue tissue": {
            "p": 0.25,
            "hue_shift_limit": 10,
            "sat_shift_limit": 5,
            "val_shift_limit": 0,
        },
        "HueSaturationValue non-tissue": {
            "p": 0.0,
            "hue_shift_limit": 0,
            "sat_shift_limit": 0,
            "val_shift_limit": 0,
        },
        "HueSaturationValue background": {
            "p": 0.25,
            "hue_shift_limit": 10,
            "sat_shift_limit": 5,
            "val_shift_limit": 0,
        },
        "RandomBrightnessContrast": {
           "p": 0.25,
            "brightness_limit": 0.2,
            "contrast_limit": 0.2,
        },
        "RandomGamma": {
            "p": 0.25,
            "gamma_limit": (67, 150),
        },
        "GaussNoise": {
            "p": 0.2,
            "var_limit": (0, 25),
        },
        "GaussianBlur": {
            "p": 0.2,
            "sigma_limit": (0.00001, 2),
        },
        "JPEGCompression": {
            "p": 0.2,
            "quality_limit": (25, 100),
        },
        "Padding": [
            {
                "mode": "reflect",
                "p": 0.9,
            },
            {
                "mode": "constant",
                "p": 0.0,
                "value": 0,
            }, 
            {
                "mode": "constant",
                "p": 0.1,
                "value": 255,
            },
        ],
        "PenMarkings": {
            "p": 0.5,
            "N": (1, 8),
        }
    }
}

# logging settings
experiment_folder = configuration['experiment_name']

if __name__ == '__main__':

    # create a folder to store the model and settings after training
    output_folder = models_folder / experiment_folder
    if output_folder.exists():
        raise FileExistsError('Specified output directory already exists.')
    else:
        output_folder.mkdir()

    # save variables as json
    with open(output_folder / 'settings.json', 'w') as f:
        json.dump(configuration, f)

    # parse JSON into an object with attributes corresponding to dict keys.
    with open(output_folder / 'settings.json', 'r') as f:
        settings = Settings(json.load(f))

    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(output_folder, 'training_log.txt'),
        format='%(asctime)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        encoding='utf-8',
    )
    logger = logging.getLogger(__name__)

    # configure the number of workers and the device
    device = 'cuda' if (torch.cuda.is_available() and not USE_CPU) else 'cpu'
    logger.info(f'CUDA available: {torch.cuda.is_available()}')
    logger.info(f'Device: {device}')

    # seed randomness
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.backends.cudnn.benchmark = True
    settings.dataloader['generator'].manual_seed(settings.seed)
    logger.info(f'Using seed: {settings.seed}')

    # load the dataset information
    df_all = pd.read_excel(sheets_folder / settings.dataset_filename, sheet_name='images')
    df_pen = pd.read_excel(sheets_folder / settings.dataset_filename, sheet_name='pen_markings')
    
    # create full image and annotation paths
    for df in [df_all, df_pen]:
        for column, folder in zip(['image_paths', 'annotation_paths'], [images_folder, annotations_folder]):
            full_paths = []
            for relative_path in list(df[column]):
                full_paths.append((folder / relative_path).as_posix())      
            df[column] = full_paths

    # separate splits
    df_train = df_all[df_all['set']=='train']
    df_pen_train = df_pen[df_pen['set']=='train']
    df_val = df_all[df_all['set']=='val']

    # initialize training dataset instance and dataloader
    train_dataset = SupervisedTrainingDataset(
        df=df_train,
        df_pen=df_pen_train,
        length=settings.training['iterations']*settings.dataloader['batch_size'],
        shape=settings.dataloader['image_shape'],
        max_shape=settings.dataloader['max_image_shape'],
        divisor=np.prod(settings.model['downsample_factors']),
        augmentations=settings.augmentation,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler= RandomSampler(train_dataset, replacement=True),
        batch_size=settings.dataloader['batch_size'],
        num_workers=settings.dataloader['num_workers'],
        pin_memory=settings.dataloader['pin_memory'],
    )

    # initialize validation dataset instance and dataloader
    val_dataset = SupervisedTrainingDataset(
        df=df_val,
        df_pen=None,
        length=None,
        shape=None,
        max_shape=None,
        divisor=np.prod(settings.model['downsample_factors']),
        augmentations={},
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        sampler= SequentialSampler(val_dataset),
        batch_size=1,
        num_workers=settings.dataloader['num_workers'],
        pin_memory=settings.dataloader['pin_memory'],
    )

    # initialize model
    if settings.model_name == 'ModifiedUNet':
        model = ModifiedUNet(**settings.model)
    else:
        raise ValueError('Model name was not recognized.')

    # load checkpoint if specified
    if isinstance(settings.checkpoint_path, str):
        # load the checkpoint model settings
        settings_path = Path(settings.checkpoint_path).parent / 'settings.json'
        with open(settings_path, 'r') as f:
            checkpoint_settings = Settings(json.load(f))

        # check if the models are the same
        if checkpoint_settings.model_name == 'UNet':
            for key in checkpoint_settings.model:
                if key not in ['N_classes']:
                    if checkpoint_settings.model[key] != settings.model[key]:
                        raise ValueError('Atleast one of the model settings differs.')
            # initialize the model parameters based on the model checkpoint
            checkpoint = torch.load(
                settings.checkpoint_path,
                map_location=torch.device('cpu')
            )
            # convert state_dict to account for changes between UNet and ModifiedUNet model
            converted_state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                if 'final' in key:
                    for replacement in ['final_conv_tissue', 'final_conv_distance']:
                        new_key = key.replace('final_conv', replacement)
                        converted_state_dict[new_key] = model.state_dict()[new_key]
                elif 'up' in key:
                    converted_state_dict[key.replace('up', 'up_tissue')] = value
                    converted_state_dict[key.replace('up', 'up_distance')] = value
                else:
                    converted_state_dict[key] = value
            # initialize model parameters from state dict
            model.load_state_dict(converted_state_dict)

        elif checkpoint_settings.model_name == 'ModifiedUNet':
            # check if the models are the same
            if checkpoint_settings.model != settings.model:
                raise ValueError(
                    'Specified model settings do not match the checkpoint model settings.'
                )

            # initialize the model parameters based on the model checkpoint
            checkpoint = torch.load(
                settings.checkpoint_path,
                map_location=torch.device('cpu')
            )
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError('Model name not recognized.')

    # transfer the model to the selected device
    model = model.to(device)

    # capture model summary in variable
    f = io.StringIO()
    with redirect_stdout(f):
        if settings.dataloader['image_shape'] is None:
            summary(model=model, depth=4, col_names=['num_params'])
        else:
            summary(
                model=model,
                input_size=(
                    settings.dataloader['batch_size'],
                    settings.model['input_channels'],
                    *settings.dataloader['image_shape'],
                ),
                depth=4,
                col_names=['input_size', 'output_size', 'num_params'],
            )
    logger.info('\n'+f.getvalue())
    logger.info(model)
        
    # compile the model
    if platform.system() == 'Linux' and settings.compile_model:
        model = torch.compile(model)

    # initialize optimizer and loss function
    learning_rate = settings.training['learning_rate']
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_function = CombinedLoss(
        device=device,
        weights=settings.training['loss']['weights'],
        class_weights=settings.training['loss']['class_weights'],
        fp_weight=settings.training['loss']['fp_weight'],
        fn_weight=settings.training['loss']['fn_weight'],
        gamma=settings.training['loss']['gamma'],
        smooth_nom=settings.training['loss']['smooth_nom'],
        smooth_denom=settings.training['loss']['smooth_denom'],
    )

    # start training loop
    iteration_losses = {name: [] for name in loss_function.names+['sum']}
    update_losses = {name: [] for name in ['index']+loss_function.names+['sum']}
    validation_losses = {name: [] for name in ['index']+loss_function.names+['sum']}
    for i, (X, y) in tqdm(enumerate(train_dataloader)):
        
        index = i+1

        # update learning rate
        if index % settings.training['iterations_per_decay'] == 0:
            learning_rate *= settings.training['decay_factor']
            for g in optimizer.param_groups:
                g['lr'] = learning_rate
            message = (f'Iteration {str(index).zfill(4)}:   '
                       f'Learning rate set to {learning_rate:.1e}.')
            logger.info(message)

        # ---------------- TRAINING -------------------

        # bring the data to the correct device
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)

        # calculate loss
        losses = loss_function(y_pred, y)
        # correct for gradient accumulation and log values of loss components
        for name in losses:
            losses[name] = losses[name] / settings.training['iterations_per_update']
            iteration_losses[name].append(losses[name].item())
        
        # perform the backwards pass
        loss = sum(losses.values())
        iteration_losses['sum'].append(loss.item())
        loss.backward()

        # for debugging purposes
        if False: print(X.shape); print(y.shape)
        if False: rgb_image_viewer(X.cpu().detach())
        if False: image_viewer(y_pred.cpu().detach(), vmin=-1, vmax=1)
        if False: image_viewer(y.cpu().detach(), vmin=-1, vmax=1)

        if index % settings.training['iterations_per_update'] == 0:
            # update the network parameters and reset the gradient
            optimizer.step()
            optimizer.zero_grad() # set the gradient to 0 again

            # log values of loss components and combined
            message = f'Iteration {str(index).zfill(4)}:   Training loss:    '
            for name in iteration_losses:
                value = sum(
                    iteration_losses[name][-settings.training['iterations_per_update']:]
                )
                message += f'[{name}]: {value:0.3f},   '
                update_losses[name].append(value)
            update_losses['index'].append(index)
            logger.info(message)

        # --------------- VALIDATION ------------------
        # periodically evaluate on the validation set
        if index % settings.training['iterations_per_checkpoint'] == 0:

            # set the model in evaluation mode
            model.eval()

            losses_per_image = {name: [] for name in loss_function.names+['sum']}
            # deactivate autograd engine (backpropagation not required here)
            with torch.no_grad():
                for X, y in val_dataloader:

                    # bring the data to the correct device
                    X = X.to(device)
                    y = y.to(device)
                    y_pred = model(X)

                    # for debugging purposes
                    if False: rgb_image_viewer(X.cpu())
                    if False: image_viewer(y_pred.cpu(), vmin=-1, vmax=1)
                    if False: image_viewer(y.cpu(), vmin=-1, vmax=1)

                    # calculate loss
                    losses = loss_function(y_pred, y)
                    for name in losses:
                        losses_per_image[name].append(losses[name].item())
                    losses_per_image['sum'].append(sum(losses.values()).item())

            # calculate the average loss over all images in the validation set
            for name in losses_per_image:
                validation_losses[name].append(
                    sum(losses_per_image[name])/len(losses_per_image[name]),
                )  

            # log values of validation loss components and combined
            message = f'Iteration {str(index).zfill(4)}:   Validation loss:  '
            for name in losses_per_image:
                value = validation_losses[name][-1]
                message += f'[{name}]: {value:0.3f},   '
            validation_losses['index'].append(index)
            logger.info(message+'\n ')

            # save model checkpoint
            torch.save({
                'iteration': index,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': validation_losses['sum'][-1],
                },
                output_folder / f'checkpoint_I{str(index).zfill(4)}.tar',
            )

            # save training and validation loss as excel file
            train_loss_df = pd.DataFrame.from_dict(update_losses)
            val_loss_df = pd.DataFrame.from_dict(validation_losses)

            # create a excel writer object
            with pd.ExcelWriter(output_folder / 'loss.xlsx') as writer:
                train_loss_df.to_excel(writer, sheet_name='Training', index=False)
                val_loss_df.to_excel(writer, sheet_name="Validation", index=False)

            # set the model to training mode
            model.train()

    # plot training and validation loss
    fig, ax = plt.subplots()

    # plot loss
    i = 0
    colors = ['royalblue', 'forestgreen', 'firebrick', 'gold', 'black']
    for name in update_losses:
        if name != 'index':
            ax.plot(update_losses['index'], update_losses[name], zorder=1,
                    color=colors[i], alpha=0.40, label=f'{name} (train)')
            ax.plot(validation_losses['index'], validation_losses[name], 
                    color=colors[i], zorder=2, label=f'{name} (val)')
            ax.scatter(validation_losses['index'], validation_losses[name], 
                    zorder=2, marker='o', facecolor='white', edgecolor=colors[i], 
                    linewidth=1.5, s=15)
            i += 1

    # change axis setup
    plt.xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_ylim(bottom=0)
    plt.xlim([-25, update_losses['index'][-1]+25])
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'loss.png'), dpi=300)