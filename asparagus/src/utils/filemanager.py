# This should manage checkpoint creation and loading writer to tensorboardX
import os
import re
import string
import random
import datetime
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import torch
from tensorboardX import SummaryWriter

from .. import settings
from .. import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['FileManager']


class FileManager():
    """
    File manager for model files
    """

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        model_directory: Optional[str] = None,
        max_checkpoints: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize file manager class.

        Parameters
        ----------

        config: (str, dict, object)
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        model_directory: str, optional, default config['model_directory']
            Model directory that contains checkpoint and log files.

        Returns
        -------
        callable object
            Mode FileManager object
        """

        ###################################
        # # # Check FileManager Input # # #
        ###################################

        # Get configuration object
        config = settings.get_config(config)

        # Get model directory from input or config dictionary
        if model_directory is None:
            self.model_directory = config.get('model_directory')
        elif utils.is_string(model_directory):
            self.model_directory = model_directory
        else:
            raise ValueError(
                "Input for model directory is not as expected!\n"
                + "A directory path in form of a string is required.")

        # Get number of maximum checkpoints files from input or config dict.
        if max_checkpoints is None:
            self.max_checkpoints = None
        elif utils.is_integer(max_checkpoints):
            self.max_checkpoints = max_checkpoints
        else:
            raise ValueError(
                "Input for maximum checkpoint files is not as expected!\n"
                + "An integer number is required.")

        ###################################
        # # # Prepare Model Directory # # #
        ###################################

        # Take either defined model directory path or a generate a generic one
        if self.model_directory is None:
            self.model_directory = datetime.datetime.now().strftime(
                "%Y%m%d%H%M%S")
            config['model_directory'] = self.model_directory
        # I would prefer if we keep the specifications of the NN model in the
        # name of the directory...LIVS
        # I see your point, but the commented version seems very PhysNet specific.
        # Maybe add a __str__() function to the model which returns a model
        # tag for the directory name.
        #(
            #datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            #+ "_" + id_generator() + "_F" + str(config['input_n_atombasis'])
            #+ "K" + str(config['input_nradialbasis']) + "b" + str(config['graph_n_blocks'])
            #+ "a" + str(config['graph_n_residual_atomic'])
            #+ "i" + str(config['graph_n_residual_interaction'])
            #+ "o" + str(config['output_n_residual']) + "cut" + str(config['input_cutoff_descriptor'])
            #+ "e" + str(config['model_electrostatic']) + "d" + str(config['model_dispersion']) + "r" + str(config['model_repulsion']))

        # Prepare model subdirectory paths
        self.best_dir = os.path.join(self.model_directory, 'best')
        self.ckpt_dir = os.path.join(self.model_directory, 'checkpoints')
        self.logs_dir = os.path.join(self.model_directory, 'logs')

        # Check existence of the directories
        self.create_model_directory()

        # Initialize training summary writer
        self.writer = SummaryWriter(log_dir=self.logs_dir)

    def create_model_directory(self):
        """
        Create folders for checkpoints and tensorboardX
        """

        # Create model directory
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        # Create directory for best model checkpoints
        if not os.path.exists(self.best_dir):
            os.makedirs(self.best_dir)
        # Create directory for model parameter checkpoints
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        # Create directory for tensorboardX/logs
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

    def save_checkpoint(
        self,
        model: object,
        optimizer: object,
        scheduler: object,
        epoch: int,
        best: Optional[bool] = False,
        num_checkpoint: Optional[int] = None,
        max_checkpoints: Optional[int] = None,
    ):
        """
        Save model parameters and training state to checkpoint file.

        Parameters
        ----------

        model: object
            Torch calculator model
        optimizer: object
            Torch optimizer
        scheduler: object
            Torch scheduler
        best: bool, optional, default False
            If True, save as best model checkpoint file.
        num_checkpoint: int, optional, default None
            Alternative checkpoint index other than epoch.
        max_checkpoints: int, optional, default 100
            Maximum number of checkpoint files. If the threshold is reached and
            a checkpoint of the best model (best=True) or specific number
            (num_checkpoint is not None), respectively many checkpoint files
            with the lowest indices will be deleted.
        """

        # Current model training state
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            }

        # Checkpoint file name
        if best:
            ckpt_name = os.path.join(self.best_dir, 'best_model.pt')
        elif num_checkpoint is None:
            ckpt_name = os.path.join(
                self.ckpt_dir, f'model_{epoch:d}.pt')
            self.check_max_checkpoints(max_checkpoints)
        else:
            if utils.is_integer(num_checkpoint):
                ckpt_name = os.path.join(
                    self.ckpt_dir, f'model_{num_checkpoint:d}.pt')
            else:
                raise ValueError(
                    "Checkpoint file index number 'num_checkpoint' is not "
                    + "an integer!")

        # Write checkpoint file
        torch.save(state, ckpt_name)

    def load_checkpoint(
        self,
        best: Optional[bool] = False,
        num_checkpoint: Optional[int] = None,
    ):
        """
        Load model parameters and training state from checkpoint file.

        Parameters
        ----------

        best: bool, optional, default False
            If True, load best model checkpoint file.
        num_checkpoint: int, optional, default None
            if None, load checkpoint file with highest index number.
        """

        if best:

            ckpt_name = os.path.join(self.best_dir, 'best_model.pt')

            # Check existence
            if not os.path.exists(ckpt_name):
                raise FileNotFoundError(
                    f"Checkpoint file '{ckpt_name}' for best model "
                    + "does not exist!")

        elif num_checkpoint is None:

            # Get highest index checkpoint file
            ckpt_max = -1
            for ckpt_file in os.listdir(self.ckpt_dir):
                ckpt_num = re.findall("model_(\d+).pt", ckpt_file)
                ckpt_num = (int(ckpt_num[0]) if ckpt_num else -1)
                if ckpt_max < ckpt_num:
                    ckpt_max = ckpt_num

            # If no checkpoint files available return None
            if ckpt_max < 0:
                return None
            else:
                ckpt_name = os.path.join(
                    self.ckpt_dir, f'model_{ckpt_max:d}.pt')

        else:

            ckpt_name = os.path.join(
                self.ckpt_dir, f'model_{num_checkpoint:d}.pt')

            # Check existence
            if not os.path.exists(ckpt_name):
                raise FileNotFoundError(
                    f"Checkpoint file '{ckpt_name}' of index "
                    + f"{num_checkpoint:d} does not exist!")

        # Load checkpoint
        checkpoint = torch.load(ckpt_name)

        return checkpoint

    def check_max_checkpoints(
        self,
        max_checkpoints: Optional[int] = None,
    ):
        """
        Check number of checkpoint files and in case of exceeding the
        maximum checkpoint threshold, delete the ones with lowest indices.
        """
        
        # Skip in checkpoint threshold is None
        if max_checkpoints is None and self.max_checkpoints is None:
            return
        elif max_checkpoints is None:
            max_checkpoints = self.max_checkpoints

        # Gather checkpoint files
        num_checkpoints = []
        for ckpt_file in os.listdir(self.ckpt_dir):
            ckpt_num = re.findall("model_(\d+).pt", ckpt_file)
            if ckpt_num:
                num_checkpoints.append(int(ckpt_num[0]))
        num_checkpoints = sorted(num_checkpoints)

        # Delete in case the lowest checkpoint files
        if len(num_checkpoints) >= max_checkpoints:
            # Get checkpoints to delete
            if max_checkpoints > 0:
                remove_num_checkpoints = num_checkpoints[:-max_checkpoints]
            # If max_checkpoints is zero (or less), delete everyone
            else:
                remove_num_checkpoints = num_checkpoints
            # Delete checkpoint files
            for ckpt_num in remove_num_checkpoints:
                ckpt_name = os.path.join(
                    self.ckpt_dir, f'model_{ckpt_num:d}.pt')
                os.remove(ckpt_name)

        return

    def save_config(
        self,
        config: object,
        max_backup: Optional[int] = 10,
    ):
        """
        Save config object in current model directory with the default file
        name. If such file already exist, backup the old one and overwrite.

        Parameters
        ----------

        config: object
            Config class object
        max_backup: optional, int, default 100
            Maximum number of backup config files
        """

        # Config file path
        config_file = os.path.join(
            self.model_directory, settings._default_args['config_file'])

        # Check for old config files
        if os.path.exists(config_file):

            default_file = settings._default_args['config_file']

            # Check for backup config files
            list_backups = []
            for f in os.listdir(self.model_directory):
                num_backup = re.findall(
                    "(\d+)_" + default_file, f)
                num_backup = (int(num_backup[0]) if num_backup else -1)
                if num_backup >= 0:
                    list_backups.append(num_backup)
            list_backups = sorted(list_backups)

            # Rename old config file
            if len(list_backups):
                backup_file = os.path.join(
                    self.model_directory,
                    f"{list_backups[-1] + 1:d}_" + default_file)
            else:
                backup_file = os.path.join(
                    self.model_directory,
                    f"{1:d}_" + default_file)
            os.rename(config_file, backup_file)

            # If maximum number of back file reached, delete the oldest
            if len(list_backups) >= max_backup:
                for num_backup in list_backups[:-(max_backup - 1)]:
                    backup_file = os.path.join(
                        self.model_directory,
                        f"{num_backup:d}_" + default_file)
                    os.remove(backup_file)

        # Dump config in file path
        config.dump(config_file=config_file)
