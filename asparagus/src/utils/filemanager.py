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

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = {}
        for arg, item in locals().items():

            # Skip 'config' argument and possibly more
            if arg in [
                    'self', 'config', 'config_update', 'kwargs', '__class__']:
                continue
            
            # Take argument from global configuration dictionary if not defined
            # directly
            if item is None:
                item = config.get(arg)

            # Set default value if the argument is not defined (None)
            if arg in settings._default_args.keys() and item is None:
                item = settings._default_args[arg]

            # Check datatype of defined arguments
            if arg in settings._dtypes_args.keys():
                match = utils.check_input_dtype(
                    arg, item, settings._dtypes_args, raise_error=True)

            # Append to update dictionary
            config_update[arg] = item

            # Assign as class parameter
            setattr(self, arg, item)

        # Update global configuration dictionary
        config.update(config_update)

        ###################################
        # # # Prepare Model Directory # # #
        ###################################
        
        # Take either defined model directory path or a generate a generic one
        if self.model_directory is None:
            self.model_directory = datetime.datetime.now().strftime(
                "%Y%m%d%H%M%S")
            config['model_directory'] = self.model_directory
        # I would prefer if we keep the specifications of the NN model in the name of the directory...LIVS
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
        # Create directory for tensorboardX/logs in previous physnet_torch version
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
    ):
        """
        Save model parameters and training state to checkpoint file.
        If 'best' True, save as best model checkpoint file.
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
        If 'best' True, load best model checkpoint file.
        if 'num_checkpoint' is None, load checkpoint file with highest index
        number.
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
        










#def file_managment(
    #config: Optional[Union[str, dict, object]] = None,
    #restart: Optional[bool] = False,
#):
    
    ## Get configuration object
    #config = settings.get_config(config)
    
    #if restart or config['restart']:
        
        ## Get and check model directory where the checkpoints are saved
        #directory = config['model_directory'] 
        #if not os.path.exists(directory):
            #raise FileNotFoundError(
                #f"Restart from model directory '{directory:s}' failed!"
                #+ "Directory does not exist.")
        
        #logs_dir = os.path.join(directory, 'logs')
        #best_dir = os.path.join(directory, 'best')
        #ckpt_dir = os.path.join(directory, 'checkpoints')
        
        ## Initialize training summary writer
        #writer = SummaryWriter(log_dir=logs_dir)
        
    #else:
        
        ## Take either deined model directory path or a generate a generic one
        #if config['model_directory'] is None:
            #directory = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            #config['model_directory'] = directory
        #else:
            #directory = config['model_directory']
        
        ## I would prefer if we keep the specifications of the NN model in the name of the directory...LIVS
        ## I see your point, but the commented version seems very PhysNet specific.
        ## Maybe add a __str__() function to the model which returns a model
        ## tag for the directory name.
        ##(
            ##datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            ##+ "_" + id_generator() + "_F" + str(config['input_n_atombasis'])
            ##+ "K" + str(config['input_nradialbasis']) + "b" + str(config['graph_n_blocks'])
            ##+ "a" + str(config['graph_n_residual_atomic'])
            ##+ "i" + str(config['graph_n_residual_interaction'])
            ##+ "o" + str(config['output_n_residual']) + "cut" + str(config['input_cutoff_descriptor'])
            ##+ "e" + str(config['model_electrostatic']) + "d" + str(config['model_dispersion']) + "r" + str(config['model_repulsion']))

        ## Prepare model directory
        #best_dir = os.path.join(directory, 'best')
        #logs_dir = os.path.join(directory, 'logs')
        #ckpt_dir = os.path.join(directory, 'checkpoints')
        #create_model_directory(directory, ckpt_dir, logs_dir, best_dir)

        ## Initialize training summary writer
        #writer = SummaryWriter(log_dir=logs_dir)
        
    #return writer, best_dir, ckpt_dir


#def id_generator(
    #size=8, 
    #chars=(
        #string.ascii_uppercase
        #+ string.ascii_lowercase
        #+ string.digits)
    #):
    #"""
    #Generate an (almost) unique id for the training session
    #"""
    
    #return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

#def create_model_directory(
    #directory: str, 
    #ckpt_dir: str, 
    #logs_dir: str, 
    #best_dir: str,
#):
    #"""
    #Create folders for checkpoints and tensorboardX
    #"""
    
    ## Create model directory
    #if not os.path.exists(directory):
        #os.makedirs(directory)
    ## Create directory for model parameter checkpoints
    #if not os.path.exists(ckpt_dir):
        #os.makedirs(ckpt_dir)
    ## Create directory for tensorboardX/logs in previous physnet_torch version
    #if not os.path.exists(logs_dir):
        #os.makedirs(logs_dir)
    ## Create directory for best model checkpoints
    #if not os.path.exists(best_dir):
        #os.makedirs(best_dir)


#def save_checkpoint(
    #directory: str, 
    #model, 
    #optimizer, 
    #scheduler, 
    #epoch, 
    #best=False
    #num_checkpoint=None, 
#):
    #"""
    #Save model parameters and training state to checkpoint file
    #"""
    
    ## Current model training state
    #state = {
        #'model_state_dict': model.state_dict(),
        #'optimizer_state_dict': optimizer.state_dict(),
        #'scheduler_state_dict': scheduler.state_dict(),
        #'epoch': epoch, 
        #}
    
    ## Checkpoint file name
    #if best:
        #ckpt_name = os.path.join(directory, 'best_model.pt')
    #elif num_checkpoint is None:
        #ckpt_name = os.path.join(directory, f'model_{epoch:d}.pt')
    #else:
        #if utils.is_integer(num_checkpoint):
            #ckpt_name = os.path.join(directory, f'model_{num_checkpoint:d}.pt')
        #else:
            #raise ValueError(
                #"Checkpoint file index number 'num_checkpoint' is not an"
                #+ "integer!")

    ## Write checkpoint file
    #torch.save(state, ckpt_name)


#def load_checkpoint(
    #path
#):
    
    
    #if path is not None:
        #checkpoint = torch.load(path)
        #return checkpoint
    #else:
        #return None
    

#def save_checkpoint_Trainer(                                                                                                  
    #calculator: object,
    #optimizer: object,
    #scheduler: object, 
    #epoch: int, 
    #name_of_ckpt: Optional[str] = '', 
    #best: Optional[bool] = False,
#):
    
    ## Store current state dictionary data
    #state = {
        #'model_state_dict': model.state_dict(),
        #'optimizer_state_dict': optimizer.state_dict(),
        #'scheduler_state_dict': scheduler.state_dict(),
        #'epoch': epoch}                                                                                               
    
    #if best:
        #path = os.path.join(best_dir, best_file)
    #else:
        #path = os.path.join(checkpoint_dir, checkpoint_file.format(epoch))

    ## Save checkpoint file
    #torch.save(state, path)


#def load_checkpoint_Trainer(
    #ckpt_file: Optional[str] = '',
    #best: Optional[bool] = False,
#):
    
    #if best:
        #checkpoint = torch.load(os.path.join(best_dir, best_file))                                                    
    #elif len(ckpt_file):
        #checkpoint = torch.load(ckpt_file)                                                                            
        ##checkpoint = torch.load(os.path.join(checkpoint_dir, ckpt_file))
    #else:
        #checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_file))                                        
    
    #return checkpoint

