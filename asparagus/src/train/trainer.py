import time
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import torch

import numpy as np

from .. import data
from .. import settings
from .. import utils

from .optimizer import get_optimizer
from .scheduler import get_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['Trainer']


# ======================================
# NNP Model Trainer
# ======================================

class Trainer:
    """
    NNP model Trainer class
    """
    
    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        data_container: Optional[object] = None,
        model_calculator: Optional[object] = None,
        trainer_max_epochs: Optional[int] = None,
        trainer_properties_train: Optional[List[str]] = None,
        trainer_properties_metrics: Optional[Dict[str, str]] = None,
        trainer_properties_weights: Optional[Dict[str, float]] = None,
        trainer_optimizer: Optional[Union[str, object]] = None,
        trainer_optimizer_args: Optional[Dict[str, float]] = None,
        trainer_scheduler: Optional[Union[str, object]] = None,
        trainer_scheduler_args: Optional[Dict[str, float]] = None,
        trainer_ema: Optional[bool] = None,
        trainer_ema_decay: Optional[float] = None,
        trainer_max_gradient_norm: Optional[float] = None,
        trainer_save_interval: Optional[int] = None,
        trainer_validation_interval: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize NNP Trainer.

        Parameters
        ----------

        config: (str, dict, object)
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        data_container: callable object, optional
            Data container object of the reference data set. 
            If not provided, the data container will be initialized according 
            to config input.
        model_calculator: callable object, optional
            NNP model calculator to train matching training and validation
            data in the reference data set. If not provided, the model 
            calculator will be initialized according to config input.
        trainer_max_epochs: int, optional, default 10000
            Maximum number of training epochs
        trainer_properties_train: list, optional, default []
            Properties contributing to the prediction quality value.
            If the list is empty or None, all properties will be considered
            both predicted by the model calculator and provided by the
            reference data set.
        trainer_properties_metrics: dict, optional, default 'MSE' for all
            Quantification of the property prediction quality only for 
            properties in the reference data set.
            Can be given for each property individually and by keyword 'all' 
            for every property else wise.
        trainer_properties_weights: dict, optional, default {...}
            Weighting factors for the combination of single property loss 
            values to total loss value.
        trainer_optimizer: (str, object), optional, default 'AMSgrad'
            Optimizer class for the NNP model training
        trainer_optimizer_args: dict, optional, default {}
            Additional optimizer class arguments
        trainer_scheduler: (str, object), optional, default 'ExponentialLR'
            Learning rate scheduler class for the NNP model training
        trainer_scheduler_args: dict, optional, default {}
            Additional learning rate scheduler class arguments
        trainer_ema: bool, optional, default True
            Apply exponential moving average scheme for NNP model training
        trainer_ema_decay: float, optional, default 0.999
            Exponential moving average decay rate
        trainer_max_gradient_norm: float, optional, default 1000.0
            Maximum model parameter gradient norm to clip its step size.
        trainer_save_interval: int, optional, default 5
            Interval between epoch to save current and best set of model
            parameters.
        trainer_validation_interval: int, optional, default 5
            Interval between epoch to evaluate model performance on
            validation data.
        **kwargs: dict, optional
            Additional arguments

        Returns
        -------
        callable object
            NNP model trainer object
        """

        ##########################################
        # # # Check Calculator Trainer Input # # #
        ##########################################

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
        
        # Assign global arguments
        self.dtype = settings._global_dtype
        self.device = settings._global_device

        ################################
        # # # Check Data Container # # #
        ################################
        
        # Assign DataContainer
        if self.data_container is None:
            
            self.data_container = data.DataContainer(
                config=config,
                **kwargs)

        # Assign training, validation and test data loader
        self.data_train = self.data_container.train_loader
        self.data_valid = self.data_container.valid_loader
        self.data_test = self.data_container.test_loader
        
        # Get reference data properties
        self.data_properties = self.data_container.data_load_properties
        self.data_units = self.data_container.data_unit_properties
        
        ################################
        # # # Check NNP Calculator # # #
        ################################
        
        # Assign DataContainer
        if self.model_calculator is None:
            
            # Set global dropout rate value
            if config_train.get("trainer_dropout_rate") is not None:
                settings.set_global_rate(config_train.get(
                    "trainer_dropout_rate"))
                
            # Get property scaling guess from reference data to link
            # 'normalized' output to average reference data shift and
            # distribution width.
            if ('model_properties_scaling' not in kwargs.keys() or
                'model_properties_scaling' not in config):
                
                model_properties_scaling = (
                    self.data_container.get_property_scaling())
                
                config.update(
                    {'model_properties_scaling': model_properties_scaling})
                
            # Assign NNP calculator model
            self.model_calculator = model.get_calculator(
                config=config,
                **kwargs)
            
        # Get reference data properties
        self.model_properties = self.model_calculator.model_properties
        
        ####################################
        # # # Check Trained Properties # # #
        ####################################
        
        # If training properties 'trainer_properties_train' is empty,
        # consider all properties covered in reference data set and the
        # model calculator.
        if (not len(self.trainer_properties_train) or     
                self.trainer_properties_train is None):
            
            # Reinitialize training properties list
            self.trainer_properties_train = []
            
            # Iterate over model properties, check for property in reference 
            # data set and eventually add to training properties.
            for prop in self.model_properties:
                if prop in self.data_properties:
                    self.trainer_properties_train.append(prop)
                    

        # Else check training properties and eventually correct for
        # not covered properties in the reference data set or the
        # model calculator.
        else:
            
            # Iterate over training properties, check for property in reference 
            # data set and model calculator prediction and eventually remove
            # training property.
            for prop in self.trainer_properties_train:
                if not (prop in self.data_properties and 
                        prop in self.model_properties):
                    
                    self.trainer_properties_train.remove(prop)
                    logger.warning(
                        f"WARNING:\nProperty '{prop}' in " +
                        f"'trainer_properties_train' is not stored in the " +
                        f"reference data set and/or predicted by the model " + 
                        f"model calculator!\n" +
                        f"Property '{prop}' is removed from training " + 
                        f"property list.")
        
        # Check for default property metric and weight
        if self.trainer_properties_metrics.get('else') is None:
            self.trainer_properties_metrics['else'] = 'MSE'
        if self.trainer_properties_weights.get('else') is None:
            self.trainer_properties_weights['else'] = 1.0
        
        # Check training property metrics and weights by iterating over 
        # training properties and eventually complete values
        for prop in self.trainer_properties_train:
            
            if self.trainer_properties_metrics.get(prop) is None:
                self.trainer_properties_metrics[prop] = (
                    self.trainer_properties_metrics['else'])
            if self.trainer_properties_weights.get(prop) is None:
                self.trainer_properties_weights[prop] = (
                    self.trainer_properties_weights['else'])

        # Collect training property units
        self.trainer_units_train = {}
        for prop in self.trainer_properties_train:
            if self.data_units.get(prop) is None:
                self.trainer_units_train[prop] = "a.u."
            else:
                self.trainer_units_train[prop] = self.data_units.get(prop)

        # Show current training property status
        msg = "Property    Metric    Unit      Weight\n"
        msg += "-"*len(msg) + "\n"
        for prop in self.trainer_properties_train:
            msg += f"{prop:10s}  {self.trainer_properties_metrics[prop]:8s}  "
            msg += f"{self.trainer_units_train[prop]:8s}  "
            msg += f"{self.trainer_properties_weights[prop]: 6.1f}\n"
        logger.info("INFO:\n" + msg)

        #############################
        # # # Prepare Optimizer # # #
        #############################
        
        # Assign optimizer
        self.trainer_optimizer = get_optimizer(
            self.trainer_optimizer,
            self.model_calculator.parameters(),
            self.trainer_optimizer_args)
        
        #############################
        # # # Prepare Scheduler # # #
        #############################
        
        # Assign scheduler
        self.trainer_scheduler = get_scheduler(
            self.trainer_scheduler,
            self.trainer_optimizer,
            self.trainer_scheduler_args)
        
        #######################
        # # # Prepare EMA # # #
        #######################
        
        # Assign Exponential Moving Average model
        if self.trainer_ema:
            from torch_ema import ExponentialMovingAverage
            self.trainer_ema_model = ExponentialMovingAverage(
                self.model_calculator.parameters(),
                decay=self.trainer_ema_decay)
        
        ##################################
        # # # Load Latest Checkpoint # # #
        ##################################
        
        # Initialize file management tools
        self.summary_writer, self.best_dir, self.ckpt_dir = (
            utils.file_managment(config))
        
        # Load, if exists, lated model calculator and training state 
        # checkpoint file
        latest_checkpoint = utils.load_checkpoint(
            config.get('checkpoint_file'))

        if latest_checkpoint is not None:
            self.model_calculator.load_state_dict(
                latest_checkpoint['model_state_dict'])
            self.trainer_optimizer.load_state_dict(
                latest_checkpoint['optimizer_state_dict'])
            self.trainer_scheduler.load_state_dict(
                latest_checkpoint['scheduler_state_dict'])
            self.epoch_start = latest_checkpoint['epoch']
        else:
            self.trainer_epoch_start = 1


    def train(self, verbose=True):
        
        # Initialize training mode for calculator
        self.model_calculator.train()
        torch.set_grad_enabled(True)
        #torch.autograd.set_detect_anomaly(True)
        
        # Initialize best total loss value of validation reference data
        self.best_loss = None
        
        # Reset property metrics
        metrics_best = self.reset_metrics()
                
        # Define loss function
        loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        
        # Count number of training batches
        Nbatch_train = len(self.data_train)
        
        # Initialize training time estimation per epoch
        train_time_estimation = np.nan
        
        # Loop over epochs
        for epoch in torch.arange(
                self.trainer_epoch_start, self.trainer_max_epochs):
            
            # Start epoch train timer
            train_time_epoch_start = time.time()
            
            # Reset property metrics
            metrics_train = self.reset_metrics()
            
            # Loop over training batches
            for ib, batch in enumerate(self.data_train):
                
                # Start batch train timer
                train_time_batch_start = time.time()
                
                # Eventually show training progress
                if verbose:
                    utils.printProgressBar(
                        ib, Nbatch_train, 
                        prefix=f"Epoch {epoch: 5d}",
                        suffix=(
                            "Complete - Remaining Epoch Time: "
                            + f"{train_time_estimation: 4.1f} s     "
                            ),
                        length=42)

                # Reset optimizer gradients
                self.trainer_optimizer.zero_grad(set_to_none=True)

                # Predict model properties from data batch
                prediction = self.model_calculator(batch)
                
                # Compute total and single loss values for training properties
                metrics_batch = self.compute_metrics(
                    prediction, batch, loss_fn=loss_fn)
                loss = metrics_batch['loss']

                # Predict parameter gradients by backwards propagation
                loss.backward()
                
                # Clip parameter gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model_calculator.parameters(),
                    self.trainer_max_gradient_norm)
                
                # Update model parameters
                self.trainer_optimizer.step()
                
                # Apply Exponential Moving Average
                if self.trainer_ema:
                    self.trainer_ema_model.update()
                
                # Update average metrics
                self.update_metrics(metrics_train, metrics_batch)
                
                # End batch train timer
                train_time_batch_end = time.time()
                
                # Eventually update training batch time estimation
                if verbose:
                    train_time_batch = (
                        train_time_batch_end - train_time_batch_start)
                    if ib:
                        train_time_estimation = (
                            0.5*(train_time_estimation - train_time_batch)
                            + 0.5*train_time_batch*(Nbatch_train - ib - 1))
                    else:
                        train_time_estimation = (
                            train_time_batch*(Nbatch_train - 1))
                        
            # Increment scheduler step
            self.trainer_scheduler.step()
            
            # Stop epoch train timer
            train_time_epoch_end = time.time()
            train_time_epoch = train_time_epoch_end - train_time_epoch_start
            
            # Eventually show final training progress
            if verbose:
                utils.printProgressBar(
                    Nbatch_train, Nbatch_train,
                    prefix=f"Epoch {epoch: 5d}",
                    suffix=(
                        f"Done - Epoch Time: " +
                        f"{train_time_epoch: 4.1f} s, " +
                        f"Loss: {metrics_train['loss']: 4.4f}   "),
                    length=42)

            # Save current model each interval
            if not (epoch % self.trainer_save_interval):
                number_of_ckpt = epoch//self.trainer_save_interval
                utils.save_checkpoint(
                    self.ckpt_dir,
                    model=self.model_calculator, 
                    optimizer=self.trainer_optimizer,
                    scheduler=self.trainer_scheduler, 
                    epoch=epoch, 
                    name_of_ckpt=epoch)

            
            # Perform model validation each interval
            if not (epoch % self.trainer_validation_interval):
                
                # Reset property metrics
                metrics_valid = self.reset_metrics()
                
                # Loop over validation batches
                for batch in self.data_valid:
                    
                    # Predict model properties from data batch
                    prediction = self.model_calculator(batch)
                                        
                    # Compute total and single loss values for training
                    # properties
                    metrics_batch = self.compute_metrics(
                        prediction, batch, loss_fn=loss_fn, loss_only=False)
                    
                    # Update average metrics
                    self.update_metrics(metrics_valid, metrics_batch)
                    
                # Check for model improvement and save as best model eventually
                if (self.best_loss is None or 
                        metrics_valid['loss'] < self.best_loss):
                    
                    # Store best metrics
                    metrics_best = metrics_valid
                    
                    # Save model calculator state
                    utils.save_checkpoint(
                        self.best_dir,
                        model=self.model_calculator, 
                        optimizer=self.trainer_optimizer,
                        scheduler=self.trainer_scheduler, 
                        epoch=epoch,
                        best=True)
                    
                    # Write to training summary
                    for prop, value in metrics_best.items():
                        if utils.is_dictionary(value):
                            for metric, val in value.items():
                                self.summary_writer.add_scalar(
                                    prop + '_' + metric, 
                                    metrics_best[prop][metric], 
                                    global_step=epoch)
                        else:
                            self.summary_writer.add_scalar(
                                prop, metrics_best[prop], 
                                global_step=epoch)
                        
                    # Update best total loss value
                    self.best_loss = metrics_valid['loss']
                    
                # Print validation metrics summary
                msg = (
                    f"Summary Epoch: {epoch:d}/" +
                    f"{self.trainer_max_epochs:d}\n" +
                    f"  Loss   train / valid: " +
                    f" {metrics_train['loss']:.2E} /" +
                    f" {metrics_valid['loss']:.2E}" +
                    f"  Best Loss valid: {metrics_best['loss']:.2E}\n"
                    f"  Property Metrics (valid):\n")
                for prop in self.trainer_properties_train:
                    msg += (
                        f"    {prop:10s}  MAE (Best) / RMSE (Best): " + 
                        f" {metrics_valid[prop]['mae']:.2E}" +
                        f" ({metrics_best[prop]['mae']:.2E}) /" +
                        f" {np.sqrt(metrics_valid[prop]['mse']):.2E}" +
                        f" ({np.sqrt(metrics_best[prop]['mse']):.2E})" +
                        f" {self.trainer_units_train[prop]:s}\n")
                logger.info("INFO:\n" + msg)
                
            
    def predict_batch(self, batch):
        
        # Predict properties
        return self.model_calculator(
            batch['atoms_number'], 
            batch['atomic_numbers'], 
            batch['positions'], 
            batch['idx_i'], 
            batch['idx_j'], 
            batch['charge'], 
            batch['atoms_seg'],
            batch['pbc_offset'])


    def reset_metrics(self):
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Add loss total value
        metrics['loss'] = 0.0
        
        # Add data counter
        metrics['Ndata'] = 0
        
        # Add training property metrics
        for prop in self.trainer_properties_train:
            metrics[prop] = {
                'loss': 0.0,
                'mae': 0.0,
                'mse': 0.0}
            
        return metrics


    def update_metrics(
        self, 
        metrics: Dict[str, float],
        metrics_update: Dict[str, float], 
    ) -> Dict[str, float]:
        
        # Get data sizes and metric ratio
        Ndata = metrics['Ndata']
        Ndata_update = metrics_update['Ndata']
        fdata = float(Ndata)/float((Ndata + Ndata_update))
        fdata_update = 1. - fdata
        
        # Update metrics
        metrics['loss'] = (
            metrics['loss'] + metrics_update['loss'].detach().item())
        metrics['Ndata'] = metrics['Ndata'] + metrics_update['Ndata']
        for prop in self.trainer_properties_train:
            for metric in metrics_update[prop].keys():
                metrics[prop][metric] = (
                    fdata*metrics[prop][metric] 
                    + fdata_update*metrics_update[prop][metric].detach().item()
                    )
        
        return metrics
        
    def compute_metrics(
        self, 
        prediction: Dict[str, Any], 
        reference: Dict[str, Any], 
        loss_fn: Optional[object] = None,
        loss_only: Optional[bool] = True,
    ) -> Dict[str, float]:
        
        # Check loss function input
        if loss_fn is None:
            loss_fn = torch.nn.L1Loss(reduction="mean")
        
        # Initialize MAE calculator function if needed
        if not loss_only:
            mae_fn = torch.nn.L1Loss(reduction="mean")
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Add batch size
        metrics['Ndata'] = reference[
            self.trainer_properties_train[0]].size()[0]
        
        # Iterate over training properties
        for ip, prop in enumerate(self.trainer_properties_train):
            
            # Initialize single property metrics dictionary
            metrics[prop] = {}
            
            # Compute loss value
            metrics[prop]['loss'] = loss_fn(
                torch.flatten(prediction[prop]), 
                torch.flatten(reference[prop]))
            
            # Weigth and add to total loss
            if ip:
                metrics['loss'] = metrics['loss'] + (
                    self.trainer_properties_weights[prop]
                    *metrics[prop]['loss'])
            else:
                metrics['loss'] = (
                    self.trainer_properties_weights[prop]
                    *metrics[prop]['loss'])
            
            # Compute MAE and MSE if requested
            if not loss_only:
                metrics[prop]['mae'] = mae_fn(
                    torch.flatten(prediction[prop]), 
                    torch.flatten(reference[prop]))
                metrics[prop]['mse'] = metrics[prop]['mae']**2
            
        return metrics