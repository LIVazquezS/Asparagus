import time
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import torch

import numpy as np

from .. import data
from .. import settings
from .. import utils
from .. import model

from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .tester import Tester

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
        trainer_restart: Optional[int] = None,
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
        trainer_evaluate_testset: Optional[bool] = None,
        trainer_max_checkpoints: Optional[int] = None,
        trainer_store_neighbor_list: Optional[bool] = None,
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
        trainer_restart: bool, optional, default False
            Restart the model training from state in config['model_directory']
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
        trainer_evaluate_testset: bool, optional, default True
            Each validation interval and in case of a new best loss function,
            apply Tester class on the test set.
        trainer_max_checkpoints: int, optional, default 50
            Maximum number of checkpoint files stored before deleting the
            oldest ones up to the number threshold.
        trainer_store_neighbor_list: bool, optional, default True
            Store neighbor list parameter in the database file instead of
            computing in situ.
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

        # Assign training and validation data loader
        self.data_train = self.data_container.train_loader
        self.data_valid = self.data_container.valid_loader

        # Get reference data properties
        self.data_properties = self.data_container.data_load_properties
        self.data_units = self.data_container.data_unit_properties

        ################################
        # # # Check NNP Calculator # # #
        ################################

        # Assign NNP calculator model
        if self.model_calculator is None:

            # Set global dropout rate value
            if config.get("trainer_dropout_rate") is not None:
                settings.set_global_rate(config.get(
                    "trainer_dropout_rate"))

            # Get property scaling guess from reference data to link
            # 'normalized' output to average reference data shift and
            # distribution width.
            if (
                    'model_properties_scaling' not in kwargs.keys()
                    or 'model_properties_scaling' not in config
            ):
                model_properties_scaling = (
                    self.data_container.get_property_scaling())

                config.update(
                    {'model_properties_scaling': model_properties_scaling})

            self.model_calculator = model.get_calculator(
                config=config,
                **kwargs)

        # Get model properties
        self.model_properties = self.model_calculator.model_properties
        self.model_units = self.model_calculator.model_unit_properties

        ######################################
        # # # Check Model and Data Units # # #
        ######################################

        # Check model units
        self.model_units = self.check_model_units()

        # Get Model to reference data property unit conversion factors
        self.model2data_unit_conversion = {}
        for prop, unit in self.model_units.items():
            self.model2data_unit_conversion[prop], _ = utils.check_units(
                unit, self.data_units.get(prop))

        # Show assigned property units
        message = (
            "INFO:\nModel property units:\n"
            + " Property Label | Model Unit     | Data Unit      |"
            + " Conversion Fac.\n"
            + "-"*17*4
            + "\n")
        for prop, unit in self.model_units.items():
            message += (
                f" {prop:<16s} {unit:<16s} {self.data_units[prop]:<16s}"
                + f"{self.model2data_unit_conversion[prop]:11.9e}\n")
        logger.info(message)

        # Assign potentially new property units to the model
        self.model_calculator.set_unit_properties(self.model_units)

        ######################################
        # # # Set Model Property Scaling # # #
        ######################################

        # Get property scaling guess from reference data to link
        # 'normalized' output to average reference data shift and
        # distribution width.
        model_properties_scaling = (
            self.data_container.get_property_scaling())

        # Convert scaling from reference data units to model units (1/conv)
        for prop, item in model_properties_scaling.items():
            model_properties_scaling[prop] = (
                np.array(item)/self.model2data_unit_conversion[prop])

        # Set current model property scaling
        self.model_calculator.set_property_scaling(
            model_properties_scaling)

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
                        "'trainer_properties_train' is not stored in the " +
                        "reference data set and/or predicted by the model " +
                        "model calculator!\n" +
                        f"Property '{prop}' is removed from training " +
                        "property list.\n")

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

        # Show current training property status
        msg = "Property    Metric    Unit      Weight\n"
        msg += "-"*len(msg) + "\n"
        for prop in self.trainer_properties_train:
            msg += f"{prop:10s}  {self.trainer_properties_metrics[prop]:8s}  "
            msg += f"{self.model_units[prop]:8s}  "
            msg += f"{self.trainer_properties_weights[prop]: 6.1f}\n"
        logger.info("INFO:\n" + msg)

        #############################
        # # # Prepare Optimizer # # #
        #############################

        # Assign model parameter optimizer
        self.trainer_optimizer = get_optimizer(
            self.trainer_optimizer,
            self.model_calculator.get_trainable_parameters(),
            self.trainer_optimizer_args)

        #############################
        # # # Prepare Scheduler # # #
        #############################

        # Assign learning rate scheduler
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

        ################################
        # # # Prepare File Manager # # #
        ################################

        # Initialize checkpoint file manager and summary writer
        self.filemanager = utils.FileManager(
            config,
            max_checkpoints=self.trainer_max_checkpoints)
        self.summary_writer = self.filemanager.writer

        ##########################
        # # # Prepare Tester # # #
        ##########################

        # Assign model prediction tester if test set evaluation is requested
        if self.trainer_evaluate_testset:
            self.tester = Tester(
                config,
                data_container=self.data_container,
                test_datasets='test',
                test_store_neighbor_list=trainer_store_neighbor_list)

        #############################
        # # # Save Model Config # # #
        #############################

        # Save a copy of the current model configuration in the model directory
        self.filemanager.save_config(config)

    def check_model_units(
        self,
        model_units: Optional[Dict[str, str]] = None,
    ):
        """
        Check the definition of the model units or assign units from the
        reference dataset
        """

        if model_units is None:
            model_units = self.model_units

        # If model units are not defined, take property units from dataset
        if model_units is None:
            model_units = self.data_units
            logger.info(
                "INFO:\nModel property units are not defined!\n"
                + "Property units from the reference dataset are assigned.\n")
        # If model units are defined , check completeness
        else:
            # Check positions unit
            if 'positions' not in model_units:
                model_units['positions'] = (
                    self.data_units.get('positions'))
                logger.warning(
                    "WARNING:\nModel property unit for 'positions' is not "
                    + "defined!\nPositions unit "
                    + f"{self.data_units.get['positions']:s} from the "
                    + "reference dataset is assigned.\n")
            # Check model property units
            for prop in self.model_properties:
                if prop not in model_units:
                    model_units[prop] = (
                        self.data_units.get(prop))
                    logger.warning(
                        f"WARNING:\nModel property unit for '{prop:s}' is not "
                        + f"is defined!\nUnit {self.data_units.get[prop]:s} "
                        + "from the reference dataset is assigned.\n")

        return model_units

    def train(self, verbose=True, debug=False):

        ####################################
        # # # Prepare Model and Metric # # #
        ####################################

        # Load, if exists, latest model calculator and training state
        # checkpoint file
        latest_checkpoint = self.filemanager.load_checkpoint(best=False)

        if latest_checkpoint is not None:
            self.model_calculator.load_state_dict(
                latest_checkpoint['model_state_dict'])
            self.trainer_optimizer.load_state_dict(
                latest_checkpoint['optimizer_state_dict'])
            self.trainer_scheduler.load_state_dict(
                latest_checkpoint['scheduler_state_dict'])
            self.trainer_epoch_start = latest_checkpoint['epoch'] + 1
        else:
            self.trainer_epoch_start = 1

        # Initialize training mode for calculator
        # (torch.nn.Module function to activate, e.g., parameter dropout)
        self.model_calculator.train()
        torch.set_grad_enabled(True)
        if debug:
            torch.autograd.set_detect_anomaly(True)

        # Initialize best total loss value of validation reference data
        self.best_loss = None

        # Reset property metrics
        metrics_best = self.reset_metrics()

        # Define loss function
        loss_fn = torch.nn.SmoothL1Loss(reduction='mean')

        # Count number of training batches
        Nbatch_train = len(self.data_train)

        # Initialize training time estimation per epoch
        train_time_estimation = np.nan

        # Set maximum model cutoff for neighbor list calculation
        self.data_train.init_neighbor_list(
            cutoff=self.model_calculator.model_interaction_cutoff,
            store=True)
        self.data_valid.init_neighbor_list(
            cutoff=self.model_calculator.model_interaction_cutoff,
            store=True)

        ##########################
        # # # Start Training # # #
        ##########################

        # Loop over epochs
        for epoch in torch.arange(
            self.trainer_epoch_start, self.trainer_max_epochs
        ):

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
                        "Done - Epoch Time: " +
                        f"{train_time_epoch: 4.1f} s, " +
                        f"Loss: {metrics_train['loss']: 4.4f}   "),
                    length=42)

            # Save current model each interval
            if not (epoch % self.trainer_save_interval):
                self.filemanager.save_checkpoint(
                    model=self.model_calculator,
                    optimizer=self.trainer_optimizer,
                    scheduler=self.trainer_scheduler,
                    epoch=epoch)

            # Perform model validation each interval
            if not (epoch % self.trainer_validation_interval):

                # Change to evaluation mode for calculator
                self.model_calculator.eval()

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

                # Change back to training mode for calculator
                self.model_calculator.train()

                # Check for model improvement and save as best model eventually
                if (
                    self.best_loss is None
                    or metrics_valid['loss'] < self.best_loss
                ):

                    # Store best metrics
                    metrics_best = metrics_valid

                    # Save model calculator state
                    self.filemanager.save_checkpoint(
                        model=self.model_calculator,
                        optimizer=self.trainer_optimizer,
                        scheduler=self.trainer_scheduler,
                        epoch=epoch,
                        best=True)

                    # Evaluation of the test set if requested
                    if self.trainer_evaluate_testset:
                        self.tester.test(
                            self.model_calculator,
                            test_directory=self.filemanager.best_dir,
                            test_plot_correlation=True,
                            test_plot_histogram=True,
                            test_plot_residual=True)

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

                    # Update best total loss     value
                    self.best_loss = metrics_valid['loss']

                # Print validation metrics summary
                msg = (
                    f"Summary Epoch: {epoch:d}/" +
                    f"{self.trainer_max_epochs:d}\n" +
                    "  Loss   train / valid: " +
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
                        f" {self.model_units[prop]:s}\n")
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
        metrics['Ndata'] = metrics['Ndata'] + metrics_update['Ndata']
        metrics['loss'] = (
            fdata*metrics['loss']
            + fdata_update*metrics_update['loss'].detach().item())
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
            mse_fn = torch.nn.MSELoss(reduction="mean")

        # Initialize metrics dictionary
        metrics = {}

        # Add batch size
        metrics['Ndata'] = reference['atoms_number'].size()[0]

        # Iterate over training properties
        for ip, prop in enumerate(self.trainer_properties_train):

            # Initialize single property metrics dictionary
            metrics[prop] = {}

            # Compute loss value per atom
            metrics[prop]['loss'] = loss_fn(
                torch.flatten(prediction[prop])
                * self.model2data_unit_conversion[prop],
                torch.flatten(reference[prop]))

            # Weight and add to total loss
            if ip:
                metrics['loss'] = metrics['loss'] + (
                    self.trainer_properties_weights[prop]
                    * metrics[prop]['loss'])
            else:
                metrics['loss'] = (
                    self.trainer_properties_weights[prop]
                    * metrics[prop]['loss'])

            # Compute MAE and MSE if requested
            if not loss_only:
                metrics[prop]['mae'] = mae_fn(
                    torch.flatten(prediction[prop])
                    * self.model2data_unit_conversion[prop],
                    torch.flatten(reference[prop]))
                metrics[prop]['mse'] = mse_fn(
                    torch.flatten(prediction[prop])
                    * self.model2data_unit_conversion[prop],
                    torch.flatten(reference[prop]))

        return metrics
