import os
import sys

import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import torch
#from numpy import dtype

from .. import data
from .. import settings
from .. import utils
from .. import model

# These packages are required for all functions of plotting and analysing
# the model.
try: 
    import pandas as pd
except ImportError:
    raise UserWarning(
        "You need to install pandas to use all "
        + "plotting and analysis functions")
try: 
    import matplotlib.pyplot as plt
except ImportError:
    raise UserWarning(
        "You need to install matplotlib to use all "
        + "plotting and analysis functions")
try:
    import seaborn as sns
except ImportError:
    raise UserWarning(
        "You need to install seaborn to use all "
        + "plotting and analysis functions")
try: 
    from scipy import stats
except ImportError:
    raise UserWarning(
        "You need to install scipy to use all plotting and analysis functions")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['Tester']

class Tester:
    """
    Model Prediction Tester Class
    """
    
    def __init__(self,
        config: Optional[Union[str, dict, object]] = None,
        data_container: Optional[object] = None,
        test_datasets: Optional[Union[str, List[str]]] = None,
        test_properties: Optional[Union[str, List[str]]] = None,
        **kwargs
    ):
        """
        Initialize model tester.

        Parameters
        ----------

        config: (str, dict, object)
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        data_container: callable object, optional
            Data container object of the reference test data set. 
            If not provided, the data container will be initialized according 
            to config input.
        test_datasets: (str, list(str)) optional, default ['test']
            A string or list of strings to define the data sets ('train', 
            'valid', 'test') of which the evaluation will be performed.
            By default it is just the test set of the data container object.
            Inputs 'full' or 'all' requests the evaluation of all sets.
        test_properties: (str, list(str)), optional, default None
            Model properties to evaluate which must be available in the 
            model prediction and the reference test data set. If None, all 
            model properties will be evaluated if available in the test set.
        """
        
        ####################################
        # # # Check Model Tester Input # # #
        ####################################

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
        
        ################################
        # # # Check Data Container # # #
        ################################

        # Assign DataContainer and test data loader 
        if self.data_container is None:
            self.data_container = data.DataContainer(
                config=config,
                **kwargs)
        
        # Prepare list of data set definition for evaluation
        if utils.is_string(self.test_datasets):
            self.test_datasets = [self.test_datasets]
        if 'full' in self.test_datasets or 'all' in self.test_datasets:
            self.test_datasets = self.data_container.get_datalabels()
        
        # Collect requested data loader
        self.test_data = {
            label: self.data_container.get_dataloader(label)
            for label in self.test_datasets}

        # Get reference data properties
        self.data_properties = self.data_container.data_load_properties
        self.data_units = self.data_container.data_unit_properties
        
        #################################
        # # # Check Test Properties # # #
        #################################
        
        self.test_properties = self.check_test_properties()
        
        
    def check_test_properties(
        self, 
        test_properties: Optional[Union[str, List[str]]] = None,
        data_properties: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Check availability of 'test_properties' in 'data_properties' and 
        return eventually corrected test_properties as list.
        """
        
        # Check input
        if test_properties is None:
            test_properties = self.test_properties
        if data_properties is None:
            data_properties = self.data_properties

        # If not defined, take all reference properties, else check 
        # availability
        if test_properties is None:
            test_properties = data_properties
        else:
            if utils.is_string(test_properties):
                test_properties = [test_properties]
            checked_properties = []
            for prop in test_properties:
                if prop not in data_properties:
                    logger.warning(
                        f"WARNING:\nRequested property '{prop}' in " +
                        "'test_properties' for the model evaluation " +
                        "is not avaible in the reference data set and " + 
                        "will be ignored!")
                else:
                    checked_properties.append(prop)
            test_properties = checked_properties
            
        return test_properties


    def test(
        self,
        model_calculator: object,
        test_properties: Optional[Union[str, List[str]]] = None,
        test_directory: Optional[str] = '.',
        test_plot_correlation: Optional[bool] = True,
        test_plot_histogram: Optional[bool] = False,
        test_plot_residual: Optional[bool] = False,
        test_plot_format: Optional[str] = 'pdf',
        test_plot_dpi: Optional[int] = 300,
        test_save_csv: Optional[bool] = False,
        test_csv_file: Optional[str] = 'model_prediction.csv',
        test_save_npz: Optional[bool] = False,
        test_npz_file: Optional[str] = 'model_prediction.npz',
        test_scale_per_atom: Optional[Union[str, List[str]]] = ['energy'],
        verbose: Optional[bool] = True,
    ):
        """
        Initialize model tester.

        Parameters
        ----------

        model_calculator: callable object
            NNP model calculator to predict test properties. The prediction
            are done with the given state of parametrization, no checkpoint
            files will be loaded.
        test_properties: (str, list(str)), optional, default None
            Model properties to evaluate which must be available in the 
            model prediction and the reference test data set. If None, model
            properties will be evaluated as initialized.
        test_directory: str, optional, default '.'
            Directory to store evaluation graphics and data.
        test_plot_correlation: bool, optional, default True
            Show evaluation in property correlation plots 
            (x-axis: reference property; y-axis: predicted property).
        test_plot_histogram: bool, optional, default False
            Show prediction error spread in histogram plots.
        test_plot_residual: bool, optional, default False
            Show evaluation in residual plots.
            (x-axis: reference property; y-axis: prediction error).
        test_plot_format: str, optional, default 'pdf'
            Plot figure format (for options see matplotlib.pyplot.savefig()).
        test_plot_format: int, optional, default 300
            Plot figure dpi.
        test_save_csv: bool, optional, default False
            Save all model prediction results in a csv file.
        test_csv_file: str, optional, default 'model_prediction.csv'
            Name tag of the csv file. The respective data set label will be
            added as prefix to the tag ("{label:s}_{test_csv_file:s}").
        test_save_npz: bool, optional, default False
            Save all model prediction results in a binary npz file.
        test_npz_file: str, optional, default 'model_prediction.npz'
            Name tag of the npz file. The respective data set label will be
            added as prefix to the tag ("{label:s}_{test_npz_file:s}").
        test_scale_per_atom: (str list(str), optional, default ['energy']
            List of properties where the results will be scaled by the number 
            of atoms in the particular system.
        verbose: bool, optional, default True
            Print test metrics.
        """
        
        #################################
        # # # Check Test Properties # # #
        #################################
        
        # Get model properties
        if hasattr(model_calculator, "model_properties"):
            model_properties = model_calculator.model_properties
        else:
            raise AttributeError(
                "Model calculator has no 'model_properties' attribute")
        
        # Check test properties if defined or take initialized ones
        if test_properties is None:
            test_properties = self.test_properties
        else:
            test_properties = self.check_test_properties(
                test_properties,
                self.data_properties)
            
        # Compare model properties with test properties and store properties
        # to evaluate
        eval_properties = []
        for prop in test_properties:
            if prop in model_properties:
                eval_properties.append(prop)
            else:
                logger.warning(
                    f"WARNING:\nRequested property '{prop}' in " +
                    "'test_properties' is not predicted by the " +
                    "model calculator and will be ignored!")
        
        ##############################
        # # # Compute Properties # # #
        ##############################
        
        # Change to evaluation mode for calculator
        model_calculator.eval()
        
        # Loop over all requested data set
        for label, data in self.test_data.items():

            # Prepare dictionary for property values and number of atoms per 
            # system
            test_prediction = {prop: [] for prop in test_properties}
            test_reference = {prop: [] for prop in test_properties}
            test_prediction['atoms_number'] = []

            # Reset property metrics
            metrics_test = self.reset_metrics(eval_properties)

            # Loop over data batches
            for batch in data:

                # Predict model properties from data batch
                prediction = model_calculator(batch)

                # Compute metrics for test properties
                metrics_batch = self.compute_metrics(
                    prediction, batch, eval_properties)
                
                # Update average metrics
                self.update_metrics(
                    metrics_test, metrics_batch, eval_properties)
                
                # Store prediction and reference data
                Nsys = len(batch['atoms_number'])
                for prop in eval_properties:
                    data_prediction = prediction[prop].detach().numpy()
                    data_reference = batch[prop].detach().numpy()
                    if data_prediction.shape[0] == len(batch['atoms_seg']):
                        data_prediction = [
                            list(data_prediction[isys]) 
                            for isys in range(Nsys)]
                        data_reference = [
                            list(data_reference[isys]) for isys in range(Nsys)]
                    elif data_prediction.shape[0] == len(batch['pairs_seg']):
                        data_prediction = [
                            list(data_prediction[isys]) 
                            for isys in range(Nsys)]
                        data_reference = [
                            list(data_reference[isys]) for isys in range(Nsys)]
                    test_prediction[prop] += list(data_prediction)
                    test_reference[prop] += list(data_reference)
                
                # Store atom numbers
                test_prediction['atoms_number'] += list(batch['atoms_number'])

            # Print metrics
            if verbose:
                self.print_metric(metrics_test, eval_properties, label)

            ###########################
            # # # Save Properties # # #
            ###########################
            
            # Save test prediction to files 
            #TODO how to save inhomogeneous data to files?
            #if test_save_csv:
                #csv_file = os.path.join(test_directory, test_csv_file)
                #if self.is_imported("pandas"):
                    #df = pd.DataFrame(test_prediction)
                    #df.to_csv(csv_file, index=True)
                #else:
                    #logger.warning(
                        #"WARNING:\nModule 'pandas' is not available. " +
                        #"Test properties are not written to a csv file!")

            ###########################
            # # # Plot Properties # # #
            ###########################
            
            # Check input for scaling per atom and prepare atom number scaling
            if utils.is_string(test_scale_per_atom):
                test_scale_per_atom = [test_scale_per_atom]
            test_property_scaling = {}
            for prop in eval_properties:
                if prop in test_scale_per_atom:
                    test_property_scaling[prop] = (
                        1./np.array(
                            test_prediction['atoms_number'], dtype=float)
                        )
                else:
                    test_property_scaling[prop] = None

            # Plot correlation between model and reference properties
            if test_plot_correlation:
                for prop in eval_properties:
                    self.plot_correlation(
                        label,
                        prop,
                        self.plain_data(test_prediction[prop]),
                        self.plain_data(test_reference[prop]),
                        self.data_units[prop],
                        metrics_test[prop],
                        test_property_scaling[prop],
                        test_directory,
                        test_plot_format,
                        test_plot_dpi,
                        )
                    
            # Plot histogram of the prediction error
            if test_plot_histogram:
                for prop in eval_properties:
                    self.plot_histogram(
                        label,
                        prop,
                        self.plain_data(test_prediction[prop]),
                        self.plain_data(test_reference[prop]),
                        self.data_units[prop],
                        metrics_test[prop],
                        test_directory,
                        test_plot_format,
                        test_plot_dpi
                        )

            # Plot histogram of the prediction error
            if test_plot_residual:
                for prop in eval_properties:
                    self.plot_residual(
                        label,
                        prop,
                        self.plain_data(test_prediction[prop]),
                        self.plain_data(test_reference[prop]),
                        self.data_units[prop],
                        metrics_test[prop],
                        test_property_scaling[prop],
                        test_directory,
                        test_plot_format,
                        test_plot_dpi
                        )

        # Change back to training mode for calculator 
        model_calculator.train()

        return


    @staticmethod
    def is_imported(module):
        """
        Check if a module is imported.
        """

        return module in sys.modules


    def reset_metrics(
        self, 
        test_properties: List[str] = None
    ) -> Dict[str, float]:
        
        # Check input
        if test_properties is None:
            test_properties = self.test_properties
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Add data counter
        metrics['Ndata'] = 0
        
        # Add training property metrics
        for prop in test_properties:
            metrics[prop] = {
                'mae': 0.0,
                'mse': 0.0}
            
        return metrics


    def compute_metrics(
        self, 
        prediction: Dict[str, Any], 
        reference: Dict[str, Any], 
        test_properties: List[str] = None
    ) -> Dict[str, float]:
        
        # Check input
        if test_properties is None:
            test_properties = self.test_properties
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Add batch size
        metrics['Ndata'] = reference[
            test_properties[0]].size()[0]
        
        # Iterate over test properties
        mae_fn = torch.nn.L1Loss(reduction="mean")
        mse_fn = torch.nn.MSELoss(reduction="mean")
        for ip, prop in enumerate(test_properties):
            
            # Initialize single property metrics dictionary
            metrics[prop] = {}
            
            # Compute MAE and MSE
            metrics[prop]['mae'] = mae_fn(
                torch.flatten(prediction[prop]), 
                torch.flatten(reference[prop]))
            metrics[prop]['mse'] = mse_fn(
                torch.flatten(prediction[prop]), 
                torch.flatten(reference[prop]))
            
        return metrics


    def update_metrics(
        self, 
        metrics: Dict[str, float],
        metrics_update: Dict[str, float], 
        test_properties: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        
        # Check property input
        if test_properties is None:
            test_properties = self.test_properties
        
        # Get data sizes and metric ratio
        Ndata = metrics['Ndata']
        Ndata_update = metrics_update['Ndata']
        fdata = float(Ndata)/float((Ndata + Ndata_update))
        fdata_update = 1. - fdata
        
        # Update metrics
        metrics['Ndata'] = metrics['Ndata'] + metrics_update['Ndata']
        for prop in test_properties:
            for metric in metrics_update[prop].keys():
                metrics[prop][metric] = (
                    fdata*metrics[prop][metric] 
                    + fdata_update*metrics_update[prop][metric].detach().item()
                    )
        
        return metrics


    def print_metric(
        self, 
        metrics: Dict[str, float],
        test_properties: Optional[List[str]] = None,
        test_label: Optional[str] = '',
    ):

        # Check property and label input
        if test_properties is None:
            test_properties = self.test_properties
        if len(test_label):
            msg_label = f" for {test_label:s} set"

        msg = (
            f"Summary{msg_label:s}:\n" +
            f"  Property Metrics    MAE,       RMSE\n")
        for prop in test_properties:
            msg += f"   {prop:<16s} "
            msg += f"{metrics[prop]['mae']:3.2e},  "
            msg += f"{np.sqrt(metrics[prop]['mse']):3.2e} "
            msg += f"{self.data_units[prop]:s}\n"
        logger.info("INFO:\n" + msg)
        

    def plain_data(
        self, 
        data: List[Any],
    ) -> List[Any]:

        return np.array([
            data_i 
            for data_sys in data
            for data_i in np.array(data_sys).reshape(-1)])


    def plot_correlation(
        self,
        label_dataset: str,
        label_property: str,
        data_prediction: List[float],
        data_reference: List[float],
        unit_property: str,
        data_metrics: Dict[str, float],
        test_scaling: List[float],
        test_directory: str,
        test_plot_format: str,
        test_plot_dpi: int,
    ):
        """
        Plot property data correlation data.
        (x-axis: reference data; y-axis: predicted data)
        """
        
        # Plot property: Fontsize
        SMALL_SIZE = 12
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 16
        plt.rc('font', size=SMALL_SIZE, weight='bold')
        plt.rc('xtick', labelsize=SMALL_SIZE)
        plt.rc('ytick', labelsize=SMALL_SIZE)
        plt.rc('legend', fontsize=SMALL_SIZE)
        plt.rc('axes', titlesize=MEDIUM_SIZE)
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('figure', titlesize=BIGGER_SIZE)
        
        # Plot property: Figure size and arrangement
        figsize = (6, 6)
        sfig = float(figsize[0])/float(figsize[1])
        left = 0.20
        bottom = 0.15
        column = [0.75, 0.00]
        row = [column[0]*sfig]
        
        # Initialize figure
        fig = plt.figure(figsize=figsize)
        axs1 = fig.add_axes(
            [left + 0.*np.sum(column), bottom, column[0], row[0]])
        
        # Data label
        metrics_label = (
            f"{label_property:s} ({label_dataset:s})\n" +
            f"RMSE = {np.sqrt(data_metrics['mse']):3.2e} {unit_property:s}")
        if self.is_imported("scipy"):
            r2 = stats.pearsonr(data_prediction, data_reference).statistic
            metrics_label += (
                "\n" + r"1 - $R^2$ = " + f"{1.0 - r2:3.2e}")

        # Scale data if requested
        if test_scaling is not None:
            data_prediction = data_prediction*test_scaling
            data_reference = data_reference*test_scaling
            scale_label = "per atom "
        else:
            scale_label = ""

        # Plot data
        data_min = np.min(
            (np.nanmin(data_reference), np.nanmin(data_prediction)))
        data_max = np.max(
            (np.nanmax(data_reference), np.nanmax(data_prediction)))
        data_dif = data_max - data_min
        axs1.plot(
            [data_min - data_dif*0.05, data_max + data_dif*0.05],
            [data_min - data_dif*0.05, data_max + data_dif*0.05],
            color='black',
            marker='None', linestyle='--')
        axs1.plot(
            data_reference, 
            data_prediction, 
            color='blue', markerfacecolor='None',
            marker='o', linestyle='None',
            label=metrics_label)

        # Axis range
        axs1.set_xlim(data_min - data_dif*0.05, data_max + data_dif*0.05)
        axs1.set_ylim(data_min - data_dif*0.05, data_max + data_dif*0.05)
        
        # Figure title
        axs1.set_title(
            f"Correlation plot - {label_property:s} ({label_dataset:s})",
            fontweight='bold')

        # Axis labels
        axs1.set_xlabel(
            f"Reference {label_property:s} {scale_label:s}({unit_property:s})",
            fontweight='bold')
        axs1.get_xaxis().set_label_coords(0.5, -0.12)
        axs1.set_ylabel(
            f"Model {label_property:s} {scale_label:s}({unit_property:s})",
            fontweight='bold')
        axs1.get_yaxis().set_label_coords(-0.18, 0.5)

        # Figure legend
        axs1.legend(loc='upper left')

        # Save figure
        plt.savefig(
            os.path.join(
                test_directory,
                f"{label_dataset:s}_correlation_{label_property:s}"
                + f".{test_plot_format:s}"),
            format=test_plot_format, 
            dpi=test_plot_dpi)
        plt.close()


    def plot_histogram(
        self,
        label_dataset: str,
        label_property: str,
        data_prediction: List[float],
        data_reference: List[float],
        unit_property: str,
        data_metrics: Dict[str, float],
        test_directory: str,
        test_plot_format: str,
        test_plot_dpi: int,
        test_binnum: Optional[int] = 101,
        test_histlog: Optional[bool] = False,
    ):
        """
        Plot prediction error spread as histogram.
        """
        
        # Plot property: Fontsize
        SMALL_SIZE = 12
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 16
        plt.rc('font', size=SMALL_SIZE, weight='bold')
        plt.rc('xtick', labelsize=SMALL_SIZE)
        plt.rc('ytick', labelsize=SMALL_SIZE)
        plt.rc('legend', fontsize=SMALL_SIZE)
        plt.rc('axes', titlesize=MEDIUM_SIZE)
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('figure', titlesize=BIGGER_SIZE)
        
        # Plot property: Figure size and arrangement
        figsize = (6, 6)
        sfig = float(figsize[0])/float(figsize[1])
        left = 0.20
        bottom = 0.15
        column = [0.75, 0.00]
        row = [column[0]*sfig]
        
        # Initialize figure
        fig = plt.figure(figsize=figsize)
        axs1 = fig.add_axes(
            [left + 0.*np.sum(column), bottom, column[0], row[0]])
        
        # Data label
        metrics_label = (
            f"{label_property:s} ({label_dataset:s})\n" +
            f"RMSE = {np.sqrt(data_metrics['mse']):3.2e} {unit_property:s}")
        if self.is_imported("scipy"):
            r2 = stats.pearsonr(data_prediction, data_reference).statistic
            metrics_label += (
                "\n" + r"1 - $R^2$ = " + f"{1.0 - r2:3.2e}")
        
        # Plot data
        data_dif = data_reference - data_prediction
        data_min = np.nanmin(data_dif)
        data_max = np.nanmax(data_dif)
        data_absmax = np.max((np.abs(data_min), np.abs(data_max)))
        data_absmax += data_absmax/(2.0*test_binnum)
        data_bin = np.linspace(-data_absmax, data_absmax, num=test_binnum)
        axs1.hist(
            data_reference - data_prediction, 
            bins=data_bin,
            density=True,
            color='red',
            log=test_histlog,
            label=metrics_label)

        # Axis range
        axs1.set_xlim(-data_absmax, data_absmax)
        
        # Figure title
        axs1.set_title(
            f"Prediction error distribution - {label_property:s} "
            + f"({label_dataset:s})",
            fontweight='bold')

        # Axis labels
        axs1.set_xlabel(
            f"Error in {label_property:s} ({unit_property:s})",
            fontweight='bold')
        axs1.get_xaxis().set_label_coords(0.5, -0.12)
        if test_histlog:
            ylabel = "log(Probability)"
        else:
            ylabel = "Probability"
        axs1.set_ylabel(
            ylabel,
            fontweight='bold')
        axs1.get_yaxis().set_label_coords(-0.18, 0.5)

        # Figure legend
        axs1.legend(loc='upper left')
        
        # Save figure
        plt.savefig(
            os.path.join(
                test_directory,
                f"{label_dataset:s}_histogram_{label_property:s}"
                + f".{test_plot_format:s}"),
            format=test_plot_format, 
            dpi=test_plot_dpi)
        plt.close()
        

    def plot_residual(
        self,
        label_dataset: str,
        label_property: str,
        data_prediction: List[float],
        data_reference: List[float],
        unit_property: str,
        data_metrics: Dict[str, float],
        test_scaling: List[float],
        test_directory: str,
        test_plot_format: str,
        test_plot_dpi: int,
    ):
        """
        Plot property data residual data.
        (x-axis: reference data; y-axis: prediction error)
        """
        
        # Plot property: Fontsize
        SMALL_SIZE = 12
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 16
        plt.rc('font', size=SMALL_SIZE, weight='bold')
        plt.rc('xtick', labelsize=SMALL_SIZE)
        plt.rc('ytick', labelsize=SMALL_SIZE)
        plt.rc('legend', fontsize=SMALL_SIZE)
        plt.rc('axes', titlesize=MEDIUM_SIZE)
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('figure', titlesize=BIGGER_SIZE)
        
        # Plot property: Figure size and arrangement
        figsize = (12, 6)
        left = 0.10
        bottom = 0.15
        column = [0.85, 0.00]
        row = [0.75, 0.00]
        
        # Initialize figure
        fig = plt.figure(figsize=figsize)
        axs1 = fig.add_axes(
            [left + 0.*np.sum(column), bottom, column[0], row[0]])
        
        # Data label
        metrics_label = (
            f"{label_property:s} ({label_dataset:s})\n" +
            f"RMSE = {np.sqrt(data_metrics['mse']):3.2e} {unit_property:s}")
        if self.is_imported("scipy"):
            r2 = stats.pearsonr(data_prediction, data_reference).statistic
            metrics_label += (
                "\n" + r"1 - $R^2$ = " + f"{1.0 - r2:3.2e}")

        # Scale data if requested
        if test_scaling is not None:
            data_prediction = data_prediction*test_scaling
            data_reference = data_reference*test_scaling
            scale_label = "per atom "
        else:
            scale_label = ""

        # Plot data
        data_min = np.nanmin(data_reference)
        data_max = np.nanmax(data_reference)
        data_dif = data_max - data_min
        axs1.plot(
            [data_min - data_dif*0.05, data_max + data_dif*0.05],
            [0.0, 0.0],
            color='black',
            marker='None', linestyle='--')
        data_deviation = data_reference - data_prediction
        data_devmin = np.nanmin(data_deviation)
        data_devmax = np.nanmax(data_deviation)
        data_devdif = data_devmax - data_devmin
        axs1.plot(
            data_reference, 
            data_deviation, 
            color='darkgreen', markerfacecolor='None',
            marker='o', linestyle='None',
            label=metrics_label)

        # Axis range
        axs1.set_xlim(
            data_min - data_dif*0.05, data_max + data_dif*0.05)
        axs1.set_ylim(
            data_devmin - data_devdif*0.05, data_devmax + data_devdif*0.05)
        
        # Figure title
        axs1.set_title(
            f"Residual plot - {label_property:s} ({label_dataset:s})",
            fontweight='bold')

        # Axis labels
        axs1.set_xlabel(
            f"Reference {label_property:s} {scale_label:s}({unit_property:s})",
            fontweight='bold')
        axs1.get_xaxis().set_label_coords(0.5, -0.12)
        axs1.set_ylabel(
            f"Prediction error {label_property:s} {scale_label:s}" 
            + f"({unit_property:s})",
            fontweight='bold')
        axs1.get_yaxis().set_label_coords(-0.08, 0.5)

        # Figure legend
        axs1.legend(loc='upper left')
        
        # Save figure
        plt.savefig(
            os.path.join(
                test_directory,
                f"{label_dataset:s}_residual_{label_property:s}"
                + f".{test_plot_format:s}"),
            format=test_plot_format, 
            dpi=test_plot_dpi)
        plt.close()




#class Tester:
    #"""
    #Model Prediction Tester Class
    #"""

    #def __init__(self,
        #config: Optional[Union[str, dict, object]] = None,
        #data_container: Optional[object] = None,
        #model_calculator: Optional[object] = None,
        #model_checkpoint: Optional[str] = None,
        #test_properties_evaluation: Optional[List[str]] = None,
        #**kwargs
    #):

        #####################################
        ## # # Check Model Tester Input # # #
        #####################################

        ## Get configuration object
        #config = settings.get_config(config)

        ## Check input parameter, set default values if necessary and
        ## update the configuration dictionary
        #config_update = {}
        #for arg, item in locals().items():

            ## Skip 'config' argument and possibly more
            #if arg in [
                    #'self', 'config', 'config_update', 'kwargs', '__class__']:
                #continue

            ## Take argument from global configuration dictionary if not defined
            ## directly
            #if item is None:
                #item = config.get(arg)

            ## Set default value if the argument is not defined (None)
            #if arg in settings._default_args.keys() and item is None:
                #item = settings._default_args[arg]

            ## Check datatype of defined arguments
            #if arg in settings._dtypes_args.keys():
                #match = utils.check_input_dtype(
                    #arg, item, settings._dtypes_args, raise_error=True)

            ## Append to update dictionary
            #config_update[arg] = item

            ## Assign as class parameter
            #setattr(self, arg, item)

        ## Update global configuration dictionary
        #config.update(config_update)

        #################################
        ## # # Check Data Container # # #
        #################################

        ## Assign DataContainer and test data loader 
        #if self.data_container is None:
            #self.data_container = data.DataContainer(
                #config=config,
                #**kwargs)
        #self.data_test = self.data_container.test_loader

        ## Get reference data properties
        #self.data_properties = self.data_container.data_load_properties
        #self.data_units = self.data_container.data_unit_properties

        #################################
        ## # # Check NNP Calculator # # #
        #################################

        ### Assign NNP calculator model
        ##if self.model_calculator is None:
            ##self.model_calculator = model.get_calculator(
                ##config=config,
                ##**kwargs)

        ## Load checkpoint (best model parameters by default).
        ##if self.model_checkpoint is None:
            ##self.model_checkpoint = os.path.join(
                ##config['model_directory'], 'best/best_model.pt')
            
            ##print('No model checkpoint provided, loading best model from {}'.format(config['model_directory']))
        
        
        ### Get reference data properties
        ##self.model_properties = self.model_calculator.model_properties

        ## Check for test properties, Ideally test properties should be defined in the config file
        ## If not, consider all properties covered in reference data set and the
        ## model calculator.
        #if (self.test_properties_evaluation is None) or (len(self.test_properties_evaluation) == 0):

            ## Reinitialize test properties list
            #self.test_properties_evaluation = []

            ## Iterate over model properties, check for property in reference
            ## data set and eventually add to training properties.
            #for prop in self.model_properties:
                #if prop in self.data_properties:
                    #self.test_properties_evaluation.append(prop)


        ## Else check training properties and eventually correct for
        ## not covered properties in the reference data set or the
        ## model calculator.
        #else:
            ## Iterate over training properties, check for property in reference
            ## data set and model calculator prediction and eventually remove
            ## training property.
            ## NOTE: Eventually we would like to add evaluation to Normal modes or derived properties not
            ## covered in the reference data set. We should think about a way to do this.
            #for prop in self.test_properties_evaluation:
                #if not (prop in self.data_properties and
                        #prop in self.model_properties):
                    #self.test_properties_evaluation.remove(prop)
                    #logger.warning(
                        #f"WARNING:\nProperty '{prop}' in " +
                        #f"'test_properties_evaluation' is not stored in the " +
                        #f"reference data set and/or predicted by the model " +
                        #f"model calculator!\n" +
                        #f"Property '{prop}' is removed from evaluation " +
                        #f"property list.")
    #@staticmethod
    #def is_imported(module):
        #'''
        #Check if a module is imported in the current session.
        #'''
        #return module in sys.modules

    #def test(self, verbose=True,
             #save_npz=False, npz_name='test_vals.npz',
             #save_csv=False, csv_name='test_vals.csv',
             #plot=False, plots_to_show=None, save_plots=False, show_plots=False,
             #residual_plots=False, residuals_to_show=None, save_residuals=False, show_residuals=False,
             #histogram_plots=False, histograms_to_show=None, save_histograms=False, show_histograms=False,
             #):

        #if plot or residual_plots or histogram_plots:
            #if not self.is_imported('matplotlib') or not self.is_imported('seaborn'):
                #raise UserWarning("You need to install matplotlib to use all plotting and analysis functions. Some functions might not work.")

        #if save_csv:
            #if not self.is_imported('pandas'):
                #raise UserWarning("You need to install pandas to save a .csv file. The file will not be saved. Try saving a .npz file instead.")

        ## Check if saving a file is required and then create a folder for the files in a directory called 'test_results'
        #if save_npz or save_csv or save_plots or save_residuals or save_histograms:
            #if not os.path.exists('test_results'):
                #os.makedirs('test_results')
                #self.save_directory = os.path.join(os.getcwd(),'test_results')
                #print('Results will be saved in the directory: ',self.save_directory)

        ## Check if both .csv and .npz files are saved else raise a warning
        #if save_npz and save_csv:
            #raise UserWarning('You are saving both a .csv and a .npz file. This is not recommended.')

        ## Check which plots to show, by default show only the energy plot, if 'all' is selected show plots for all available properties
        #if plot and plots_to_show is None:
            #plots_to_show = ['energy']
        #elif plots_to_show == 'all':
            #plots_to_show = self.test_properties_evaluation
        #else:
            #plots_to_show = plots_to_show

        ## Check which residuals to show, by default show only the energy plot, if 'all' is selected show plots for all available properties
        #if residual_plots and residuals_to_show is None:
            #residuals_to_show = ['energy']
        #elif residuals_to_show == 'all':
            #residuals_to_show = self.test_properties_evaluation
        #else:
            #residuals_to_show = residuals_to_show

        ## Check which histograms to show, by default show only the energy plot, if 'all' is selected show plots for all available properties
        #if histogram_plots and histograms_to_show is None:
            #histograms_to_show = ['energy']
        #elif histograms_to_show == 'all':
            #histograms_to_show = self.test_properties_evaluation
        #else:
            #histograms_to_show = histograms_to_show

        ## Initialize training mode for calculator
        #self.model_calculator.eval()
        #values = {}
        #for ib, batch in enumerate(self.data_test):
            ## Predict model properties from data batch
            #prediction = self.model_calculator(batch)
            ## Compute metrics
            #vals = self.compute_vals(prediction, batch)
            #values.update(vals)
        ## Print absolute errors
        #if verbose:
            #self.print_averages(values)
        #if save_npz:
            #self.save_npz(values,npz_name)
        #if save_csv:
            #self.save_csv(values,csv_name)
        #if plot:
            #print('Plotting results as scatter plot, other options available are residuals and histograms of the errors')
            #self.plot_results(values,plots_to_show=plots_to_show,save_plots=save_plots,show_plots=show_plots)
        #if residual_plots:
            #self.plot_residuals(values,residuals_to_show,save_residuals=save_residuals,show_residuals=show_residuals)
        #if histogram_plots:
            #self.plot_histograms(values,histograms_to_show,save_histograms=save_histograms,show_histograms=show_histograms)


    #def reset_metrics(self):
        ## Initialize metrics dictionary
        #metrics = {}
        ## Add training property metrics
        #for prop in self.test_properties_evaluation:
            #metrics[prop] = {
                #'mae': 0.0,
                #'mse': 0.0}
        #return metrics

    #def compute_metrics(self,vals):
        ## Initialize metrics dictionary
        #metrics = self.reset_metrics()
        ## Add training property metrics
        #for prop in self.test_properties_evaluation:
            #metrics[prop]['mae'] = np.mean(np.abs(vals['error {}'.format(prop)]))
            #metrics[prop]['mse'] = np.mean(np.square(vals['error {}'.format(prop)]))
        #return metrics

    #def print_averages(self,vals):
        #averages = self.compute_metrics(vals)
        #msg = 'Averages for test set:\n'
        #msg += '----------------------\n'
        ## print(msg)
        #for prop in self.test_properties_evaluation:
            #msg += f"{prop}:\n"
            #msg += f"    MAE: {averages[prop]['mae']:.6f}\n"
            #msg += f"    MSE: {averages[prop]['mse']:.6f}\n"
            ## print(msg)
        #msg += '----------------------\n'
        #print(msg)

    #def save_npz(self,vals,npz_name):
        #print('Saving the results of the test set to file {}'.format(npz_name))
        #path_to_save = os.path.join(self.save_directory,npz_name)
        #np.savez(path_to_save, **vals)


    #def save_csv(self,vals,csv_name):
        #print('Saving the results of the test set to file {}'.format(csv_name))
        #path_to_save = os.path.join(self.save_directory,csv_name)
        #df = pd.DataFrame(vals)
        #df.to_csv(path_to_save, index=False)

    #def plot_results(self, vals, plots_to_show, save_plots=False, show_plots=False):
        #for prop in plots_to_show:
            #if prop == 'forces' or prop == 'dipole':
                #vals['{} reference'.format(prop)] = np.reshape(vals['{} reference'.format(prop)],(-1,3))
                #vals['{} prediction'.format(prop)] = np.reshape(vals['{} prediction'.format(prop)],(-1,3))
                #fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                #for i in range(3):
                    #sns.scatterplot(x=vals['{} reference'.format(prop)][:, i], y=vals['{} prediction'.format(prop)][:, i],
                                #ax=ax[i])
                    #trendline = np.arange(np.min(vals['{} prediction'.format(prop)][:, i]),
                                          #np.max(vals['{} prediction'.format(prop)][:, i]), 0.1)
                    #ax[i].plot(trendline, trendline, color='black', linestyle='--')
                    #ax[i].set_xlim(np.min(vals['{} prediction'.format(prop)][:, i]),
                                   #np.max(vals['{} prediction'.format(prop)][:, i]))
                    #ax[i].set_ylim(np.min(vals['{} prediction'.format(prop)][:, i]),
                                   #np.max(vals['{} prediction'.format(prop)][:, i]))
                    #ax[i].set_xlabel('Reference {}'.format(prop))
                    #ax[i].set_ylabel('Predicted {}'.format(prop))
                #plt.tight_layout()
                #if save_plots:
                    #path_to_save = os.path.join(self.save_directory, '{}_test.pdf'.format(prop))
                    #plt.savefig(path_to_save, dpi=300)
                #if show_plots:
                    #plt.show()
                #plt.close()
            #else:
                #slope, intercept, r, p, se = stats.linregress(vals['{} prediction'.format(prop)],
                                                              #vals['{} reference'.format(prop)])
                #rsquare = r ** 2
                #fig, ax = plt.subplots(figsize=(5, 5))
                #sns.scatterplot(x=vals['{} reference'.format(prop)], y=vals['{} prediction'.format(prop)], ax=ax)
                #ax.text(0.05, 0.95, '$R^2$ = {:.3f}'.format(rsquare), transform=ax.transAxes)
                #trendline = np.arange(np.min(vals['{} prediction'.format(prop)]),
                                      #np.max(vals['{} prediction'.format(prop)]), 0.1)
                #ax.plot(trendline, trendline, color='black', linestyle='--')
                #ax.set_xlim(np.min(vals['{} prediction'.format(prop)]), np.max(vals['{} prediction'.format(prop)]))
                #ax.set_ylim(np.min(vals['{} prediction'.format(prop)]), np.max(vals['{} prediction'.format(prop)]))
                #ax.set_xlabel('Reference {}'.format(prop))
                #ax.set_ylabel('Predicted {}'.format(prop))
                #plt.tight_layout()
                #if save_plots:
                    #path_to_save = os.path.join(self.save_directory, '{}_test.pdf'.format(prop))
                    #plt.savefig(path_to_save, dpi=300)
                #if show_plots:
                    #plt.show()
                #plt.close()

    #def plot_residuals(self, vals, residuals_to_show, save_residuals=False, show_residuals=False):
        #'''
        #I hate residuals, but Kai likes them, so here they are--LIVS
        #'''
        #for prop in residuals_to_show:
            #fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            #ax.plot(vals['{} reference'.format(prop)], vals['error {}'.format(prop)], 'o')
            #ax.set_xlabel('Reference {}'.format(prop))
            #ax.set_ylabel('Error {}'.format(prop))
            #if save_residuals:
                #path_to_save = os.path.join(self.save_directory, 'residuals_{}_test.pdf'.format(prop))
                #plt.savefig(path_to_save, dpi=300)
            #if show_residuals:
                #plt.show()
            #plt.close()

    #def plot_histograms(self, vals, histograms_to_show, save_histograms=False, show_histograms=False):
        #for prop in histograms_to_show:
            #fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            #sns.histplot(vals['error {}'.format(prop)], ax=ax, kde=True)
            #ax.set_xlabel('Error {}'.format(prop))
            #ax.set_ylabel('Count')
            #if save_histograms:
                #path_to_save = os.path.join(self.save_directory, 'histogram_{}_test.pdf'.format(prop))
                #plt.savefig(path_to_save, dpi=300)
            #if show_histograms:
                #plt.show()
            #plt.close()

    #def compute_vals(self,
            #prediction: Dict[str, Any],
            #reference: Dict[str, Any],
    #) -> Dict[str, float]:
        ##This function computes the individual values of the error for each property in the test set
        ## Initialize metrics dictionary
        #metrics = {}
        #for ip,prop in enumerate(self.test_properties_evaluation):
            #metrics['{} reference'.format(prop)] = reference[prop].detach().cpu().numpy().flatten()
            #metrics['{} prediction'.format(prop)] = prediction[prop].detach().cpu().numpy().flatten()
            #metrics['error {}'.format(prop)] = prediction[prop].detach().cpu().numpy().flatten() - reference[prop].detach().cpu().numpy().flatten()
        #return metrics



