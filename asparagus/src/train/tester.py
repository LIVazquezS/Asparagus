import os
import sys

import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import torch

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

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        data_container: Optional[object] = None,
        test_datasets: Optional[Union[str, List[str]]] = None,
        test_properties: Optional[Union[str, List[str]]] = None,
        test_store_neighbor_list: Optional[bool] = None,
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
        test_store_neighbor_list: bool, optional, default True
            Store neighbor list parameter in the database file instead of
            computing in situ.
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
        for label, datasubset in self.test_data.items():

            # Set maximum model cutoff for neighbor list calculation
            datasubset.init_neighbor_list(
                cutoff=model_calculator.model_interaction_cutoff,
                store=self.test_store_neighbor_list)

            # Prepare dictionary for property values and number of atoms per
            # system
            test_prediction = {prop: [] for prop in test_properties}
            test_reference = {prop: [] for prop in test_properties}
            test_prediction['atoms_number'] = []

            # Reset property metrics
            metrics_test = self.reset_metrics(eval_properties)

            # Loop over data batches
            for batch in datasubset:

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

            # Check if both .csv and .npz files are saved else raise a warning
            if test_save_npz and test_save_csv:
                raise UserWarning(
                    "You are saving both a .csv and a .npz file."
                    + "This is not recommended!")

            # Save test prediction to files
            if test_save_csv:
                self.save_csv(test_prediction, test_directory, test_csv_file)
            if test_save_npz:
                self.save_npz(test_prediction, test_directory, test_npz_file)

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
    def is_imported(module: str):
        """
        Check if a module is imported.
        """

        return module in sys.modules

    def save_npz(
        self,
        vals: Dict,
        test_directory: str,
        npz_name: str,
    ):

        path_to_save = os.path.join(test_directory, npz_name)
        logger.info(
            "INFO:\nSaving results of the test set to file "
            + f"'{path_to_save:s}'!")
        np.savez(path_to_save, **vals)

    def save_csv(
        self,
        vals: Dict,
        test_directory: str,
        csv_name: str
    ):

        # Check for .csv file extension
        if '.csv' == csv_name[-4:]:
            csv_name += '.csv'
        path_to_save = os.path.join(test_directory, csv_name)
        logger.info(
            "INFO:\nSaving results of the test set to file "
            + f"'{path_to_save:s}'!")

        # Check that all the keys have the same length

        # First get the lenghts of each of the properties in the dictionary
        lengths = [len(item) for key, item in vals.items()]
        max_length = np.max(lengths)

        # Pad the lenghts with nan
        vals_padded = {}
        for key, item in vals.items():
            if len(item) < max_length:
                vals_padded[key] = np.pad(
                    vals[key],
                    (0, max_length - len(vals[key])),
                    'constant',
                    constant_values=np.nan)
            else:
                vals_padded[key] = item

        if self.is_imported("pandas"):
            df = pd.DataFrame(vals_padded)
            df.to_csv(path_to_save, index=False)
        else:
            logger.warning(
                "WARNING:\nModule 'pandas' is not available. "
                + "Test properties are not written to a csv file!")
        return

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
            f"Summary{msg_label:s}:\n"
            + "  Property Metrics    MAE,       RMSE\n")
        for prop in test_properties:
            msg += f"   {prop:<16s} "
            msg += f"{metrics[prop]['mae']:3.2e},  "
            msg += f"{np.sqrt(metrics[prop]['mse']):3.2e} "
            msg += f"{self.data_units[prop]:s}\n"
        logger.info("INFO:\n" + msg)

    def plain_data(
        self,
        data_nd: List[Any],
    ) -> List[Any]:

        return np.array([
            data_i
            for data_sys in data_nd
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
