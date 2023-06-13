import logging
import torch
from typing import Optional, List, Dict, Tuple, Union, Any
import numpy as np
import os

from .. import data
from .. import settings
from .. import utils

try: #These packages are necessary for the plotting and analysis but only here.
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['Testing']

class Testing:

    def __init__(self,
        config: Optional[Union[str, dict, object]] = None,
        data_container: Optional[object] = None,
        model_calculator: Optional[object] = None,
        checkpoint: Optional[str] = None,**kwargs):


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

        self.data_test = self.data_container.test_loader

        # Get reference data properties
        self.data_properties = self.data_container.data_load_properties
        self.data_units = self.data_container.data_unit_properties

        # Assign DataContainer
        if self.model_calculator is None:
            # Assign NNP calculator model
            self.model_calculator = model.get_calculator(
                config=config,
                **kwargs)

        # Load checkpoint if not provided, it takes the best model by default.
        if checkpoint is None:
            print('No checkpoint provided, loading best model from {}'.format(config['model_directory']))
            self.checkpoint = os.path.join(config['model_directory'],'best/best_model.pt')
        else:
            self.checkpoint = checkpoint

        latest_ckpt = utils.load_checkpoint(self.checkpoint)
        self.model_calculator.load_state_dict(latest_ckpt['model_state_dict'])

        # Get reference data properties
        self.model_properties = self.model_calculator.model_properties

        # Check for test properties, Ideally test properties should be defined in the config file
        # If not, consider all properties covered in reference data set and the
        # model calculator.
        if (not len(self.test_properties_evaluation) or
                self.test_properties_evaluation is None):

            # Reinitialize test properties list
            self.test_properties_evaluation = []

            # Iterate over model properties, check for property in reference
            # data set and eventually add to training properties.
            for prop in self.model_properties:
                if prop in self.data_properties:
                    self.test_properties_evaluation.append(prop)


        # Else check training properties and eventually correct for
        # not covered properties in the reference data set or the
        # model calculator.
        else:
            # Iterate over training properties, check for property in reference
            # data set and model calculator prediction and eventually remove
            # training property.
            # NOTE: Eventually we would like to add evaluation to Normal modes or derived properties not
            # covered in the reference data set. We should think about a way to do this.
            for prop in self.test_properties_evaluation:
                if not (prop in self.data_properties and
                        prop in self.model_properties):
                    self.test_properties_evaluation.remove(prop)
                    logger.warning(
                        f"WARNING:\nProperty '{prop}' in " +
                        f"'test_properties_evaluation' is not stored in the " +
                        f"reference data set and/or predicted by the model " +
                        f"model calculator!\n" +
                        f"Property '{prop}' is removed from evaluation " +
                        f"property list.")

    def test(self, verbose=True,
             save_npz=False, npz_name='test_vals.npz',
             save_csv=False, csv_name='test_vals.csv',
             plot=False, plots_to_show=None,save_plots=False,
             residual_plots=False, residuals_to_show=None,save_residuals=False,
             histogram_plots=False, histograms_to_show=None,save_histograms=False,
             ):

        # Check if saving a file is required and then create a folder for the files in a directory called 'test_results'
        if save_npz or save_csv or save_plots or save_residuals or save_histograms:
            if not os.path.exists('test_results'):
                os.makedirs('test_results')
                self.save_directory = os.path.join(os.getcwd(),'test_results')
                print('Results will be saved in the directory: ',self.save_directory)

        # Check if both .csv and .npz files are saved else raise a warning
        if save_npz and save_csv:
            raise UserWarning('You are saving both a .csv and a .npz file. This is not recommended.')

        # Check which plots to show, by default show only the energy plot, if 'all' is selected show plots for all available properties
        if plot and plots_to_show is None:
            plots_to_show = ['Energy']
        elif plots_to_show == 'all':
            plots_to_show = self.test_properties_evaluation
        else:
            plots_to_show = plots_to_show

        # Check which residuals to show, by default show only the energy plot, if 'all' is selected show plots for all available properties
        if residual_plots and residuals_to_show is None:
            residuals_to_show = ['Energy']
        elif residuals_to_show == 'all':
            residuals_to_show = self.test_properties_evaluation
        else:
            residuals_to_show = residuals_to_show

        # Check which histograms to show, by default show only the energy plot, if 'all' is selected show plots for all available properties
        if histogram_plots and histograms_to_show is None:
            histograms_to_show = ['Energy']
        elif histograms_to_show == 'all':
            histograms_to_show = self.test_properties_evaluation
        else:
            histograms_to_show = histograms_to_show

        # Initialize training mode for calculator
        self.model_calculator.eval()
        values = {}
        for ib, batch in enumerate(self.data_test):
            # Predict model properties from data batch
            prediction = self.model_calculator(batch)
            # Compute metrics
            vals = self.compute_vals(prediction, batch)
            values.update(vals)

        # Print absolute errors
        if verbose:
            self.print_averages(values)
        if save_npz:
            self.save_npz(values,npz_name)
        if save_csv:
            self.save_csv(values,csv_name)
        if plot:
            print('Plotting results as scatter plot, other options available are residuals and histograms of the errors')
            self.plot_results(values,plots_to_show=plots_to_show,save_plots=save_plots)
        if residual_plots:
            self.plot_residuals(values,residuals_to_show,save_residuals=save_residuals)
        if histogram_plots:
            self.plot_histograms(values,histograms_to_show,save_histograms=save_histograms)


    def reset_metrics(self):
        # Initialize metrics dictionary
        metrics = {}
        # Add training property metrics
        for prop in self.test_properties_evaluation:
            metrics[prop] = {
                'mae': 0.0,
                'mse': 0.0}
        return metrics

    def compute_metrics(self,vals):
        # Initialize metrics dictionary
        metrics = self.reset_metrics()
        # Add training property metrics
        for prop in self.test_properties_evaluation:
            metrics[prop]['mae'] = np.mean(np.abs(vals['error {}'.format(prop)]))
            metrics[prop]['mse'] = np.mean(np.square(vals['error {}'.format(prop)]))
        return metrics

    def print_averages(self,vals):
        averages = self.compute_metrics(vals)
        msg = 'Averages for test set:\n'
        msg += '----------------------\n'
        # print(msg)
        for prop in self.test_properties_evaluation:
            msg += f"{prop}:\n"
            msg += f"    MAE: {averages[prop]['mae']:.6f}\n"
            msg += f"    MSE: {averages[prop]['mse']:.6f}\n"
            # print(msg)
        msg += '----------------------\n'
        print(msg)

    def save_npz(self,vals,npz_name):
        print('Saving the results of the test set to file {}'.format(npz_name))
        path_to_save = os.path.join(self.save_directory,npz_name)
        np.savez(path_to_save, **vals)


    def save_csv(self,vals,csv_name):
        print('Saving the results of the test set to file {}'.format(csv_name))
        path_to_save = os.path.join(self.save_directory,csv_name)
        df = pd.DataFrame(vals)
        df.to_csv(path_to_save, index=False)

    def plot_results(self,vals,plots_to_show,save_plots=False):
        for prop in plots_to_show:
            if prop == 'Forces' or prop=='Dipole':
                fig, ax = plt.subplots(3,1,figsize=(15,5))
                for i in range(3):
                    sns.regplot(x=vals['{} reference'.format(prop)][:,i], y=vals['{} prediction'.format(prop)][:,i], ax=ax[i])
                    trendline = np.arange(np.min(vals['{} reference'.format(prop)][:,i]), np.max(vals['{} reference'.format(prop)][:,i]), 0.1)
                    ax[i].plot(trendline, trendline, color='black', linestyle='--')
                    ax[i].set_xlim(np.min(vals['{} reference'.format(prop)][:,i]), np.max(vals['{} reference'.format(prop)][:,i]))
                    ax[i].set_ylim(np.min(vals['{} reference'.format(prop)][:,i]), np.max(vals['{} reference'.format(prop)][:,i]))
                    ax[i].set_xlabel('Reference {}'.format(prop))
                    ax[i].set_ylabel('Predicted {}'.format(prop))
                if save_plots:
                    path_to_save = os.path.join(self.save_directory,'{}_test.pdf'.format(prop))
                    plt.savefig(path_to_save,dpi=300)
                plt.show()
            else:
                slope, intercept, r, p, se = stats.linregress(vals['{} prediction'.format(prop)], vals['{} reference'.format(prop)])
                rsquare = r ** 2
                fig, ax = plt.subplots(figsize=(5,5))
                sns.regplot(x=vals['{} reference'.format(prop)], y=vals['{} prediction'.format(prop)], ax=ax)
                ax.text(0.05, 0.95, '$R^2$ = {:.3f}'.format(rsquare), transform=ax.transAxes)
                trendline = np.arange(np.min(vals['{} reference'.format(prop)]), np.max(vals['{} reference'.format(prop)]), 0.1)
                ax.plot(trendline, trendline, color='black', linestyle='--')
                ax.set_xlim(np.min(vals['{} reference'.format(prop)]), np.max(vals['{} reference'.format(prop)]))
                ax.set_ylim(np.min(vals['{} reference'.format(prop)]), np.max(vals['{} reference'.format(prop)]))
                ax.set_xlabel('Reference {}'.format(prop))
                ax.set_ylabel('Predicted {}'.format(prop))
                if save_plots:
                    path_to_save = os.path.join(self.save_directory,'{}_test.pdf'.format(prop))
                    plt.savefig(path_to_save,dpi=300)
                plt.show()

    def plot_residuals(self,vals,residuals_to_show,save_residuals=False):
        '''
        I hate residuals, but Kai likes them, so here they are--LIVS
        '''
        for prop in residuals_to_show:
            fig, ax = plt.subplots(1,1,figsize=(5,5))
            if prop == 'Forces' or prop=='Dipole':
                error_vals = np.mean(vals['error {}'.format(prop)],ax=1)
                ax.plot(vals['{} reference'.format(prop)],error_vals,'o')
            else:
                ax.plot(vals['{} reference'.format(prop)],vals['error {}'.format(prop)],'o')
            ax.set_xlabel('Reference {}'.format(prop))
            ax.set_ylabel('Error {}'.format(prop))
            if save_residuals:
                path_to_save = os.path.join(self.save_directory,'residuals_{}_test.pdf'.format(prop))
                plt.savefig(path_to_save, dpi=300)
            plt.show()

    def plot_histograms(self,vals,histograms_to_show,save_histograms=False):
        for prop in histograms_to_show:
            fig, ax = plt.subplots(1,1,figsize=(5,5))
            if prop == 'Forces' or prop=='Dipole':
                error_vals = np.mean(vals['error {}'.format(prop)],ax=1)
                sns.histplot(error_vals,ax=ax,kde=True)
            else:
                sns.histplot(vals['error {}'.format(prop)],ax=ax,kde=True)
            ax.set_xlabel('Error {}'.format(prop))
            ax.set_ylabel('Count')
            if save_histograms:
                path_to_save = os.path.join(self.save_directory,'histogram_{}_test.pdf'.format(prop))
                plt.savefig(path_to_save, dpi=300)
            plt.show()


    def compute_vals(self,
            prediction: Dict[str, Any],
            reference: Dict[str, Any],
    ) -> Dict[str, float]:
        #This function computes the individual values of the error for each property in the test set
        # Initialize metrics dictionary
        metrics = {}
        for ip,prop in enumerate(self.test_properties_evaluation):
            metrics['{} reference'.format(prop)] = reference[prop].detach().cpu().numpy()
            metrics['{} prediction'.format(prop)] = prediction[prop].detach().cpu().numpy()
            metrics['error {}'.format(prop)] = prediction[prop].detach().cpu().numpy() - reference[prop].detach().cpu().numpy()
        return metrics




