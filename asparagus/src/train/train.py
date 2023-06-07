# Note: This class maybe can gain a lot from Pytorch Lightning. TODO: Check if it is possible to use it.
# Standard importations
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import os
import sys
import numpy as np
import argparse
import logging
from typing import Optional, List, Dict, Tuple, Union, Any
import string
import random
from torch_ema import ExponentialMovingAverage
# For Time measurement
from datetime import datetime
from time import time
# Neural Network importations
from .. import utils
from .. import settings
from ..model import Get_Properties


class Train:
    def __init__(
        self,
        model: Optional[object] = None,
        config: Optional[Union[str, dict, object]] = None,
        trainer_max_epochs: Optional[int] = None,
        trainer_metrics_properties: Optional[Dict[str, str]] = None,
        trainer_weigths_properties: Optional[Dict[str, float]] = None,
        trainer_optimizer: Optional[Union[str, object]] = None,
        trainer_optimizer_args: Optional[Dict[str, float]] = None,
        trainer_scheduler: Optional[Union[str, object]] = None,
        trainer_scheduler_args: Optional[Dict[str, float]] = None,
        trainer_ema: Optional[bool] = None,
        trainer_ema_decay: Optional[float] = None,
        trainer_max_gradient_norm: Optional[float] = None,
        trainer_save_interval: Optional[int] = None,
        trainer_validation_interval: Optional[int] = None,
        optimizer=None,
        scheduler=None,
        **kwargs
    ):

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

        self.args = config

        self.device = settings._global_device

    @staticmethod
    def reset_averages(forces=False,charges=False,Dipole=False):
        quantities = {}
        quantities['loss_avg'] = 0.0
        quantities['emse_avg'] = 0.0
        quantities['emae_avg'] = 0.0
        if forces:
            quantities['fmse_avg'] = 0.0
            quantities['fmae_avg'] = 0.0
        if charges:
            quantities['qmse_avg'] = 0.0
            quantities['qmae_avg'] = 0.0
        if Dipole:
            quantities['dmse_avg'] = 0.0
            quantities['dmae_avg'] = 0.0
        return quantities


    def l2_regularizer(self,model, l2_lambda=0.0):
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        return l2_lambda * l2_norm

    # ====================================
    # Some functions
    # ====================================
    def compute_pnorm(self,model):
        """Computes the norm of the parameters of a model."""
        return np.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))

    def compute_gnorm(self,model):
        """Computes the norm of the gradients of a model."""
        return np.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))

    #====================================
    # Loss function
    #====================================


    def loss_fn(self,val1, val2):
        ''' Calculate error values and loss function '''
        if val1 is None or val2 is None:
            loss = torch.zeros(1, dtype=torch.float64, device=self.device)
            rmse = torch.zeros(1, dtype=torch.float64, device=self.device)
            mae = torch.zeros(1, dtype=torch.float64, device=self.device)
        else:
            lf = nn.L1Loss(reduction="mean")
            loss = lf(val1.reshape(-1), val2.reshape(-1))
            delta2 = loss ** 2
            mae = loss
            # Mean squared error
            mse = torch.sum(delta2)
            rmse = torch.sqrt(mse)
        return loss, rmse, mae


    @staticmethod
    def averaging_quantities(
        quantities, batch_size, loss_avg, emse_avg, emae_avg,
        fmse_avg=None, fmae_avg=None, qmse_avg=None, qmae_avg=None,
        dmse_avg=None, dmae_avg=None, num=0):
        ''' Averaging quantities '''
        #Test if quantities are attached
        vals = [loss_avg, emse_avg, emae_avg, fmse_avg, fmae_avg, qmse_avg, qmae_avg, dmse_avg, dmae_avg]
        for val in vals:
            if val is not None:
                utils.detach_tensor(val)

        f = num/(num + batch_size)
        loss_avg = f*loss_avg + (1-f)*loss_avg
        emse_avg = f*emse_avg + (1-f)*emse_avg
        emae_avg = f*emae_avg + (1-f)*emae_avg
        if fmse_avg is not None:
            fmse_avg = f*fmse_avg + (1-f)*fmse_avg
            fmae_avg = f*fmae_avg + (1-f)*fmae_avg
        if qmse_avg is not None:
            qmse_avg = f*qmse_avg + (1-f)*qmse_avg
            qmae_avg = f*qmae_avg + (1-f)*qmae_avg
        if dmse_avg is not None:
            dmse_avg = f*dmse_avg + (1-f)*dmse_avg
            dmae_avg = f*dmae_avg + (1-f)*dmae_avg

        quantities_new = {'loss_avg': loss_avg, 'emse_avg': emse_avg, 'emae_avg': emae_avg, 'fmse_avg': fmse_avg,
                        'fmae_avg': fmae_avg, 'qmse_avg': qmse_avg, 'qmae_avg': qmae_avg, 'dmse_avg': dmse_avg,
                        'dmae_avg': dmae_avg}
        quantities.update(quantities_new)
        return quantities

    @staticmethod
    def reporter(quantities,train=False,valid=False):
        results = {}
        if train:
            results['train_loss'] = quantities['loss_avg']
            results['train_emse'] = quantities['emse_avg']
            results['train_emae'] = quantities['emae_avg']
            if quantities['fmse_avg'] is not None:
                results['train_fmse'] = quantities['fmse_avg']
                results['train_fmae'] = quantities['fmae_avg']
            if quantities['qmse_avg'] is not None:
                results['train_qmse'] = quantities['qmse_avg']
                results['train_qmae'] = quantities['qmae_avg']
            if quantities['dmse_avg'] is not None:
                results['train_dmse'] = quantities['dmse_avg']
                results['train_dmae'] = quantities['dmae_avg']
        if valid:
            results['valid_loss'] = quantities['loss_avg']
            results['valid_emse'] = quantities['emse_avg']
            results['valid_emae'] = quantities['emae_avg']
            if quantities['fmse_avg'] is not None:
                results['valid_fmse'] = quantities['fmse_avg']
                results['valid_fmae'] = quantities['fmae_avg']
            if quantities['qmse_avg'] is not None:
                results['valid_qmse'] = quantities['qmse_avg']
                results['valid_qmae'] = quantities['qmae_avg']
            if quantities['dmse_avg'] is not None:
                results['valid_dmse'] = quantities['dmse_avg']
                results['valid_dmae'] = quantities['dmae_avg']
        else:
            results['best_loss'] = quantities['loss_avg']
            results['best_emse'] = quantities['emse_avg']
            results['best_emae'] = quantities['emae_avg']
            if quantities['fmse_avg'] is not None:
                results['best_fmse'] = quantities['fmse_avg']
                results['best_fmae'] = quantities['fmae_avg']
            if quantities['qmse_avg'] is not None:
                results['best_qmse'] = quantities['qmse_avg']
                results['best_qmae'] = quantities['qmae_avg']
            if quantities['dmse_avg'] is not None:
                results['best_dmse'] = quantities['dmse_avg']
                results['best_dmae'] = quantities['dmae_avg']
        return results


    def train_step(self,batch, quantities,device='cpu'):
        # Initialize training mode
        #self.model.train()

        idx_t = batch['pairs_seg']
        batch_seg_t = batch['atoms_seg']

        # Evaluate model
        idx_i = batch['idx_i']
        idx_j = batch['idx_j']
        dct_prop = self.model(batch['atoms_number'],
            batch['atomic_numbers'], batch['positions'], idx_i, idx_j, batch['charge'], batch_seg_t,batch['pbc_offset'])

        # Evaluate error and losses for ...
        #Basic properties for energy and charges.
        # Energy
        eloss_t, emse_t, emae_t = self.loss_fn(batch['energy'], dct_prop['energy'])
        # Total charge
        qloss_t, qmse_t, qmae_t = self.loss_fn(batch['charge'], dct_prop['charge'])

        #Other properties
        # Dipole moment
        dloss_t, dmse_t, dmae_t = self.loss_fn(batch['dipole'], dct_prop['dipole'])
        # # Forces
        floss_t, fmse_t, fmae_t = self.loss_fn(batch['forces'], dct_prop['forces'])

        # Atomic charges
        qaloss_t, qamse_t, qamae_t = self.loss_fn(
            batch.get('atomic_charges'), dct_prop.get('atomic_charges'))

        weights = self.args.get("trainer_weigths_properties")
        props = [
            'energy', 'forces', 'charge', 'dipole', 'atomic_charges',
            'nhlambda']
        for prop in props:
            if prop not in weights:
                weights[prop] = 1.0
        loss_t = (weights.get('energy') * eloss_t
                  + weights.get('forces') * floss_t
                  + weights.get('charge') * qloss_t
                  + weights.get('dipole') * dloss_t
                  + weights.get('atomic_charges') * qaloss_t
                  + weights.get('nhlambda') * dct_prop['nhloss']
                  + self.l2_regularizer(self.model))
        batch_size = batch["atoms_number"].size(0)

        quantities = self.averaging_quantities(
            quantities, batch_size, loss_t, emse_t, emae_t,
            fmse_t, fmae_t, qmse_t, qmae_t, dmse_t, dmae_t)

        loss_t.backward(retain_graph=True)
        
        # Gradient clip
        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.args.get('trainer_max_gradient_norm'))
        loss_t.detach()

        return quantities

    def valid_step(self,batch, quantities,device='cpu'):
        # Initialize training mode
        self.model.eval()

        idx_v = batch['pairs_seg']
        batch_seg_v = batch['atoms_seg']
        #if not utils.is_grad_enabled(batch['positions']):
            #R_v = batch['positions'].requires_grad_(True)

        ## Gather data
        #Z_v = utils.gather_nd(batch['atomic_numbers'], idx_v)
        #R_v = utils.gather_nd(batch['positions'], idx_v)

        #if batch.has_key('forces'):
            #Fref_v = utils.gather_nd(batch['forces'], idx_v)
        #if batch.has_key('atomic_charges'):
            #Qaref_v = utils.gather_nd(batch['atomic_charges'], idx_v)

        # Evaluate model
        idx_i = batch['idx_i']
        idx_j = batch['idx_j']
        dct_prop = self.model(
            batch['atoms_number'], batch['atomic_numbers'], batch['positions'],
            idx_i, idx_j, batch['charge'], batch_seg_v, batch['pbc_offset'])

        # Evaluate error and losses for ...
        #Basic properties
        #Energy
        eloss_v, emse_v, emae_v = self.loss_fn(batch['energy'], dct_prop['energy'])
        # Total charge
        qloss_v, qmse_v, qmae_v = self.loss_fn(batch['charge'], dct_prop['charge'])

        #Other properties
        # # Forces
        floss_v, fmse_v, fmae_v = self.loss_fn(batch['forces'], dct_prop['forces'])
        # # Dipole moment
        dloss_v, dmse_v, dmae_v = self.loss_fn(batch['dipole'], dct_prop['dipole'])

        # Atomic charges
        qaloss_v, qamse_v, qamae_v = self.loss_fn(
            batch.get('atomic_charges'), dct_prop.get('atomic_charges'))

        weights = self.args.get("trainer_weigths_properties")
        props = [
            'energy', 'forces', 'charge', 'dipole', 'atomic_charges',
            'nhlambda']
        for prop in props:
            if prop not in weights:
                weights[prop] = 1.0
        loss_v = (weights.get('energy') * eloss_v
                  + weights.get('forces') * floss_v
                  + weights.get('charge') * qloss_v
                  + weights.get('dipole') * dloss_v
                  + weights.get('atomic_charges') * qaloss_v
                  + weights.get('nhlambda') * dct_prop['nhloss']
                  + self.l2_regularizer(self.model))
        batch_size = batch["atoms_number"].size(0)

        quantities = self.averaging_quantities(
            quantities, batch_size, loss_v, emse_v, emae_v,
            fmse_v, fmae_v, qmse_v, qmae_v, dmse_v, dmae_v)

        return quantities

    def train(self, train_loader, val_loader, verbose=True):

        # TODO: Mix with the optimizer class
        if utils.is_object(self.optimizer):
            optimizer = self.optimizer
        else:
            optimizer_param = self.args.get("trainer_optimizer_args")
            optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                lr=optimizer_param.get('lr'),
                #weight_decay=optimizer_param.decay_rate, 
                amsgrad=True)

        if utils.is_object(self.scheduler):
            lr_schedule = self.scheduler
        else:
            scheduler_param = self.args.get("trainer_scheduler_args")
            #lr_schedule = torch.optim.lr_scheduler.ExponentialLR(
                #optimizer,
                #gamma=np.power(
                    #scheduler_param.decay_rate,
                    #1./scheduler_param.decay_steps))
            lr_schedule = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=scheduler_param.get('gamma'))

        # Initialize EMA
        #if self.args.get('ema'):
            #ema = ExponentialMovingAverage(self.model.parameters(), decay=self.args.ema_decay)

        # File management
        summary_writer, best_dir, ckpt_dir = utils.file_managment(self.args)

        # Initiate epoch and step counter
        step = 1
        epoch_i = 1
        # Initiate checkpoints and load the latest checkpoint

        latest_ckpt = utils.load_checkpoint(self.args.get('checkpoint_file'))
        if latest_ckpt is not None:
            self.model.load_state_dict(latest_ckpt['model_state_dict'])
            optimizer.load_state_dict(latest_ckpt['optimizer_state_dict'])
            epoch_i = latest_ckpt['epoch']

        # Initialize counter for estimated time per epoch
        time_train_estimation = np.nan
        time_train = 0.0
        best_loss = np.Inf
        # Training loop
        # Terminate training when maximum number of iterations is reached
        for epoch in range(epoch_i,self.args.get('trainer_max_epochs')):
            # Reset error averages
            quantities_t = self.reset_averages(forces=True,charges=True,Dipole=True) #TODO: Use option to activate deactivate forces, charge, etc.
            # Start train timer
            train_start = time()

            # Iterate over batches
            N_train_batches = len(train_loader)
            for ib, batch in enumerate(train_loader):
                optimizer.zero_grad()
                # Start batch timer
                batch_start = time()

                # Show progress bar
                if verbose:
                    utils.printProgressBar(
                        ib, N_train_batches, prefix="Epoch {0: 5d}".format(
                            epoch),
                        suffix=(
                            "Complete - Remaining Epoch Time: "
                            + "{0: 4.1f} s     ".format(time_train_estimation)
                            ),
                        length=42)

                # Training step
                quantities_t = self.train_step(batch, quantities_t)

                optimizer.step()
                
                #print(list(self.model.parameters()))
                
                #if self.args.get('ema'):
                    #ema.update()

                # Stop batch timer
                batch_end = time()

                # Actualize time estimation
                if verbose:
                    if ib == 0:
                        time_train_estimation = (
                                (batch_end - batch_start) * (N_train_batches - 1))
                    else:
                        time_train_estimation = (
                                0.5 * (time_train_estimation - (batch_end - batch_start))
                                + 0.5 * (batch_end - batch_start) * (N_train_batches - ib - 1))

                # Increment step number
                step = step + 1

            # Stop train timer
            train_end = time()
            time_train = train_end - train_start

            # Show final progress bar and time
            if verbose:
                lat = quantities_t['loss_avg'][0]
                utils.printProgressBar(
                    N_train_batches, N_train_batches,
                    prefix="Epoch {0: 5d}".format(epoch),
                    suffix=("Done - Epoch Time: "
                            + "{0: 4.1f} s, Average Loss: {1: 4.4f}   ".format(
                                time_train, lat)))  # length=42))

            # Save progress
            if (epoch % self.args.get('trainer_save_interval') == 0):
                number_of_ckpt = epoch//self.args.get('trainer_save_interval')
                utils.save_checkpoint(ckpt_dir,model=self.model, epoch=epoch, optimizer=optimizer, name_of_ckpt=number_of_ckpt)

            # Check performance on the validation set
            # Start of validation step
            if (epoch % self.args.get('trainer_validation_interval')) == 0:
                # Update training results
                quantities_t['time_train'] = time_train
                results_t = self.reporter(quantities_t,train=True)
                # Write Results to tensorboard
                #for key, value in results_t.items():
                    #summary_writer.add_scalar(key, value, global_step=epoch)

                # Reset error averages
                quantities_v = self.reset_averages(
                    forces=True, charges=True, Dipole=True)
                # Start validation timer
                val_start = time()
                # Iterate over batches for validation
                for ib, batch in enumerate(val_loader):
                    # Validation step
                    quantities_v = self.valid_step(batch, quantities_v)

                # Stop validation timer
                val_end = time()
                time_val = val_end - val_start

                # Update validation results
                quantities_v['time_val'] = time_val
                results_v = self.reporter(quantities_v, valid=True)

                # Write Results to tensorboard
                #for key, value in results_v.items():
                    #summary_writer.add_scalar(key, value, global_step=epoch)

                # Save best model
                if quantities_v['loss_avg'] < best_loss:
                    best_loss = quantities_v['loss_avg']
                    results_b = self.reporter(quantities_v)
                    #utils.save_checkpoint(
                        #model=self.model, epoch=epoch,
                        #optimizer=optimizer, best=True)
                    #for key, value in results_b.items():
                        #summary_writer.add_scalar(key, value, global_step=epoch)

                # Print summary
                if (verbose):
                    if (not epoch % self.args.get('trainer_validation_interval')
                        and epoch):
                        print(
                              "Summary Epoch: " + \
                               str(epoch) + '/' + str(self.args.get('trainer_max_epochs')),
                               "\n    Loss   train/valid: {0: 1.3e}/{1: 1.3e}, ".format(
                                   results_t["train_loss"][0],
                                   results_v["valid_loss"][0]),
                               " Best valid loss:   {0: 1.3e}, ".format(results_b["best_loss"][0]),
                               "\n    MAE(E) train/valid: {0: 1.3e}/{1: 1.3e}, ".format(
                                    results_t["train_emae"],
                                    results_v["valid_emae"]),
                               " Best valid MAE(E): {0: 1.3e}, ".format(
                                   results_b["best_emae"]))

            # Increment epoch number
            lr_schedule.step()
            epoch += 1






