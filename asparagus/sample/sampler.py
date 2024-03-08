import os
import queue
import logging
import threading
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import ase
from ase import optimize

from .. import sample

from .. import data
from .. import settings
from .. import utils
from .. import interface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['Sampler']

# ======================================
# Threadsafe Iterator Class
# ======================================

class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

# ======================================
# General Conformation Sampler Class
# ======================================

class Sampler:
    """
    Conformation Sampler main class for generation of reference structures.


    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    sample_data_file: str, optional, default None
        Database file name to store a selected set of systems with
        computed reference data. If None, data file name is the respective
        sample method tag.
    sample_data_file_format: str, optional, default None
        Database file format. If None, data file prefix is taken as file
        format tag.
    sample_directory: str, optional, default None
        Working directory where to store eventually temporary ASE
        calculator files, ASE trajectory files and/or model calculator
        files. If None, files will be stored in parent directory.
    sample_system_queue: queue.Queue, optional, default None
        Queue object including sample systems or where 'sample_systems' input
        will be added. If not defined, an empty queue will be assigned.
    sample_systems: (str, list, object), optional, default ''
        System coordinate file or a list of system coordinate files or
        ASE atoms objects that are considered as initial conformations for
        reference structure sampling.
    sample_systems_format: (str, list), optional, default ''
        System coordinate file format string (e.g. 'xyz') for the
        definition in 'sample_systems' in case of file paths.
    sample_calculator: (str, callable object), optional, default 'XTB'
        Definition of the ASE calculator type for reference data
        computation. The input can be either directly a ASE calculator
        class object or a string with available ASE calculator classes.
    sample_calculator_args: dict, optional, default {}
        In case of string type input for 'sample_calculator', this
        dictionary is passed as keyword arguments at the initialization
        of the ASE calculator.
    sample_num_threads: int, optional, default 1
        Number of parallel threads of property calculation using the sample
        calculator. Default is 1 (serial computation). Parallel computation
        is not possible for all sampling methods but for:
            Sampler, NMSampler, NMScanner
    sample_properties: List[str], optional, default None
        List of system properties which are computed by the ASE
        calculator class. Requested properties will be checked with the
        calculator available property list and return an error when one
        requested property is unavailable. By default all available
        properties will be stored.
    sample_systems_optimize: bool, optional, default False
        Instruction flag if the system coordinates shall be
        optimized using the ASE calculator defined by 'sample_calculator'.
    sample_systems_optimize_fmax: float, optional, default 0.01
        Instruction flag, if the system coordinates shall be
        optimized using the ASE calculator defined by 'sample_calculator'.
    sample_data_overwrite: bool, optional, default False
        If False, add new sampling data to an eventually existing data
        file. If True, overwrite an existing one.
    sample_tag: str, optional, default 'sample'
        Sampling method tag of the specific sampling methods for
        log and ASE trajectory files or the data file name if not defined.

    Returns
    -------
    callable object
        Sampler class object
    """

    # Default arguments for sample module
    _default_args = {
        'sample_directory':             None,
        'sample_data_file':             None,
        'sample_data_file_format':      None,
        'sample_systems_queue':         None,
        'sample_systems':               None,
        'sample_systems_format':        None,
        'sample_calculator':            'XTB',
        'sample_calculator_args':       {},
        'sample_save_trajectory':       False,
        'sample_num_threads':           1,
        'sample_properties':            ['energy', 'forces', 'dipole'],
        'sample_systems_optimize':      False,
        'sample_systems_optimize_fmax': 0.001,
        'sample_data_overwrite':        False,
        'sample_tag':                   'sample',
        }

    # Expected data types of input variables
    _dtypes_args = {
        'sample_directory':             [utils.is_string, utils.is_None],
        'sample_data_file':             [utils.is_string, utils.is_None],
        'sample_data_file_format':      [utils.is_string, utils.is_None],
        'sample_systems':               [utils.is_None,
                                         utils.is_string,
                                         utils.is_string_array,
                                         utils.is_ase_atoms,
                                         utils.is_ase_atoms_array],
        'sample_systems_format':        [utils.is_None,
                                         utils.is_string, 
                                         utils.is_string_array],
        'sample_calculator':            [utils.is_string, 
                                         utils.is_object],
        'sample_calculator_args':       [utils.is_dictionary],
        'sample_save_trajectory':       [utils.is_bool],
        'sample_num_threads':           [utils.is_integer],
        'sample_properties':            [utils.is_string, 
                                         utils.is_string_array],
        'sample_systems_optimize':      [utils.is_bool, 
                                         utils.is_boolean_array],
        'sample_systems_optimize_fmax': [utils.is_numeric],
        'sample_data_overwrite':        [utils.is_bool],
        'sample_tag':                   [utils.is_string],
        }

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None, 
        sample_data_file: Optional[str] = None,
        sample_data_file_format: Optional[str] = None,
        sample_directory: Optional[str] = None,
        sample_systems_queue: Optional[queue.Queue] = None,
        sample_systems: Optional[Union[str, List[str], object]] = None,
        sample_systems_format: Optional[Union[str, List[str]]] = None,
        sample_calculator: Optional[Union[str, object]] = None,
        sample_calculator_args: Optional[Dict[str, Any]] = None,
        sample_save_trajectory: Optional[bool] = None,
        sample_num_threads: Optional[int] = None,
        sample_properties: Optional[List[str]] = None,
        sample_systems_optimize: Optional[bool] = None,
        sample_systems_optimize_fmax: Optional[float] = None,
        sample_data_overwrite: Optional[bool] = None,
        sample_tag: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Sampler class.
        """

        #####################################
        # # # Check Sampler Class Input # # #
        #####################################

        # Get configuration object
        config = settings.get_config(
            config, config_file, config_from=self)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = config.set(
            instance=self,
            argitems=locals().items(),
            argsskip=['self', 'config', 'metadata', 'kwargs', '__class__'],
            check_default=utils.get_default_args(self, sample),
            check_dtype=utils.get_dtype_args(self, sample)
        )

        # Set global configuration as class parameter
        self.config = config

        # Check system input
        if self.sample_systems is None and self.sample_systems_queue is None:
            logger.warning(
                "WARNING:\nNo input in 'sample_systems' is given!\n"
                + "Please provide either a chemical structure file or "
                + "an ASE Atoms object as initial sample structure.")
            self.sample_systems = []

        ############################
        # # # Prepare Sampling # # #
        ############################

        # Initialize sampling counter
        if config.get('sample_counter') is None:
            self.sample_counter = 1
        else:
            self.sample_counter = config.get('sample_counter') + 1

        # Generate working directory
        if self.sample_directory is None or not len(self.sample_directory):
            self.sample_directory = '.'
        elif not os.path.exists(self.sample_directory):
            os.makedirs(self.sample_directory)

        # Check sample data file
        if self.sample_data_file is None:
            self.sample_data_file = f'{self.sample_tag:s}.db'
        elif not utils.is_string(self.sample_data_file):
            raise ValueError(
                "Sample data file 'sample_data_file' must be a string "
                + "of a valid file path but is of type "
                + f"'{type(self.sample_data_file)}'.")
        self.sample_data_file_format = sample_data_file_format

        # Define sample log file path and trajectory file
        self.sample_log_file = os.path.join(
            self.sample_directory,
            f'{self.sample_counter:d}_{self.sample_tag:s}.log')
        self.sample_trajectory_file = os.path.join(
            self.sample_directory, 
            f'{self.sample_counter:d}_{self.sample_tag:s}_{{:d}}.traj')

        #####################################
        # # # Prepare Sample Calculator # # #
        #####################################

        # Get ASE calculator
        ase_calculator, ase_calculator_tag = (
            interface.get_ase_calculator(
                self.sample_calculator,
                self.sample_calculator_args)
            )

        # Check requested system properties
        self.check_properties(ase_calculator)

        # Store calculator tag name
        self.sample_calculator_tag = ase_calculator_tag
        
        # Check number of calculation threads
        if self.sample_num_threads <= 0:
            raise ValueError(
                "Number of sample threads 'sample_num_threads' must be "
                + "larger or equal 1, but "
                + f"'{self.sample_num_threads:d}' is given!")

        #############################
        # # # Prepare Optimizer # # #
        #############################

        if self.sample_systems_optimize:

            # Assign ASE optimizer
            self.optimizer_tag = "bfgs"
            self.ase_optimizer = optimize.BFGS

        #####################################
        # # # Initialize Sample DataSet # # #
        #####################################

        self.sample_dataset = data.DataSet(
            self.sample_data_file,
            data_file_format=self.sample_data_file_format,
            data_load_properties=self.sample_properties,
            data_unit_properties=self.sample_unit_properties,
            data_overwrite=self.sample_data_overwrite)

    def __str__(self):
        """
        Return class descriptor
        """
        return "Sampler class"

    def read_systems(
        self,
        sample_systems_queue,
        sample_systems,
        sample_systems_format
    ):
        """
        Iterator to read next sample system and return as ASE atoms object
        """
        
        # Check system and format input
        if sample_systems is None or not len(sample_systems):
            return sample_systems_queue, ['Queue']
        
        if not utils.is_array_like(sample_systems):
            sample_systems = [sample_systems]
        
        if sample_systems_format is None:
            sample_systems_format = [None]*len(sample_systems)
        elif utils.is_string(sample_systems_format):
            sample_systems_format = (
                [sample_systems_format]*len(sample_systems))
        elif len(sample_systems) != len(sample_systems_format):
            raise ValueError(
                "Sample system input 'sample_systems' and "
                + "'sample_systems_format' have different input size of "
                + f"{len(sample_systems):d} and "
                + f"{len(sample_systems_format):d}, respectively.")

        # Iterate over system input and eventually read file to store as
        # (ASE Atoms object, index, sample source)
        for isample, (source, source_format) in enumerate(
            zip(sample_systems, sample_systems_format)
        ):

            # Check for ASE Atoms object or read system file
            if utils.is_ase_atoms(source):
                sample_systems_queue.put((source, isample, str(source), 1))
            else:
                isys=0
                complete = False
                while not complete:
                    try:
                        system = ase.io.read(
                            source, index=isys, format=source_format)
                        sample_systems_queue.put(
                            (system, isample, source, isys)
                            )
                    except (StopIteration, AssertionError):
                        complete = True
                    else:
                        isys += 1

        return sample_systems_queue, sample_systems

    def assign_calculator(
        self,
        sample_system: ase.Atoms,
        sample_calculator: Optional[Union[str, object]] = None,
        sample_calculator_args: Optional[Dict[str, Any]] = None,
        ithread: Optional[int] = None,
    ):
        """
        Assign calculator to a list of sample ASE Atoms objects

        Parameters
        ----------
        sample_system : ase.Atoms
            ASE Atoms object to assign the calculator
        sample_calculator : (str, object), optional, default None
            ASE calculator object or string of an ASE calculator class
            name to assign to the sample systems
        sample_calculator_args : dict, optional, default None
            Dictionary of keyword arguments to initialize the ASE
            calculator
        ithread: int, optional, default None
            Thread number to avoid conflict between files written by the
            calculator.
        """

        # Check calculator input
        if sample_calculator is None:
            sample_calculator = self.sample_calculator
        if sample_calculator_args is None:
            sample_calculator_args = self.sample_calculator_args

        # Get ASE calculator
        ase_calculator, ase_calculator_tag = (
            interface.get_ase_calculator(
                sample_calculator,
                sample_calculator_args,
                ithread=ithread)
            )

        # Check requested system properties
        self.check_properties(ase_calculator)

        # Assign ASE calculator
        sample_system.set_calculator(ase_calculator)

        return sample_system

    def check_properties(
        self, 
        sample_calculator: object
    ):
        """
        Check requested sample properties and units with implemented properties
        of the calculator
        
        Parameters
        ----------
        sample_calculator: object
            ASE Calculator to compare requested and available properties
            from the calculator.
        """

        # Check requested system properties
        for prop in self.sample_properties:
            # TODO Special calculator properties list for special properties
            # not supported by ASE such as, e.g., charge, hessian, etc.
            if prop not in sample_calculator.implemented_properties:
                raise ValueError(
                    f"Requested property '{prop:s}' is not implemented "
                    + f"in the ASE calculator '{sample_calculator}'! "
                    + "Available ASE calculator properties are:\n"
                    + f"{sample_calculator.implemented_properties}")

        # Define positions and property units
        self.sample_unit_properties = {
            prop: interface.ase_calculator_units.get(prop)
            for prop in self.sample_properties}
        if 'positions' not in self.sample_unit_properties:
            self.sample_unit_properties['positions'] = 'Ang'
        self.sample_unit_positions = self.sample_unit_properties['positions']

        return

    def get_info(self):
        """
        Dummy function for sampling parameter dictionary
        """
        return {            
            'sample_data_file': self.sample_data_file,
            'sample_directory': self.sample_directory,
            'sample_systems': self.sample_systems,
            'sample_systems_format': self.sample_systems_format,
            'sample_calculator': self.sample_calculator_tag,
            'sample_calculator_args': self.sample_calculator_args,
            'sample_properties': self.sample_properties,
            'sample_systems_optimize': self.sample_systems_optimize,
            'sample_systems_optimize_fmax': self.sample_systems_optimize_fmax,
            'sample_data_overwrite': self.sample_data_overwrite,
        }

    def run(
        self,
        sample_systems_queue: Optional[queue.Queue] = None,
        sample_systems: Optional[Union[str, List[str], object]] = None,
        sample_systems_format: Optional[Union[str, List[str]]] = None,
        **kwargs
    ):
        """
        Perform sampling of all sample systems or a selection of them.
        """

        ################################
        # # # Check Sampling Input # # #
        ################################
        
        # Check input
        if sample_systems_queue is None:
            sample_systems_queue = self.sample_systems_queue
        if sample_systems is None:
            sample_systems = self.sample_systems
        if sample_systems_format is None:
            sample_systems_format = self.sample_systems_format

        # Collect sampling parameters
        config_sample_tag = f'{self.sample_counter}_{self.sample_tag}'
        config_sample = {
            config_sample_tag: self.get_info()
            }
        
        # Read sample systems into queue
        if sample_systems_queue is None:
            sample_systems_queue = queue.Queue()
        sample_systems_queue, sample_systems = self.read_systems(
            sample_systems_queue,
            sample_systems,
            sample_systems_format)

        # Update configuration file with sampling parameters
        if 'sampler_schedule' in self.config:
            config_sample = {
                **self.config['sampler_schedule'],
                **config_sample,
                }
        self.config.update({
            'sampler_schedule': config_sample,
            'sample_counter': self.sample_counter
            })

        # Print sampling overview
        msg = f"Perform sampling method '{self.sample_tag:s}' on systems:\n"
        for isys, system in enumerate(sample_systems):
            msg += f" {isys + 1:3d}. '{system:s}'\n"
        logger.info(f"INFO:\n{msg:s}")

        ##########################
        # # # Start Sampling # # #
        ##########################
        
        self.run_systems(
            sample_systems_queue=sample_systems_queue,
            **kwargs
            )

        # Increment sample counter
        self.sample_counter += 1

    def run_systems(
        self,
        sample_systems_queue: queue.Queue,
        **kwargs,
    ):
        """
        Iterate over systems using 'sample_num_threads' threads.
        """

        # Initialize thread continuation flag
        # Currently all False, might become useful later for adaptive sampling
        self.thread_keep_going = np.array(
            [True for ithread in range(self.sample_num_threads)],
            dtype=bool
            )
        
        # Add stop flag
        for _ in range(self.sample_num_threads):
            sample_systems_queue.put('stop')

        if self.sample_num_threads == 1:
            
            self.run_system(sample_systems_queue)
        
        else:

            # Create threads
            threads = [
                threading.Thread(
                    target=self.run_system, 
                    args=(sample_systems_queue, ),
                    kwargs={
                        'ithread': ithread}
                    )
                for ithread in range(self.sample_num_threads)]

            # Start threads
            for thread in threads:
                thread.start()

            # Wait for threads to finish
            for thread in threads:
                thread.join()
        
    def run_system(
        self, 
        sample_systems_queue: queue.Queue,
        ithread: Optional[int] = None,
    ):
        """
        Apply sample calculator on system input and write properties to 
        database.
        
        Parameters
        ----------
        sample_systems_queue: queue.Queue
            Queue of sample system information providing tuples of ase Atoms
            objects, index number and respective sample source and the total
            sample index.
        ithread: int, optional, default None
            Thread number
        """
        
        # Initialize stored sample counter
        Nsample = 0

        while self.keep_going(ithread):
            
            # Get sample parameters or wait
            sample = sample_systems_queue.get()
            
            # Check for stop flag
            if sample == 'stop':
                self.thread_keep_going[ithread] = False
                continue
            
            # Extract sample system to optimize
            (system, isample, source, index) = sample

            # Assign calculator
            system = self.assign_calculator(
                system, 
                ithread=ithread)

            # If requested, perform structure optimization
            if self.sample_systems_optimize:

                # Perform structure optimization
                system = self.run_optimization(
                    sample_system=system,
                    sample_index=isample)
                
            # Initialize trajectory file
            if self.sample_save_trajectory:
                self.sample_trajectory = Trajectory(
                    self.sample_trajectory_file.format(isample), atoms=system,
                    mode='a', properties=self.sample_properties)
            
            # Compute system properties
            system.calc.calculate(
                system,
                properties=self.sample_properties)
            
            # Store results
            Nsample = self.save_properties(system, Nsample)
            if self.sample_save_trajectory:
                self.write_trajectory(system, self.sample_trajectory)

        # Print sampling info
        msg = f"Sampling method '{self.sample_tag:s}' complete for system "
        msg += f"of index {index:d} from '{source}!'\n"
        if Nsample == 0:
            msg += f"No samples written to "
        if Nsample == 1:
            msg += f"{Nsample:d} sample written to "
        else:
            msg += f"{Nsample:d} samples written to "
        msg += f"'{self.sample_data_file:s}'.\n"
        
        logger.info(f"INFO:\n{msg:s}")

        return

    def run_optimization(
        self,
        sample_system: Optional[ase.Atoms] = None,
        sample_index: Optional[int] = None,
        sample_systems_queue: Optional[queue.Queue] = None,
        sample_optimzed_queue: Optional[queue.Queue] = None,
        ithread: Optional[int] = None,
    ):
        """
        Perform structure optimization on sample system
        
        Parameters
        ----------
        sample_system: ase.Atoms, optional, default None
            ASE Atoms object which will be optimized using the optimizer 
            defined in self.ase_optimizer.
        sample_systems_queue: queue.Queue, optional, default None
            Sample system queue cotaining ASE Atoms object which will be 
            optimized using the optimizer defined in self.ase_optimizer.
            If sample_system is not None, this queue will be ignored.
        sample_optimzed_queue: queue.Queue, optional, default None
            If defined, the optimized sample system will be put into the
            queue.
        ithread: int, optional, default None
            Thread number
        
        Returns
        -------
        ase.Atoms
            Optimized ASE atoms object
        """
        
        # Check sample system input
        if sample_system is None and sample_systems_queue is None:
            return None
        
        # Optimize sample system
        if sample_system is not None:
            
            # Prepare optimization log and trajectory file name
            ase_optimizer_log_file = os.path.join(
                self.sample_directory,
                f'{self.sample_counter:d}_{self.optimizer_tag:s}.log')
            ase_optimizer_trajectory_file = os.path.join(
                self.sample_directory,
                f'{self.sample_counter:d}_{self.optimizer_tag:s}.traj')
            
            # Assign calculator
            system = self.assign_calculator(sample_system)

            # Perform structure optimization
            self.ase_optimizer(
                system,
                logfile=ase_optimizer_log_file,
                trajectory=ase_optimizer_trajectory_file,
                ).run(
                    fmax=self.sample_systems_optimize_fmax)

            # Add optimized ASE atoms object to the queue if defined
            if sample_optimzed_queue is not None:
                sample_optimzed_queue.put((system, isample, str(system), 1))

            return system
        
        else:
            
            while self.keep_going(ithread):
                
                # Get sample parameters or wait
                sample = sample_systems_queue.get()
                
                # Check for stop flag
                if sample == 'stop':
                    self.thread_keep_going[ithread] = False
                    continue
                
                # Extract sample system to optimize
                (system, isample, source, index) = sample

                # Prepare optimization log and trajectory file name
                ase_optimizer_log_file = os.path.join(
                    self.sample_directory,
                    f'{self.sample_counter:d}_{self.optimizer_tag:s}'
                    + f'_{isample:d}.log')
                ase_optimizer_trajectory_file = os.path.join(
                    self.sample_directory,
                    f'{self.sample_counter:d}_{self.optimizer_tag:s}'
                    + f'_{isample:d}.traj')

                # Assign calculator
                system = self.assign_calculator(
                    system,
                    ithread=ithread)

                # Perform structure optimization
                try:

                    ase_optimizer = self.ase_optimizer(
                        system,
                        logfile=ase_optimizer_log_file,
                        trajectory=ase_optimizer_trajectory_file,
                        )
                    ase_optimizer.run(fmax=self.sample_systems_optimize_fmax)
                
                except ase.calculators.calculator.CalculationFailed:

                    msg = "ERROR:\nSingle point calculation of the system "
                    msg += f"from '{source}' of index {index:d} "
                    msg += "is not converged "
                    msg += "during structure optimization for sampling method "
                    msg += f"'{self.sample_tag:s}' "
                    msg += f"(see log file '{ase_optimizer_log_file:s}')!\n"
                    if sample_optimzed_queue is not None:
                        msg += "System will be skipped for further sampling.\n"
                    logger.error(msg)
                    
                else:

                    msg = "INFO:\nOptimization of system "
                    msg += f"from '{source}' of index {index:d} "
                    msg += f"is converged "
                    msg += f"(see log file '{ase_optimizer_log_file:s}').\n"
                    logger.info(msg)

                # Add optimized ASE atoms object to the queue if defined
                if sample_optimzed_queue is not None:
                    sample_optimzed_queue.put((system, isample, source, index))

        return

    def keep_going(
        self,
        ithread
    ):
        """
        Return thread continuation flag
        
        Parameters
        ----------
        ithread: int
            Thread number.
        """
        if ithread is None:
            return self.thread_keep_going[0]
        if hasattr(self, 'thread_keep_going'):
            return self.thread_keep_going[ithread]
        return False

    def get_properties(self, system):
        """
        Collect system properties and calculator results
        """

        return interface.get_ase_properties(system, self.sample_properties)

    def save_properties(self, system, Nsample):
        """
        Save system properties
        """
        
        system_properties = self.get_properties(system)
        self.sample_dataset.add_atoms(system, system_properties)
        Nsample += 1
        
        return Nsample

    def write_trajectory(self, system, sample_trajectory):
        """
        Write current image to trajectory file but without constraints
        """
        
        system_noconstraint = system.copy()
        system_noconstraint.calc = system.calc
        system_noconstraint.set_constraint()
        sample_trajectory.write(system_noconstraint)
