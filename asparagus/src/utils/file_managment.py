# This should manage checkpoint creation and loading
# writer to tensorboardX
import os
import torch
from .. import settings
import string
import random
import datetime
from tensorboardX import SummaryWriter

def file_managment(config, restart=False):
    
    if config['restart'] or restart:
        
        # This is the directory where the checkpoints are saved
        directory = config['save_dir'] 
        writer = None
        best_dir = os.path.join(directory, 'best')
        ckpt_dir = os.path.join(directory, 'checkpoints')
        
    else:
        
        directory = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # I would prefer if we keep the specifications of the NN model in the name of the directory...LIVS
        #(
                #datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
                #+ "_" + id_generator() + "_F" + str(config['input_n_atombasis'])
                #+ "K" + str(config['input_nradialbasis']) + "b" + str(config['graph_n_blocks'])
                #+ "a" + str(config['graph_n_residual_atomic'])
                #+ "i" + str(config['graph_n_residual_interaction'])
                #+ "o" + str(config['output_n_residual']) + "cut" + str(config['input_cutoff_descriptor'])
                #+ "e" + str(config['model_electrostatic']) + "d" + str(config['model_dispersion']) + "r" + str(config['model_repulsion']))
        config['model_directory'] = directory
        if not os.path.exists(directory):
            os.makedirs(directory)
            best_dir = os.path.join(directory, 'best')
            log_dir = os.path.join(directory, 'logs')
            ckpt_dir = os.path.join(directory, 'checkpoints')
            create_folders(ckpt_dir, log_dir, best_dir)
        
        writer = SummaryWriter(log_dir=log_dir)
        
    return writer,best_dir,ckpt_dir

# Generate an (almost) unique id for the training session
def id_generator(
    size=8, 
    chars=(
        string.ascii_uppercase
        + string.ascii_lowercase
        + string.digits)
    ):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

def create_folders(ckpt_dir, tb_dir, best_dir):
    """
    Create folders for checkpoints and tensorboardX
    """
    # Create folder for checkpoints
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    # Create folder for tensorboardX/logs in previous physnet_torch version
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    # Create folder for best model
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)


def save_checkpoint(
    dir,model, epoch, optimizer, scheduler, name_of_ckpt=None, best=False):
    
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch, 
        }
    
    if best:
        path = os.path.join(dir, 'best_model.pt')
    else:
        name = f'model_{name_of_ckpt:d}.pt'
        path = os.path.join(dir, name)

    torch.save(state, path)


def load_checkpoint(path):
    if path is not None:
        checkpoint = torch.load(path)
        return checkpoint
    else:
        return None
    

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

