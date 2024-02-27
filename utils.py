import os
import numpy as np
import torch
import random
import re
import matplotlib.pyplot as plt



def check_directories(args):
    task_path = os.path.join(args.output_dir)
    if not os.path.exists(task_path):
        os.mkdir(task_path)
        print(f"Created {task_path} directory")
    
    folder = args.task
    
    save_path = os.path.join(task_path, folder)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print(f"Created {save_path} directory")
    args.save_dir = save_path

    cache_path = os.path.join(args.input_dir, 'cache')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
        print(f"Created {cache_path} directory")

    if args.debug:
        args.log_interval /= 10

    return args

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def setup_gpus(args):
    n_gpu = 0  # set the default to 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    args.n_gpu = n_gpu
    if n_gpu > 0:   # this is not an 'else' statement and cannot be combined
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return args


def plot_losses(train_losses, val_losses, fname):
    """
    Plots the training and validation losses across epochs and saves the plot as an image file with name - fname(function argument). 

    Args:
    train_losses (list): List of training losses for each epoch.
    val_losses (list): List of validation losses for each epoch.
    fname (str): Name of the file to save the plot (without extension).

    Returns:
    None
    """

    # Create 'plots' directory if it doesn't exist
    if not os.path.isdir('plots'):
        os.mkdir('plots')

    # # added this
    # train_losses = [t.cpu().detach().numpy() for t in train_losses]
    # val_losses = [t.cpu().detach().numpy() for t in val_losses]

    # Plotting training and validation losses
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    # Saving the plot as an image file in 'plots' directory
    plt.savefig("./plots/" + fname + ".png")
    plt.savefig("./plots/" + fname + ".svg")


def dumpArgs(args, fname):
    """
    Dumps the arguments to a file in the save_dir
    """
    if not os.path.isdir('args'):
        os.mkdir('args')

    with open(f"./args/{fname}.txt", 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

def get_name(args):
    # model_task_lr-lr_bs-batchsize_ep-epochs_dr-drop_rate_eps-epsilon_hdim-hidden
    return f"{args.model}_{args.task}_lr-{args.learning_rate}_bs-{args.batch_size}_ep-{args.n_epochs}_dr-{args.drop_rate}_eps-{args.adam_epsilon}_hdim-{args.hidden_dim}_scheduler-{args.scheduler}"