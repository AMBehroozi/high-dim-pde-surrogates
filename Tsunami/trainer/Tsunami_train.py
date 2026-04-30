# %%
import os
import numpy as np
import torch
import sys
import argparse
import time
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split, dataset, Subset, DistributedSampler
import pandas as pd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import gc

sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

from lib.DerivativeComputer import batchJacobian_AD_Dist
from lib.util import MHPI, UnitGaussianNormalizer,run_nvidia_smi, split_dataset, prepare_data_loaders_low_rank
from lib.utilities3 import load_and_normalize_datasets, ensure_directory, count_parameters, compute_jacobian_blocks, upscale_tensor, scale_tensors_to_largest_order
from lib.utiltools import loss_live_plot, AutomaticWeightedLoss
from lib.batchJacobian import batchJacobian_PDE

from models.NO_3d_t_dist_DDP import create_model

from lib.low_rank_jacobian import *

# %%
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_model(rank, world_size, model_fn, awl_fn, learning_rate, 
                operator_type, T_in, T_out, S,

                width_CNO, depth_CNO, kernel_size, unet_depth,  # CNO inputs
                mode1, mode2, mode3, width_FNO,                 # FNO inputs
                wavelet, level, layers, grid_range, width_WNO,  # WNO inputs
                branch_layers, trunk_layers,                    # DeepONet inputs

                epochs, PATH_saved_models, PATH_saved_loss, Mode, num_samples_x_y, 
                enable_ig_loss, L_x, L_y, criterion, plot_live_loss, 
                PATH_dataset, case, train_size, eval_size, batch_size, train_dataset, eval_dataset):
    
    setup(rank, world_size)

    # Setup model for DDP
    model = model_fn.to(rank)  # model_fn should return a new model instance
    awl = awl_fn.to(rank)

    model = DDP(model, device_ids=[rank])
    awl = DDP(awl, device_ids=[rank])
    weight_decay_rate = 5e-4 

    optimizer = optim.Adam([
    {'params': model.parameters(), 'lr': learning_rate},
    {'params': awl.parameters()  , 'lr': learning_rate}
    ])

    scheduler = StepLR(optimizer, step_size=50, gamma=0.85)
        
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, sampler=eval_sampler)

    train_fnolosses, train_iglosses, val_losses = [], [], []
    outer_loop = tqdm(range(epochs), desc="Progress", position=0)
    torch.cuda.empty_cache()

    stage = True    

    for ep in outer_loop:
        # Training phase
        model.train()
        total_fnoloss = 0.0
        total_igloss = 0.0
        total_samples = 0

        for batch_data in train_loader:
            batch_data = [item.to(rank) for item in batch_data]
            epicentric, bed, depth, du_dp__low_rank_true = batch_data    # epicentric ([nb, np])  #[epicenter_x, epicenter_y, depth, length, width, slip, strike, dip, rake]
            jump = 1
            jump_timestep = 2
            if stage:
                batch_u_in =  depth[...,::jump, ::jump, ::jump_timestep][..., :T_in] + bed[..., ::jump, ::jump].unsqueeze(-1).repeat(1, 1, 1, T_in)
                batch_u_out = depth[...,::jump, ::jump, ::jump_timestep][..., T_in:] + bed[..., ::jump, ::jump].unsqueeze(-1).repeat(1, 1, 1, T_out)

                # Replace values greater than 5 with 10 using masking
                batch_u_in[batch_u_in > 5] = 10.0
                batch_u_out[batch_u_out > 5] = 10.0
            else:
                batch_u_in =  depth[...,::jump, ::jump, ::jump_timestep][..., :T_in] 
                batch_u_out = depth[...,::jump, ::jump, ::jump_timestep][..., T_in:]

            batch_parameter = 0.01 * bed[..., ::jump, ::jump]
            batch_parameter.requires_grad_(True) 

            batch_size = batch_u_in.shape[0]
            total_samples += batch_size
            
            optimizer.zero_grad()
            # batch_u_in: [nb, nx, ny, T_in]
            # batch_parameter: [nb, nx, ny]
            U_pred = model(batch_u_in, batch_parameter)
            # U_pred: [nb, nx, ny, T_out]
            data_loss = criterion(U_pred, batch_u_out)

            jac_rank = 5
            if enable_ig_loss:
                U_mat_pred, V_mat_pred = compute_low_rank_jacobian_1(model, U_pred, batch_parameter, batch_u_in, rank=5, epsilon=1e-1, seed=None)
                ig_loss = compute_low_rank_jacobian_loss(du_dp__low_rank_true, U_mat_pred, V_mat_pred, method='action')

                loss = awl(data_loss, ig_loss)
                fnoloss = data_loss
                ig_loss = ig_loss

            else:
                ig_loss = torch.zeros_like(data_loss)
                loss = data_loss
                fnoloss = data_loss
                
            loss.backward()
            optimizer.step()

            total_igloss += ig_loss.item() * batch_size
            total_fnoloss += fnoloss.item() * batch_size
            
        epoch_fnoloss = total_fnoloss / total_samples
        epoch_igloss = total_igloss / total_samples
        train_fnolosses.append(epoch_fnoloss)
        train_iglosses.append(epoch_igloss)

        # Evaluation phase
        model.eval()
        total_fnoloss = 0.0
        total_samples = 0
        for batch_data in eval_loader:

            batch_data = [item.to(rank) for item in batch_data]
            epicentric, bed, depth , jac = batch_data

            if stage:
                batch_u_in =  depth[...,::jump, ::jump, ::jump_timestep][..., :T_in] + bed[..., ::jump, ::jump].unsqueeze(-1).repeat(1, 1, 1, T_in)
                batch_u_out = depth[...,::jump, ::jump, ::jump_timestep][..., T_in:] + bed[..., ::jump, ::jump].unsqueeze(-1).repeat(1, 1, 1, T_out)

                # Replace values greater than 5 with 10 using masking
                batch_u_in[batch_u_in > 5] = 10.0
                batch_u_out[batch_u_out > 5] = 10.0
            else:
                batch_u_in =  depth[...,::jump, ::jump, ::jump_timestep][..., :T_in] 
                batch_u_out = depth[...,::jump, ::jump, ::jump_timestep][..., T_out:]

            du_dparam_true = jac

            batch_size = batch_u_in.shape[0]
            total_samples += batch_size
            
            with torch.no_grad():
                U_pred = model(batch_u_in, 0.01 * bed[..., ::jump, ::jump])
                data_loss = criterion(U_pred, batch_u_out)
            
            total_fnoloss += data_loss.item() * batch_size
        
        epoch_fnoloss = total_fnoloss / total_samples
        val_losses.append(epoch_fnoloss)

        losses_dict = {'Training FNO Loss': train_fnolosses, 'Train IG loss': train_iglosses, 'Validation Loss': val_losses}       

        df = pd.DataFrame(losses_dict)
        save_results = False
        if save_results:
            if ep % 5 == 0 and rank == 0:
                torch.save({
                    # General configuration
                    'operator_type': operator_type,
                    'enable_ig_loss': enable_ig_loss,
                    'S': S,
                    'T_in': T_in,
                    'T_out': T_out,
                    
                    # CNO inputs
                    'width_CNO': width_CNO,
                    'depth_CNO': depth_CNO,
                    'kernel_size': kernel_size,
                    'unet_depth': unet_depth,
                    
                    # FNO inputs
                    'mode1': mode1,
                    'mode2': mode2,
                    'mode3': mode3,
                    'width_FNO': width_FNO,
                    
                    # WNO inputs
                    'wavelet': wavelet,
                    'level': level,
                    'layers': layers,
                    'grid_range': grid_range,
                    'width_WNO': width_WNO,
                    
                    # DeepONet inputs
                    'branch_layers': branch_layers,
                    'trunk_layers': trunk_layers,
                    
                    # Training state
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_fnoloss,
                }, PATH_saved_models + f'saved_model_{Mode}.pth')
                df.to_excel(PATH_saved_loss + f'losses_data_{Mode}.xlsx', index=True, engine='openpyxl')

        if plot_live_loss and ep % 1 == 0:
            loss_live_plot(losses_dict)

        scheduler.step()
        outer_loop.set_description(f"Progress (Epoch {ep + 1}/{epochs}) Mode: {Mode}")
        outer_loop.set_postfix(fnoloss=f'{epoch_fnoloss:.2e}', ig_loss=f'{epoch_igloss:.2e}' , eval_loss=f'{val_losses[-1]:.2e}')

    cleanup()

# %%
def main(enable_ig_loss, PATH_dataset, train_size, eval_size, case, num_samples_x_y, batch_size, epochs, learning_rate, scheduler_step, scheduler_gamma, 
        operator_type, T_in, T_out, nx, ny, 
        width_CNO=None, depth_CNO=None, kernel_size=None, unet_depth=None,  # CNO inputs
        mode1=None, mode2=None, mode3=None, width_FNO=None,  # FNO inputs
        wavelet=None, level=None, layers=None, grid_range=None, width_WNO=None,  # WNO inputs
        branch_layers=None, trunk_layers=None):
    
    world_size = torch.cuda.device_count()
    print(f'Number of GPUs: {world_size}\n')
    
    plot_live_loss = False
    
    L_x = 1.0
    L_y = 1.0
    S = nx
    # optimizer and training configurations
    Mode = f"{case}_{'IG_Enable' if enable_ig_loss else 'IG_Disable'}_S_{nx}_Tin_{T_in}_Tout_{T_out}_Samp_{num_samples_x_y}_{operator_type}_DDP_{train_size}"
    print(Mode)

    main_path = 'Tsunami/experiments/' + case + '/'
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_dataset, eval_dataset = prepare_data_loaders_low_rank(PATH_dataset, case, train_size, eval_size, batch_size, rank=5)
    # train_dataset, eval_dataset = None, None

    PATH_saved_models =      main_path + 'saved_models/'
    PATH_saved_loss =        main_path + 'loss_epoch/'

    ensure_directory(PATH_saved_models)
    ensure_directory(PATH_saved_loss)

    #  Model setup
    # model_fn = FNO3dDDP(T_in=T_in, T_out=T_out, modes_x=mode1, modes_y=mode2, modes_t=mode3, width=width)

    model_fn = create_model(operator_type, T_in, T_out, nx, ny, 
                    width_CNO=width_CNO, depth_CNO=depth_CNO, kernel_size=kernel_size, unet_depth=unet_depth,  # CNO inputs
                    mode1=mode1, mode2=mode2, mode3=mode3, width_FNO=width_FNO,  # FNO inputs
                    wavelet=wavelet, level=level, layers=layers, grid_range=grid_range, width_WNO=width_WNO,  # WNO inputs
                    branch_layers=branch_layers, trunk_layers=trunk_layers)

    ss = 2 if enable_ig_loss else 1
    awl_fn = AutomaticWeightedLoss(ss)
    criterion=nn.MSELoss()

    torch.multiprocessing.spawn(train_model,
                                args=(world_size, model_fn, awl_fn, learning_rate, 
                                    operator_type, T_in, T_out, S,

                                    width_CNO, depth_CNO, kernel_size, unet_depth,  # CNO inputs
                                    mode1, mode2, mode3, width_FNO,                 # FNO inputs
                                    wavelet, level, layers, grid_range, width_WNO,  # WNO inputs
                                    branch_layers, trunk_layers,                    # DeepONet inputs

                                    epochs, PATH_saved_models, PATH_saved_loss, Mode, num_samples_x_y,
                                    enable_ig_loss, L_x, L_y, criterion, plot_live_loss,
                                    PATH_dataset, case, train_size, eval_size, batch_size, train_dataset, eval_dataset),
                                nprocs=world_size,
                                join=True)

# %%
if __name__ == "__main__":
    # System and environment setup
    run_nvidia_smi()  # Check GPU status
    MHPI()           # Initialize MHPI (if this is a custom function)

    # Dataset and case configuration
    PATH_dataset = '/storage/work/amb10399/project/SC-FNO-DIST/Tsunami/datasets/1000__3e5_8e5__2e5_8e5_with_jac_multi_fault_multi_epi_pert_bed/'
    case = 'Tsunami_multi_epicentric_multi_fault_pert_bed'
    enable_ig_loss = False  # Enable/disable IG loss

    # Dataset sizes
    train_size = 700
    eval_size = 150

    # Sampling configuration
    num_samples_x_y = 'All_TEST_low_rank'  # Number of random samples along x, y axes for Jacobian calculations

    # Training hyperparameters
    batch_size = 16
    epochs = 500
    learning_rate = 0.005
    scheduler_step = 100
    scheduler_gamma = 0.95

    # Model configuration
    operator_type = 'DeepONet'  # Options: 'CNO', 'FNO', 'WNO', 'DeepONet'

    # Temporal and spatial dimensions
    total_steps = 51
    T_in = 1
    T_out = total_steps - T_in
    nx = 100
    ny = 100

    # Model-specific inputs
    # CNO inputs
    width_CNO, depth_CNO = 128, 4
    kernel_size = 3
    unet_depth = 4

    # FNO inputs
    mode_x = 8
    mode_y = 8
    mode_t = 8
    width_FNO = 20

    # WNO inputs
    wavelet = 'db6'  # Wavelet basis function
    level = 4        # Level of wavelet decomposition
    width_WNO = 30   # Uplifting dimension
    layers = 4       # Number of wavelet layers
    grid_range = [1, 1, 1]

    # DeepONet inputs

    branch_layers = [64, 128, 128, 128, 64]
    trunk_layers =  [64, 128, 128, 128, 64]

    # Call the main function with organized inputs
    main(
        enable_ig_loss, PATH_dataset, train_size, eval_size, case, 
        num_samples_x_y, batch_size, 
        epochs, learning_rate, scheduler_step, scheduler_gamma, 
        operator_type, T_in, T_out, nx=nx, ny=ny, 
        # CNO inputs
        width_CNO=width_CNO if operator_type == 'CNO' else None,
        depth_CNO=depth_CNO if operator_type == 'CNO' else None,
        kernel_size=kernel_size if operator_type == 'CNO' else None,
        unet_depth=unet_depth if operator_type == 'CNO' else None,
        # FNO inputs
        mode1=mode_x if operator_type == 'FNO' else None,
        mode2=mode_y if operator_type == 'FNO' else None,
        mode3=mode_t if operator_type == 'FNO' else None,
        width_FNO=width_FNO if operator_type == 'FNO' else None,
        # WNO inputs
        wavelet=wavelet if operator_type == 'WNO' else None,
        level=level if operator_type == 'WNO' else None,
        layers=layers if operator_type == 'WNO' else None,
        grid_range=grid_range if operator_type == 'WNO' else None,
        width_WNO=width_WNO if operator_type == 'WNO' else None,
        # DeepONet inputs
        branch_layers=branch_layers if operator_type == 'DeepONet' else None,
        trunk_layers=trunk_layers if operator_type == 'DeepONet' else None
    )

#%%