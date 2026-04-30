# %%
import os
import gc
import argparse
import time
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import sys
# --------------------------------------------------------------------------- #
# Project-specific paths
# --------------------------------------------------------------------------- #
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

# --------------------------------------------------------------------------- #
# Custom modules (adjust import paths if your project structure changes)
# --------------------------------------------------------------------------- #
from lib.DerivativeComputer import batchJacobian_AD_Dist
from lib.util import (
    MHPI,
    UnitGaussianNormalizer,
    run_nvidia_smi,
    split_dataset,
    prepare_data_loaders_low_rank,
)
from lib.utilities3 import (
    load_and_normalize_datasets,
    ensure_directory,
    count_parameters,
    compute_jacobian_blocks,
    upscale_tensor,
    scale_tensors_to_largest_order,
)
from lib.utiltools import loss_live_plot, AutomaticWeightedLoss
from lib.batchJacobian import batchJacobian_PDE
from lib.low_rank_jacobian import *  # noqa: F403

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
from models.FNO_3d_t_dist_DDP import FNO3d

# %%
def refine_grid_differentiable(field, N):
    """
    Differentiable grid refinement using repeat operations.
    Gradients flow back to the original grid points.
    """
    # [nb, nx, ny] -> [nb, nx, ny, N, N]
    expanded = field.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, N, N)
    
    # [nb, nx, ny, N, N] -> [nb, nx, N, ny, N] -> [nb, N*nx, N*ny]
    refined = expanded.permute(0, 1, 3, 2, 4).contiguous()
    refined = refined.view(field.size(0), N*field.size(1), N*field.size(2))
    
    return refined

# %%

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12344'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_model(rank, world_size, model_fn, awl_fn, learning_rate, 
                operator_type, jump_timestep, T_in, T_out, S,

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
    select = 1
    for ep in outer_loop:
        # Training phase
        model.train()
        total_fnoloss = 0.0
        total_igloss = 0.0
        total_samples = 0

        for batch_data in train_loader:
            batch_data = [item.to(rank) for item in batch_data]
            _, W0_batch, sol_batch, jac_W0_batch, _ = batch_data    # epicentric ([nb, np])  #[epicenter_x, epicenter_y, depth, length, width, slip, strike, dip, rake]
            
            
            batch_size = W0_batch.shape[0]
            total_samples += batch_size
           
            sol_true = sol_batch[..., ::jump_timestep][..., T_in:]
            sol_in = sol_batch[..., ::jump_timestep][..., :T_in]
            W0_batch.requires_grad_(True) 
            w0 = refine_grid_differentiable(W0_batch, int(sol_batch.shape[1]/W0_batch.shape[1]))
            
            optimizer.zero_grad()

            sol_Pred = model(sol_in, w0)  # stage_Pred: [nb, nx, ny, T_out]
            
            data_loss = criterion(sol_Pred[:, ::select, ::select, :], sol_true[:, ::select, ::select, :])

            if enable_ig_loss:
                
                num_samples_x = 2
                num_samples_y = 2

                downscale_factor_data = 4  # ::4
                
                downscale_factor = 2

                jac_W0_batch = upscale_tensor(jac_W0_batch, downscale_factor_data)
                jac_W0_batch = jac_W0_batch[:, ::downscale_factor, ::downscale_factor, ...]

                high_res_x = sol_Pred.shape[1]  # e.g., 64
                high_res_y = sol_Pred.shape[2]  # 64
                low_res_x = high_res_x // downscale_factor
                low_res_y = high_res_y // downscale_factor
                
                sol_Pred_downscale = sol_Pred[:, ::downscale_factor, ::downscale_factor, :]  # [B, low_res_x, low_res_y, T_out]
                
                random_indices_x = torch.randperm(low_res_x)[:num_samples_x]  # CPU tensor
                random_indices_y = torch.randperm(low_res_y)[:num_samples_y]  # CPU tensor
                
                device = sol_Pred.device
                random_indices_x = random_indices_x.to(device)
                random_indices_y = random_indices_y.to(device)
                
                sol_Pred_select = sol_Pred_downscale[..., -1]  # [B, low_res_x, low_res_y]
                
                sol_Pred_select = sol_Pred_select[:, random_indices_x, :]  # [B, 3, low_res_y]
                
                sol_Pred_select = sol_Pred_select[:, :, random_indices_y]  # [B, 3, 3]
                
                du_dp_pred_selected = compute_jacobian_blocks(sol_Pred_select, W0_batch, block_size=1)
                
                du_dp_true_selected = jac_W0_batch[:, random_indices_x, :, :, :]
                du_dp_true_selected = du_dp_true_selected[:, :, random_indices_y, :, :]
                
                ig_loss = criterion(du_dp_pred_selected, du_dp_true_selected)
                
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

        with torch.no_grad():  # Efficient inference
            for batch_data in eval_loader:
                batch_data = [item.to(rank) for item in batch_data]
                _, W0_batch, sol_batch, _, _ = batch_data  # Ignore jac (not needed)

                batch_size = W0_batch.shape[0]
                total_samples += batch_size

                # True solution: time-subsampled, skip t=0
                sol_true = sol_batch[..., ::jump_timestep][..., T_in:]
                sol_in = sol_batch[..., ::jump_timestep][..., :T_in]

                # Upsample low-res input
                w0 = refine_grid_differentiable(W0_batch, int(sol_batch.shape[1] / W0_batch.shape[1]))

                # Forward
                sol_Pred = model(sol_in, w0)  # [B, N, N, T_out]

                # Data loss only
                data_loss = criterion(sol_Pred[:, ::select, ::select, :], sol_true[:, ::select, ::select, :])

                total_fnoloss += data_loss.item() * batch_size

        # Epoch average
        epoch_fnoloss = total_fnoloss / total_samples
        val_losses.append(epoch_fnoloss)  # Keep existing list name

        # Update dict (only FNO losses)
        losses_dict = {
            'Training FNO Loss': train_fnolosses,
            'Train IG Loss': train_iglosses,    # Still track train IG
            'Validation Loss': val_losses       # Pure data loss
        }

        df = pd.DataFrame(losses_dict)

        # Save (rank 0 only, every 5 epochs)
        save_results = False
        if save_results and (ep % 5 == 0) and (rank == 0):
            torch.save({
                # Config
                'operator_type': operator_type,
                'enable_ig_loss': enable_ig_loss,
                'S': S,
                'T_in': T_in,
                'T_out': T_out,
                'jump_timestep': jump_timestep,
                # Architectures
                'width_CNO': width_CNO,
                'depth_CNO': depth_CNO,
                'kernel_size': kernel_size,
                'unet_depth': unet_depth,
                'mode1': mode1,
                'mode2': mode2,
                'mode3': mode3,
                'width_FNO': width_FNO,
                'wavelet': wavelet,
                'level': level,
                'layers': layers,
                'grid_range': grid_range,
                'width_WNO': width_WNO,
                'branch_layers': branch_layers,
                'trunk_layers': trunk_layers,
                
                # State
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_fnoloss,
            }, PATH_saved_models + f'saved_model_{Mode}.pth')
            
            df.to_excel(PATH_saved_loss + f'losses_data_{Mode}.xlsx', index=True, engine='openpyxl')
            

        if plot_live_loss and ep % 1 == 0:
            loss_live_plot(losses_dict)

        scheduler.step()
        outer_loop.set_description(f"Progress (Epoch {ep + 1}/{epochs}) Mode: {Mode}")
        outer_loop.set_postfix(fnoloss=f'{epoch_fnoloss:.2e}', ig_loss=f'{epoch_igloss:.2e}' , eval_loss=f'{val_losses[-1]:.2e}')

    cleanup()

# %%
def main(P_res, enable_ig_loss, PATH_dataset, train_size, eval_size, case, num_samples_x_y, batch_size, epochs, learning_rate, scheduler_step, scheduler_gamma, 
        operator_type, jump_timestep, T_in, T_out, nx, ny, 
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


    main_path = 'NSE_sequential/expriments/' + case + '/'
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    
    full_dataset = torch.load(PATH_dataset, weights_only=False, map_location=device)    
    total_size = len(full_dataset)
    test_size = total_size - (train_size + eval_size)

    # 3. Perform the random split, passing the configured generator
    train_dataset, eval_dataset, _ = random_split(
        full_dataset,
        [train_size, eval_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Pass the generator here
    )
    # P_res = full_dataset.tensors[0].shape[1]
    # optimizer and training configurations
    Mode = f"{case}_{'IG_Enable' if enable_ig_loss else 'IG_Disable'}_S_{nx}_Tin_{T_in}_Tout_{T_out}_Samp_{num_samples_x_y}_P_res_{P_res}_{operator_type}_DDP_{train_size}"
    print(Mode)

    
    # train_dataset, eval_dataset = None, None

    PATH_saved_models =      main_path + 'saved_models/'
    PATH_saved_loss =        main_path + 'loss_epoch/'

    ensure_directory(PATH_saved_models)
    ensure_directory(PATH_saved_loss)

    #  Model setup
    model_fn = FNO3d(T_in= T_in, T_out=T_out, modes_x=mode1, modes_y=mode2, modes_t=mode3, width=width_FNO)

    ss = 2 if enable_ig_loss else 1
    awl_fn = AutomaticWeightedLoss(ss)
    criterion=nn.MSELoss()

    torch.multiprocessing.spawn(train_model,
                                args=(world_size, model_fn, awl_fn, learning_rate, 
                                    operator_type, jump_timestep, T_in, T_out, S,

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
    P_res = 64
    
    
    # Dataset and case configuration
    PATH_dataset = f'/storage/group/cxs1024/default/mehdi/datasets/NSE_seq_multi_res_inputs/dataset/navier_stokes_dataset_{P_res}.pt'
    case = 'NSE_SA_multi_res'
    
    enable_ig_loss = True  # Enable/disable IG loss


    if enable_ig_loss:
        batch_size = 1
    else:
        batch_size = 8

    # Dataset sizes
    train_size = 700
    eval_size = 150


    # Sampling configuration
    num_samples_x_y = 'omega0'  # Number of random samples along x, y axes for Jacobian calculations

    # Training hyperparameters
    epochs = 500
    learning_rate = 0.005
    scheduler_step = 100
    scheduler_gamma = 0.95

    # Model configuration
    operator_type = 'FNO'  # Options: 'CNO', 'FNO', 'WNO', 'DeepONet'

    # Temporal and spatial dimensions
    jump_timestep = 2
    if jump_timestep == 2:
        total_steps = 30
    else:
        total_steps = 60
    
    T_in = 5
    T_out = total_steps - T_in
    
    nx = 64
    ny = 64

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
    main(P_res,
        enable_ig_loss, PATH_dataset, train_size, eval_size, case, 
        num_samples_x_y, batch_size, 
        epochs, learning_rate, scheduler_step, scheduler_gamma, 
        operator_type, jump_timestep, T_in, T_out, nx=nx, ny=ny, 
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

# %%