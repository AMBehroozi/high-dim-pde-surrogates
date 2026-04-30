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

from lib.util import MHPI, UnitGaussianNormalizer,run_nvidia_smi, split_dataset, prepare_data_loaders
from lib.utilities3 import load_and_normalize_datasets, ensure_directory, count_parameters, upscale_tensor, scale_tensors_to_largest_order
from lib.utiltools import loss_live_plot, AutomaticWeightedLoss
from lib.paralell_func import *
from models.FNO_3d_t_dist_DDP import FNO3d

import torch
from torch.utils.data import Dataset, Subset
import os

class TensorDataset(Dataset):
    """
    Custom PyTorch Dataset for a tensor with shape [nbatch, nx, ny, nt].
    """
    def __init__(self, tensor):
        """
        Args:
            tensor: PyTorch tensor with shape [nbatch, nx, ny, nt]
        """
        self.tensor = tensor
    
    def __len__(self):
        return self.tensor.shape[0]
    
    def __getitem__(self, idx):
        return self.tensor[idx]

def split_and_save_dataset(input_path, n_train, n_eval, seed=42):
    """
    Load a TensorDataset from a .pt file, split into train/eval/test based on sample counts,
    and save each as .pt files. Uses a seed for reproducibility.
    
    Args:
        input_path: str, path to the input .pt file
        n_train: int, number of samples for training set
        n_eval: int, number of samples for evaluation set
        seed: int, random seed for reproducibility
        output_dir: str, directory to save split datasets
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Load the dataset
    dataset = torch.load(input_path, weights_only=False)
    
    # Get total number of samples
    n_samples = len(dataset)
    
    # Validate input sizes
    if n_train + n_eval > n_samples:
        raise ValueError(f"Requested {n_train} train + {n_eval} eval samples exceed total {n_samples} samples")
    if n_train < 0 or n_eval < 0:
        raise ValueError("Number of samples must be non-negative")
    
    # Calculate test size (remaining samples)
    n_test = n_samples - n_train - n_eval
    if n_test < 0:
        raise ValueError("Negative test set size after allocating train and eval samples")
    
    # Create indices and shuffle
    indices = torch.randperm(n_samples).tolist()
    
    # Split indices
    train_indices = indices[:n_train]
    eval_indices = indices[n_train:n_train + n_eval]
    test_indices = indices[n_train + n_eval:]
    
    # Create Subset datasets
    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, eval_dataset, test_dataset    


# %%
def train_model(rank, world_size, model_fn, awl_fn, learning_rate, T_in, T_out, modes1, modes2, modes3, width,  
                epochs, epochs_fine_tune, PATH_saved_models, PATH_saved_loss, Mode, num_samples_x_y, 
                enable_ig_loss, L_x, L_y, S, criterion, plot_live_loss, 
                PATH_dataset, case, train_size, eval_size, batch_size, train_dataset, eval_dataset, save_results):
    
    setup(rank, world_size, environ='12357')
    jump = 1
    model = model_fn.to(rank)
    awl = awl_fn.to(rank)

    model = DDP(model, device_ids=[rank])
    awl = DDP(awl, device_ids=[rank])

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': learning_rate},
        {'params': awl.parameters(), 'lr': learning_rate}
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
    belending_input = True

    blend_factor = 0.7 
    noise_std = 0.005  

    # === Stage 1: One-Step Training ===
    for ep in outer_loop:
        model.train()
        total_fnoloss, total_samples = 0.0, 0

        for batch_data in train_loader:
            U_batch = batch_data[..., :150].to(rank)
            batch_parameter = U_batch[..., 0]
            jump_timestep = 2
            U_batch = U_batch[..., ::jump_timestep]
            T_pairs = U_batch.shape[-1] - 1
            batch_size = U_batch.shape[0]
            total_samples += batch_size * T_pairs

            x = torch.linspace(0, L_x, U_batch.shape[1]).unsqueeze(0).repeat(batch_size, 1).to(rank)
            y = torch.linspace(0, L_y, U_batch.shape[2]).unsqueeze(0).repeat(batch_size, 1).to(rank)


            optimizer.zero_grad()
            batch_fnoloss = 0.0
            prev_pred = None  # For blended input

            for t in range(T_pairs):
                u_in = U_batch[..., ::jump, ::jump, t].unsqueeze(-1)

                if belending_input and prev_pred is not None:
                    alpha = torch.rand(1).item() * blend_factor
                    noise = torch.randn_like(u_in) * noise_std
                    u_in = (1 - alpha) * u_in + alpha * prev_pred.detach()
                    u_in = u_in * (1 + noise)

                u_out = U_batch[..., ::jump, ::jump, t + 1].unsqueeze(-1) 

                pred = model(u_in, x, y, batch_parameter.unsqueeze(-1))
                prev_pred = pred  # Store for next blended input

                data_loss = criterion(pred, u_out)
                batch_fnoloss += data_loss.item()
                data_loss.backward()

            optimizer.step()
            total_fnoloss += batch_fnoloss * batch_size

        train_fnolosses.append(total_fnoloss / total_samples)

        # === Stage 1 Evaluation ===
        model.eval()
        total_fnoloss, total_samples = 0.0, 0

        for batch_data in eval_loader:
            U_batch = batch_data[..., :150].to(rank)
            batch_parameter = U_batch[..., 0]
            jump_timestep = 2
            U_batch = U_batch[..., ::jump_timestep]
            T_pairs = U_batch.shape[-1] - 1
            batch_size = U_batch.shape[0]
            total_samples += batch_size * T_pairs

            x = torch.linspace(0, L_x, U_batch.shape[1]).unsqueeze(0).repeat(batch_size, 1).to(rank)
            y = torch.linspace(0, L_y, U_batch.shape[2]).unsqueeze(0).repeat(batch_size, 1).to(rank)

            with torch.no_grad():
                for t in range(T_pairs):
                    u_in = U_batch[..., ::jump, ::jump, t].unsqueeze(-1)
                    u_out = U_batch[..., ::jump, ::jump, t + 1].unsqueeze(-1)    

                    pred = model(u_in, x, y, batch_parameter.unsqueeze(-1))
                    data_loss = criterion(pred, u_out)
                    total_fnoloss += data_loss.item() * batch_size

        val_losses.append(total_fnoloss / total_samples)


        if ep % 5 == 0 and rank == 0 and save_results:
            torch.save({
                'enable_ig_loss': enable_ig_loss,
                'S': S,
                'T_in': 1,
                'T_out': 1,
                'mode1': modes1,
                'mode2': modes2,
                'mode3': modes3,
                'width': width,
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_losses[-1],
            }, PATH_saved_models + f'saved_model_{Mode}.pth')
            
            pd.DataFrame({
                'Training FNO Loss': train_fnolosses,
                'Validation Loss': val_losses
            }).to_excel(PATH_saved_loss + f'losses_data_{Mode}.xlsx', index=True, engine='openpyxl')

        scheduler.step()
        outer_loop.set_description(f"Progress (Epoch {ep + 1}/{epochs}) Mode: {Mode}")
        outer_loop.set_postfix(fnoloss=f'{train_fnolosses[-1]:.2e}', eval_loss=f'{val_losses[-1]:.2e}')

    # === Stage 2: Autoregressive Fine-Tuning ===
    for pg in optimizer.param_groups:
        pg['lr'] = learning_rate * 0.01

    rollout_schedule = [2, 3, 4, 5, 6, 7]

    for nsteps in rollout_schedule:
        print(f"\n[Fine-Tune] Starting rollout length = {nsteps}")

        for ep in range(epochs_fine_tune):
            model.train()
            total_loss, total_samples = 0.0, 0

            for U_batch in train_loader:
                # === Move batch to device ===
                U_batch = U_batch.to(rank)
                batch_parameter = U_batch[..., 0]

                # === Temporal downsampling ===
                U_batch = U_batch[..., ::jump_timestep]  # shape: [B, H, W, T//jump_timestep]
                max_start_t = U_batch.shape[-1] - nsteps - 1
                batch_size = U_batch.shape[0]
                total_samples += batch_size * max_start_t

                # === Prepare coordinates ===
                x = torch.linspace(0, L_x, U_batch.shape[1]).unsqueeze(0).repeat(batch_size, 1).to(rank)
                y = torch.linspace(0, L_y, U_batch.shape[2]).unsqueeze(0).repeat(batch_size, 1).to(rank)

                # === Rolling window autoregressive update ===
                for t0 in range(max_start_t):
                    # Input state at time t0
                    u = U_batch[..., ::jump, ::jump, t0].unsqueeze(-1)

                    optimizer.zero_grad()
                    loss_window = 0.0

                    # Rollout for nsteps
                    for step in range(nsteps):
                        u_target = U_batch[..., ::jump, ::jump, t0 + step + 1].unsqueeze(-1)
                        u_pred = model(u, x, y, batch_parameter.unsqueeze(-1))
                        loss_window += criterion(u_pred, u_target)

                        u = u_pred.detach()  # prevent backprop through history

                    loss_window.backward()
                    optimizer.step()
                    total_loss += loss_window.item()

            train_fnolosses.append(total_loss / total_samples)
            print(f"[Epoch {ep+1}] Fine-tune Loss: {train_fnolosses[-1]:.2e}")

            # === Stage 2 Evaluation ===
            model.eval()
            total_fnoloss, total_samples = 0.0, 0

            for U_batch in eval_loader:
                U_batch = U_batch.to(rank)
                batch_parameter = U_batch[..., 0]
                U_batch = U_batch[..., ::jump_timestep]
                T_pairs = U_batch.shape[-1] - 1
                batch_size = U_batch.shape[0]
                total_samples += batch_size * T_pairs

                x = torch.linspace(0, L_x, U_batch.shape[1]).unsqueeze(0).repeat(batch_size, 1).to(rank)
                y = torch.linspace(0, L_y, U_batch.shape[2]).unsqueeze(0).repeat(batch_size, 1).to(rank)

                with torch.no_grad():
                    for t in range(T_pairs):
                        u_in = U_batch[..., ::jump, ::jump, t].unsqueeze(-1)
                        u_out = U_batch[..., ::jump, ::jump, t + 1].unsqueeze(-1)
                        pred = model(u_in, x, y, batch_parameter.unsqueeze(-1))
                        data_loss = criterion(pred, u_out)
                        total_fnoloss += data_loss.item() * batch_size

            val_losses.append(total_fnoloss / total_samples)

            if ep % 5 == 0 and rank == 0 and save_results:
                torch.save({
                    'enable_ig_loss': enable_ig_loss,
                    'S': S,
                    'T_in': 1,
                    'T_out': 1,
                    'mode1': modes1,
                    'mode2': modes2,
                    'mode3': modes3,
                    'width': width,
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_losses[-1],
                }, PATH_saved_models + f'saved_model_{Mode}.pth')
                pd.DataFrame({
                    'Training FNO Loss': train_fnolosses,
                    'Validation Loss': val_losses
                }).to_excel(PATH_saved_loss + f'losses_data_{Mode}.xlsx', index=True, engine='openpyxl')

    cleanup()

# %%
def main(S, enable_ig_loss, PATH_dataset, train_size, eval_size, case, num_samples_x_y, batch_size, epochs,  epochs_fine_tune, learning_rate, scheduler_step, scheduler_gamma, T_in, T_out, mode1, mode2, mode3, width, save_results):

    world_size = torch.cuda.device_count()
    print(f'Number of GPUs: {world_size}\n')
    
    plot_live_loss = False
    
    L_x = 1.0
    L_y = 1.0

    jump = (2 if S == 50 else 1)
    # optimizer and training configurations
    Mode = f"{case}_{'IG_Enable' if enable_ig_loss else 'IG_Disable'}_S_{S}_Tin_{T_in}_Tout_{T_out}_Samp_{num_samples_x_y}_FNO_DDP_{train_size}"
    print(Mode)

    main_path = 'NSE_rollout/experiments/' + case + '/rollout/'
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_dataset, eval_dataset, _ = split_and_save_dataset(PATH_dataset, n_train=train_size, n_eval=eval_size, seed=42)

    # train_dataset, eval_dataset = None, None

    PATH_saved_models =      main_path + 'saved_models/'
    PATH_saved_loss =        main_path + 'loss_epoch/'

    ensure_directory(PATH_saved_models)
    ensure_directory(PATH_saved_loss)

    #  Model setup
    model_fn = FNO3d(T_in=T_in, T_out=T_out, modes_x=modes1, modes_y=modes2, modes_t=modes3, width=width)
    ss = 2 if enable_ig_loss else 1
    awl_fn = AutomaticWeightedLoss(ss)
    criterion=nn.MSELoss()

    torch.multiprocessing.spawn(train_model,
                                args=(world_size, model_fn, awl_fn, learning_rate, T_in, T_out, modes1, modes2, modes3, width,
                                    epochs,  epochs_fine_tune, PATH_saved_models, PATH_saved_loss, Mode, num_samples_x_y,
                                    enable_ig_loss, L_x, L_y, S, criterion, plot_live_loss,
                                    PATH_dataset, case, train_size, eval_size, batch_size, train_dataset, eval_dataset, save_results),
                                nprocs=world_size,
                                join=True)

# %%
if __name__ == "__main__":
    run_nvidia_smi()
    MHPI()
    PATH_dataset = '/storage/group/cxs1024/default/mehdi/datasets/navier_stocks/full_dataset_NSE_RE1000_90_sec_sim_step_300.pt'
    enable_ig_loss = True
    
    case = 'NSE_RE_1000'

    train_size = 700
    eval_size = 150

    # train_size = 10
    # eval_size = 10

    num_samples_x_y = 'Single_step_fine_tune_blended_input2'  # Number of random samples along the second axes (x, y) for Jacobian calculations
    S = 64

    batch_size = 4
    # batch_size = 16

    save_results = False
    epochs = 500
    epochs_fine_tune = 50

    # epochs = 1
    # epochs_fine_tune = 1


    learning_rate = 0.005
    scheduler_step = 100
    scheduler_gamma = 0.95

    modes1 = 8
    modes2 = 8
    modes3 = 8
    
    width = 60

    total_steps = 51

    T_in= 1
    T_out= 1

    main(S, enable_ig_loss, PATH_dataset, train_size, eval_size, case, 
        num_samples_x_y, batch_size, 
        epochs,  epochs_fine_tune, learning_rate, scheduler_step, scheduler_gamma, 
        T_in, T_out, modes1, modes2, modes3, width, save_results)

