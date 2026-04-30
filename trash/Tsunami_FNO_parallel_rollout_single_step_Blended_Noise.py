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
from lib.util import MHPI, UnitGaussianNormalizer,run_nvidia_smi, split_dataset, prepare_data_loaders
from lib.utilities3 import load_and_normalize_datasets, ensure_directory, count_parameters, compute_jacobian_blocks, upscale_tensor, scale_tensors_to_largest_order
from lib.utiltools import loss_live_plot, AutomaticWeightedLoss
from lib.batchJacobian import batchJacobian_PDE
from lib.paralell_func import *
from models.FNO_2d_t_dist_DDP import FNO2d
# %%

def train_model(rank, world_size, model_fn, awl_fn, learning_rate, T_in, T_out, modes1, modes2, modes3, width,  
                epochs, PATH_saved_models, PATH_saved_loss, Mode, num_samples_x_y, 
                enable_ig_loss, L_x, L_y, S, criterion, plot_live_loss, 
                PATH_dataset, case, train_size, eval_size, batch_size, train_dataset, eval_dataset):
    
    setup(rank, world_size)
    jump = 1
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

    jump_timestep = 2
    stage = True    
    add_input_noise = True

    for ep in outer_loop:
        model.train()

        total_fnoloss = 0.0
        total_igloss = 0.0
        total_samples = 0

        for batch_data in train_loader:
            batch_data = [item.to(rank) for item in batch_data]
            epicentric, bed, depth, du_dp_true = batch_data

            depth = depth[..., :51:jump_timestep]
            total_time_steps = depth.shape[-1]
            T_pairs = total_time_steps - 1
            batch_size = depth.shape[0]
            total_samples += batch_size * T_pairs

            batch_parameter = 0.01 * bed[..., ::jump, ::jump]
            batch_parameter.requires_grad_(True)

            x = torch.linspace(0, L_x, S).unsqueeze(0).repeat(batch_size, 1).to(rank)
            y = torch.linspace(0, L_y, S).unsqueeze(0).repeat(batch_size, 1).to(rank)

            batch_fnoloss = 0.0
            batch_igloss = 0.0

            optimizer.zero_grad()

            prev_pred = None  # 🔁 store previous prediction

            for t in range(T_pairs):
                u_true = depth[..., ::jump, ::jump, t] + bed[..., ::jump, ::jump]
                u_in = u_true.unsqueeze(-1)

                if add_input_noise and prev_pred is not None:

                    blend_factor = 0.7 
                    noise_std = 0.005  
                    
                    alpha = torch.rand(1).item() * blend_factor  
                    u_in = (1 - alpha) * u_in + alpha * prev_pred.detach()

                    noise = torch.randn_like(u_in) * noise_std
                    u_in = u_in * (1 + noise)

                u_in[u_in > 5] = 10.0

                u_out = depth[..., ::jump, ::jump, t + 1] + bed[..., ::jump, ::jump]
                u_out = u_out.unsqueeze(-1)
                u_out[u_out > 5] = 10.0

                pred = model(u_in, x, y, batch_parameter.unsqueeze(-1))  # [B, C, H, W, 1]
                pred[pred > 5] = 10.0

                # 🔁 Store prediction for blending in next iteration
                prev_pred = pred

                data_loss = criterion(pred, u_out)
                batch_fnoloss += data_loss.item()

                if enable_ig_loss and t == T_pairs - 1:
                    num_samples_x = 1
                    num_samples_y = 1

                    downsampled_pred = pred[..., ::int(4 / jump), ::int(4 / jump), 0]
                    rand_x = torch.randperm(downsampled_pred.shape[1])[:num_samples_x]
                    rand_y = torch.randperm(downsampled_pred.shape[2])[:num_samples_y]

                    U_pred_select = downsampled_pred[:, rand_x, :][:, :, rand_y]
                    du_dp_pred = compute_jacobian_blocks(U_pred_select, batch_parameter, block_size=1)[..., ::4, ::4]
                    du_dp_true_selected = du_dp_true[:, rand_x, :, :, :][:, :, rand_y, :, :]

                    ig_loss = 500 * criterion(du_dp_pred, du_dp_true_selected)
                    data_loss = awl(data_loss, ig_loss)
                    batch_igloss += ig_loss.item()

                data_loss.backward()

            optimizer.step()

            total_fnoloss += batch_fnoloss * batch_size
            total_igloss += batch_igloss * batch_size

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
            epicentric, bed, depth, _ = batch_data  # IG not used in eval

            depth = depth[..., :51:jump_timestep]
            total_time_steps = depth.shape[-1]

            T_pairs = total_time_steps - 1  # total (u_t → u_{t+1}) pairs
            batch_size = depth.shape[0]
            total_samples += batch_size * T_pairs

            batch_parameter = 0.01 * bed[..., ::jump, ::jump]

            x = torch.linspace(0, L_x, S).unsqueeze(0).repeat(batch_size, 1).to(rank)
            y = torch.linspace(0, L_y, S).unsqueeze(0).repeat(batch_size, 1).to(rank)

            batch_fnoloss = 0.0

            with torch.no_grad():
                for t in range(T_pairs):
                    # Model input: d_t + bed
                    u_in = depth[..., ::jump, ::jump, t] + bed[..., ::jump, ::jump]
                    u_in = u_in.unsqueeze(-1)
                    u_in[u_in > 5] = 10.0

                    # Target: d_{t+1} + bed
                    u_out = depth[..., ::jump, ::jump, t + 1] + bed[..., ::jump, ::jump]
                    u_out = u_out.unsqueeze(-1)
                    u_out[u_out > 5] = 10.0

                    # Predict one step
                    pred = model(u_in, x, y, batch_parameter.unsqueeze(-1))
                    pred[pred > 5] = 10.0

                    # Loss
                    data_loss = criterion(pred, u_out)
                    batch_fnoloss += data_loss.item()

            total_fnoloss += batch_fnoloss * batch_size

        # Average loss
        epoch_fnoloss = total_fnoloss / total_samples
        val_losses.append(epoch_fnoloss)

        # Logging and saving
        losses_dict = {
            'Training FNO Loss': train_fnolosses,
            'Train IG loss': train_iglosses, 
            'Validation Loss': val_losses
        }
        df = pd.DataFrame(losses_dict)

        save_results = True
        if save_results:
            if ep % 5 == 0 and rank == 0:
                torch.save({
                    'enable_ig_loss': enable_ig_loss,
                    'add_input_noise': add_input_noise,
                    'S': S,
                    'T_in': 1,
                    'T_out': 1,
                    'blend_factor': blend_factor,
                    'noise_std': noise_std,
                    'mode1': modes1,
                    'mode2': modes2,
                    'mode3': modes3,
                    'width': width,
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
        outer_loop.set_postfix(fnoloss=f'{epoch_fnoloss:.2e}', ig_loss=f'{epoch_igloss:.2e}', eval_loss=f'{val_losses[-1]:.2e}')
    cleanup()

# %%
def main(S, enable_ig_loss, PATH_dataset, train_size, eval_size, case, num_samples_x_y, batch_size, epochs, learning_rate, scheduler_step, scheduler_gamma, T_in, T_out, mode1, mode2, mode3, width):

    world_size = torch.cuda.device_count()
    print(f'Number of GPUs: {world_size}\n')
    
    plot_live_loss = False
    
    L_x = 1.0
    L_y = 1.0

    jump = (2 if S == 50 else 1)
    # optimizer and training configurations
    Mode = f"{case}_{'IG_Enable' if enable_ig_loss else 'IG_Disable'}_S_{S}_Tin_{T_in}_Tout_{T_out}_Samp_{num_samples_x_y}_FNO_DDP_{train_size}"
    print(Mode)

    main_path = 'Tsunami/experiments/' + case + '/rollout/'
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_dataset, eval_dataset = prepare_data_loaders(PATH_dataset, case, train_size, eval_size, batch_size)
    # train_dataset, eval_dataset = None, None

    PATH_saved_models =      main_path + 'saved_models/'
    PATH_saved_loss =        main_path + 'loss_epoch/'

    ensure_directory(PATH_saved_models)
    ensure_directory(PATH_saved_loss)

    #  Model setup
    model_fn = FNO2d(modes1=mode1, modes2=mode2, width=width, state_size=1, parameter_size=1)

    ss = 2 if enable_ig_loss else 1
    awl_fn = AutomaticWeightedLoss(ss)
    criterion=nn.MSELoss()

    torch.multiprocessing.spawn(train_model,
                                args=(world_size, model_fn, awl_fn, learning_rate, T_in, T_out, modes1, modes2, modes3, width,
                                    epochs, PATH_saved_models, PATH_saved_loss, Mode, num_samples_x_y,
                                    enable_ig_loss, L_x, L_y, S, criterion, plot_live_loss,
                                    PATH_dataset, case, train_size, eval_size, batch_size, train_dataset, eval_dataset),
                                nprocs=world_size,
                                join=True)

# %%
if __name__ == "__main__":
    run_nvidia_smi()
    MHPI()
    PATH_dataset = '/storage/work/amb10399/project/SC-FNO-DIST/Tsunami/datasets/1000__3e5_8e5__2e5_8e5_with_jac_multi_fault/'
    enable_ig_loss = True
    case = 'Tsunami_multi_epicentric_multi_fault'

    train_size = 700
    eval_size = 150

    # train_size = 10
    # eval_size = 10

    num_samples_x_y = 'All_rollout_2d_shalower_single_step_7_blended_noise_ig_enable_long'  # Number of random samples along the second axes (x, y) for Jacobian calculations
    S = 100

    batch_size = 8
    # batch_size = 16
    
    epochs = 1000
    learning_rate = 0.005
    scheduler_step = 100
    scheduler_gamma = 0.95

    modes1 = 8
    modes2 = 8
    modes3 = 1
    
    width = 60

    total_steps = 51

    T_in= 1
    T_out= 1

    main(S, enable_ig_loss, PATH_dataset, train_size, eval_size, case, 
        num_samples_x_y, batch_size, 
        epochs, learning_rate, scheduler_step, scheduler_gamma, 
        T_in, T_out, modes1, modes2, modes3, width)

