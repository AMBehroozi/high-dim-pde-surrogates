# %%
import argparse
import gc
import os
import re
import sys
import time
from torch.utils.data import (DataLoader, DistributedSampler, Subset,
                             TensorDataset, random_split)
from torch.nn.parallel import DistributedDataParallel as DDP

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Add paths to system
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

from lib.util import (MHPI, UnitGaussianNormalizer, run_nvidia_smi,
                     split_dataset, prepare_data_loaders)
from lib.utilities3 import (compute_jacobian_blocks, ensure_directory,
                           load_and_normalize_datasets, count_parameters,
                           upscale_tensor, scale_tensors_to_largest_order)
from lib.utiltools import loss_live_plot, AutomaticWeightedLoss
from lib.paralell_func import *
from models.FNO_2d_t_dist_DDP import FNO2d

# %%
def train_model(
    rank: int,
    world_size: int,
    model_fn,
    awl_fn,
    learning_rate: float,
    epochs_fine_tune: int,
    path_saved_models: str,
    path_saved_loss: str,
    pre_trained_model_path: str,
    enable_ig_loss: bool,
    L_x: float,
    L_y: float,
    S: int,
    criterion: nn.Module,
    plot_live_loss: bool,
    path_dataset: str,
    case: str,
    train_size: int,
    eval_size: int,
    batch_size: int,
    train_dataset,
    eval_dataset,
    save_results: bool
) -> None:
    """Train the FNO model with distributed data parallelism and fine-tuning."""
    # === Setup DDP and Model ===
    setup(rank, world_size)
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

    # === Dataloaders ===
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, sampler=eval_sampler)

    # === Load Pretrained Model ===
    path_save_fine_tune_model = re.sub(r"_FNO_DDP_(\d+)\.pth", r"_03_long_Fine_Tune_FNO_DDP_\1.pth", pre_trained_model_path)
    _, filename = os.path.split(path_save_fine_tune_model)
    path_save_fine_tune_loss = f"{path_saved_loss}{filename.replace('saved_model_', 'losses_data_').replace('.pth', '.xlsx')}"

    map_location = {'cuda:0': f'cuda:{rank}'}
    checkpoint = torch.load(pre_trained_model_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Lower learning rate for fine-tuning
    for pg in optimizer.param_groups:
        pg['lr'] = learning_rate

    # === Initialize Logs ===
    train_fnolosses, train_iglosses, val_losses = [], [], []
    torch.cuda.empty_cache()

    jump_timestep = 1
    # === Stage 2: Autoregressive Fine-Tuning with IG Loss ===
    rollout_schedule = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]

    for nsteps in rollout_schedule:
        print(f"\n[Fine-Tune] Starting rollout length = {nsteps}")
        outer_loop = tqdm(range(epochs_fine_tune), desc=f"Rollout {nsteps} Epoch")
        for ep in outer_loop:
            model.train()
            total_fnoloss, total_igloss, total_samples = 0.0, 0.0, 0

            for batch_data in train_loader:
                batch_data = [item.to(rank) for item in batch_data]
                epicentric, bed, depth, du_dp_true = batch_data
                depth = depth[..., :51:jump_timestep]
                batch_size = depth.shape[0]
                max_start_t = depth.shape[-1] - nsteps - 1
                total_samples += batch_size * max_start_t

                batch_parameter = 0.01 * bed[..., ::jump, ::jump]
                batch_parameter.requires_grad_(True)

                x = torch.linspace(0, L_x, S).unsqueeze(0).repeat(batch_size, 1).to(rank)
                y = torch.linspace(0, L_y, S).unsqueeze(0).repeat(batch_size, 1).to(rank)

                for t0 in range(max_start_t):
                    u = depth[..., ::jump, ::jump, t0] + bed[..., ::jump, ::jump]
                    u = u.unsqueeze(-1).clamp_max(10.0)

                    optimizer.zero_grad()
                    loss_window, ig_loss = 0.0, 0.0

                    for step in range(nsteps):
                        u_target = depth[..., ::jump, ::jump, t0 + step + 1] + bed[..., ::jump, ::jump]
                        u_target = u_target.unsqueeze(-1).clamp_max(10.0)
                        u_pred = model(u, x, y, batch_parameter.unsqueeze(-1)).clamp_max(10.0)
                        data_loss = criterion(u_pred, u_target)

                        # Compute IG-loss only on last step of rollout
                        enable_ig_loss = False
                        if enable_ig_loss and step == nsteps - 1:
                            num_samples_x, num_samples_y = 1, 1
                            downsampled_pred = u_pred[..., ::int(4 / jump), ::int(4 / jump), 0]
                            rand_x = torch.randperm(downsampled_pred.shape[1])[:num_samples_x]
                            rand_y = torch.randperm(downsampled_pred.shape[2])[:num_samples_y]

                            U_pred_select = downsampled_pred[:, rand_x, :][:, :, rand_y]
                            du_dp_pred = compute_jacobian_blocks(U_pred_select, batch_parameter, block_size=1)[..., ::4, ::4]
                            du_dp_true_selected = du_dp_true[:, rand_x, :, :, :][:, :, rand_y, :, :]

                            ig_loss = 500 * criterion(du_dp_pred, du_dp_true_selected)
                            data_loss = awl(data_loss, ig_loss)
                            total_igloss += ig_loss.item() * batch_size

                        loss_window += data_loss
                        u = u_pred.detach()

                    loss_window.backward()
                    optimizer.step()
                    total_fnoloss += loss_window.item() * batch_size

            train_fnolosses.append(total_fnoloss / total_samples)
            train_iglosses.append(total_igloss / total_samples)


            # === Stage 2 Evaluation ===
            model.eval()
            total_fnoloss, total_samples = 0.0, 0

            for batch_data in eval_loader:
                batch_data = [item.to(rank) for item in batch_data]
                epicentric, bed, depth, _ = batch_data
                depth = depth[..., :51:jump_timestep]
                T_pairs = depth.shape[-1] - 1
                batch_size = depth.shape[0]
                total_samples += batch_size * T_pairs

                batch_parameter = 0.01 * bed[..., ::jump, ::jump]
                x = torch.linspace(0, L_x, S).unsqueeze(0).repeat(batch_size, 1).to(rank)
                y = torch.linspace(0, L_y, S).unsqueeze(0).repeat(batch_size, 1).to(rank)

                with torch.no_grad():
                    for t in range(T_pairs):
                        u_in = depth[..., ::jump, ::jump, t] + bed[..., ::jump, ::jump]
                        u_out = depth[..., ::jump, ::jump, t + 1] + bed[..., ::jump, ::jump]
                        u_in = u_in.unsqueeze(-1).clamp_max(10.0)
                        u_out = u_out.unsqueeze(-1).clamp_max(10.0)
                        pred = model(u_in, x, y, batch_parameter.unsqueeze(-1)).clamp_max(10.0)
                        data_loss = criterion(pred, u_out)
                        total_fnoloss += data_loss.item() * batch_size

            val_losses.append(total_fnoloss / total_samples)

            if ep % 5 == 0 and rank == 0 and save_results:
                torch.save({
                    'enable_ig_loss': enable_ig_loss,
                    'rollout_schedule': rollout_schedule,
                    'S': S,
                    'T_in': 1,
                    'T_out': 1,
                    'mode1': checkpoint['mode1'],
                    'mode2': checkpoint['mode2'],
                    'width': checkpoint['width'],
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_losses[-1],
                }, path_save_fine_tune_model)
                pd.DataFrame({
                    'Training FNO Loss': train_fnolosses,
                    'Train IG loss': train_iglosses,
                    'Validation Loss': val_losses
                }).to_excel(path_save_fine_tune_loss, index=True, engine='openpyxl')

            # print(f"[Epoch {ep+1}] Fine-tune Loss: {train_fnolosses[-1]:.2e} | IG Loss: {train_iglosses[-1]:.2e}")
            outer_loop.set_description(f"Progress (Epoch {ep + 1}/{epochs_fine_tune})")
            outer_loop.set_postfix(train_fnolosses=f'{train_fnolosses[-1]:.2e}', eval_loss=f'{val_losses[-1]:.2e}')

    cleanup()


def main(
    S: int,
    path_dataset: str,
    train_size: int,
    eval_size: int,
    case: str,
    main_path: str,
    pre_trained_model_path: str,
    batch_size: int,
    epochs_fine_tune: int,
    learning_rate: float,
    scheduler_step: int,
    scheduler_gamma: float,
    save_results: bool
) -> None:
    """Main function for training the FNO model with distributed data parallelism."""
    # Hardware setup
    world_size = torch.cuda.device_count()
    print(f'Number of GPUs: {world_size}\n')

    # Constants
    L_x, L_y = 1.0, 1.0
    plot_live_loss = False

    # Paths
    path_saved_models = f"{main_path}saved_models/"
    path_saved_loss = f"{main_path}loss_epoch/"
    checkpoint_path = path_saved_models + pre_trained_model_path

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    jump = 2 if S == 50 else 1
    enable_ig_loss = checkpoint['enable_ig_loss']
    T_in = checkpoint['T_in']
    T_out = checkpoint['T_out']

    # Training configuration
    mode = (
        f"{case}_{'IG_Enable' if enable_ig_loss else 'IG_Disable'}_"
        f"S_{S}_Tin_{T_in}_Tout_{T_out}_Samp_Fine_Tune_FNO_DDP_{train_size}"
    )
    print(f"Mode: {mode}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preparation
    train_dataset, eval_dataset = prepare_data_loaders(
        path_dataset, case, train_size, eval_size, batch_size
    )

    # Model setup
    model_fn = FNO2d(
        modes1=checkpoint['mode1'],
        modes2=checkpoint['mode2'],
        width=checkpoint['width'],
        state_size=1,
        parameter_size=1
    )
    awl_fn = AutomaticWeightedLoss(2 if enable_ig_loss else 1)
    criterion = nn.MSELoss()

    # Start distributed training
    mp.spawn(
        train_model,
        args=(
            world_size, model_fn, awl_fn, learning_rate, epochs_fine_tune,
            path_saved_models, path_saved_loss, checkpoint_path,
            enable_ig_loss, L_x, L_y, S, criterion, plot_live_loss,
            path_dataset, case, train_size, eval_size, batch_size,
            train_dataset, eval_dataset, save_results
        ),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    run_nvidia_smi()
    MHPI()

    # Configuration
    PATH_DATASET = (
        "/storage/work/amb10399/project/SC-FNO-DIST/Tsunami/datasets/"
        "1000__3e5_8e5__2e5_8e5_with_jac_multi_fault/"
    )
    CASE = "Tsunami_multi_epicentric_multi_fault"
    MAIN_PATH = (
        "/storage/work/amb10399/project/SC-FNO-DIST/Tsunami/experiments/"
        "Tsunami_multi_epicentric_multi_fault/rollout/"
    )
    PRE_TRAINED_MODEL_PATH = (
        "saved_model_Tsunami_multi_epicentric_multi_fault_IG_Enable_S_100_"
        "Tin_1_Tout_1_Samp_All_rollout_2d_shalower_single_step_7_blended_noise_ig_enable_long_FNO_DDP_700.pth"
    )

    # Extract numbers using regex
    S = int(re.search(r'_S_(\d+)_', PRE_TRAINED_MODEL_PATH).group(1))
    TRAIN_SIZE = int(re.search(r'_(\d+)\.pth$', PRE_TRAINED_MODEL_PATH).group(1))
    EVAL_SIZE = 150

    # TRAIN_SIZE = 10
    # EVAL_SIZE = 10


    BATCH_SIZE = 8

    EPOCHS_FINE_TUNE = 10

    LEARNING_RATE = 1e-5
    SCHEDULER_STEP = 100
    SCHEDULER_GAMMA = 0.95
    SAVE_RESULTS = True

    main(
        S, PATH_DATASET, TRAIN_SIZE, EVAL_SIZE, CASE, MAIN_PATH,
        PRE_TRAINED_MODEL_PATH, BATCH_SIZE, EPOCHS_FINE_TUNE,
        LEARNING_RATE, SCHEDULER_STEP, SCHEDULER_GAMMA, SAVE_RESULTS
    )
