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
from models.FNO_3d_t_dist_DDP import FNO3d as FNO3dDDP
# %%

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_model(rank, world_size, model_fn, awl_fn, learning_rate, T_in, T_out, modes1, modes2, modes3, width,  
                epochs, PATH_saved_models, PATH_saved_loss, Mode, num_samples_x_y, 
                enable_ig_loss, L_x, L_y, S, criterion, plot_live_loss, 
                PATH_dataset, case, train_size, eval_size, batch_size, train_dataset, eval_dataset):
    
    setup(rank, world_size)
    jump = 2
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
            epicentric, bed, depth, du_dp_true = batch_data

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
            
            # x = torch.linspace(0, L_x, S).unsqueeze(0).repeat(batch_size, 1).to(rank)
            # y = torch.linspace(0, L_y, S).unsqueeze(0).repeat(batch_size, 1).to(rank)            

            optimizer.zero_grad()
            
            U_pred = model(batch_u_in, batch_parameter)
            data_loss = criterion(U_pred, batch_u_out)

            num_samples_x = 4 
            num_samples_y = 3    
            
            random_indices_x = torch.randperm(U_pred[..., ::int(4/jump), ::int(4/jump), -1].shape[1])[:num_samples_x]
            random_indices_y = torch.randperm( U_pred[..., ::int(4/jump), ::int(4/jump), -1].shape[2])[:num_samples_y]

            U_pred_select = U_pred[..., ::int(4/jump), ::int(4/jump), -1][:, random_indices_x, :][:, :, random_indices_y]  # Select 6 and 8 from 2nd dimension (nx//4)

            du_dp_pred = compute_jacobian_blocks(U_pred_select, batch_parameter, block_size=1)[..., ::2, ::2]
            du_dp_true_selected = du_dp_true[:, random_indices_x, :, :, :][:, :, random_indices_y, :, :]
            ig_loss = 500 * criterion(du_dp_pred, du_dp_true_selected)
            
            if enable_ig_loss:
                loss = awl(data_loss, ig_loss)
                # loss = data_loss + ig_loss 
                fnoloss = data_loss
                ig_loss = ig_loss
            else:
                loss = data_loss
                fnoloss = data_loss
                ig_loss = ig_loss
            
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
            x = torch.linspace(0, L_x, S).unsqueeze(0).repeat(batch_size, 1).to(rank)
            y = torch.linspace(0, L_y, S).unsqueeze(0).repeat(batch_size, 1).to(rank)            
            
            with torch.no_grad():
                U_pred = model(batch_u_in, 0.01 * bed[..., ::jump, ::jump])
                data_loss = criterion(U_pred, batch_u_out)
            
            total_fnoloss += data_loss.item() * batch_size
        
        epoch_fnoloss = total_fnoloss / total_samples
        val_losses.append(epoch_fnoloss)

        losses_dict = {'Training FNO Loss': train_fnolosses, 'Train IG loss': train_iglosses, 'Validation Loss': val_losses}       

        df = pd.DataFrame(losses_dict)

        if ep % 5 == 0 and rank == 0:
            torch.save({
                'enable_ig_loss': enable_ig_loss,
                'S':S,
                'T_in': T_in,
                'T_out':T_out,
                'mode1':modes1,
                'mode2':modes2,
                'mode3':modes3,
                'width':width,
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

    main_path = 'Tsunami/experiments/' + case + '/'
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
    model_fn = FNO3dDDP(T_in=T_in, T_out=T_out, modes_x=mode1, modes_y=mode2, modes_t=mode3, width=width)
    
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
        
    num_samples_x_y = 'All_test_test'  # Number of random samples along the second axes (x, y) for Jacobian calculations
    S = 50
    batch_size = 8
    epochs = 1000
    learning_rate = 0.005
    scheduler_step = 100
    scheduler_gamma = 0.95

    modes1 = 8
    modes2 = 8
    modes3 = 8
    width = 20

    total_steps = 51
    T_in= 5
    T_out= total_steps - T_in

    main(S, enable_ig_loss, PATH_dataset, train_size, eval_size, case, 
        num_samples_x_y, batch_size, 
        epochs, learning_rate, scheduler_step, scheduler_gamma, 
        T_in, T_out, modes1, modes2, modes3, width)

# %%
