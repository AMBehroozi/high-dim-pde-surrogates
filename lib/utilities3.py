import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
import operator
import sourcedefender
from torch.utils.data import Subset
from functools import reduce
from functools import partial
import torch.nn.functional as F
import sys

sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

from lib.batchJacobian import batchJacobian_PDE
import pickle
import random
from torch.utils.data import TensorDataset, random_split, DataLoader
import os
from tqdm import tqdm
from collections import OrderedDict  # Ensure this is included

import pandas as pd
import matplotlib.pyplot as plt

def plot_log_loss(file_path, mode, plot_ig_loss=False):
    """
    Loads an Excel file containing loss values, applies a log transformation, 
    and plots the losses using Plotly. Optionally includes "Train IG loss" in the plot.

    Parameters:
    file_path (str): Path to the Excel file.
    mode (str): Mode name to be displayed in the title.
    plot_ig_loss (bool): If True, includes "Train IG loss" in the plot; otherwise, excludes it.
    """
    import pandas as pd
    import numpy as np
    import plotly.express as px

    # Load the Excel file into a pandas DataFrame
    df = pd.read_excel(file_path)

    # Drop the unnamed index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    # Drop the unnamed index column if present
    if 'Epoch' in df.columns:
        df = df.drop(columns=['Epoch'])

    # Apply log transformation to loss values (adding a small value to avoid log(0))
    df_log = df.copy()
    df_log[df.columns] = np.log10(df[df.columns] + 1e-8)  # Add small epsilon to avoid log(0)

    # Filter columns if plot_ig_loss is False
    if not plot_ig_loss and "Train IG loss" in df_log.columns:
        df_log = df_log.drop(columns=["Train IG loss"])

    # Create an interactive plot using Plotly
    title_text = f"Log-Scaled Training and Validation Losses Over Epochs<br>{mode}"  # Use <br> for line break
    fig = px.line(df_log, x=df_log.index, y=df_log.columns, markers=True,
                  labels={"index": "Epoch", "value": "Log Loss", "variable": "Loss Type"},
                  title=title_text)

    # Show the plot
    fig.show()


def calculate_relative_errors(predicted, true):
    """
    Calculate the L1 and L2 relative errors between true and predicted tensors.

    Parameters:
    - true (torch.Tensor): The tensor containing the true values.
    - predicted (torch.Tensor): The tensor containing the predicted values.

    Returns:
    - tuple: (L1 relative error, L2 relative error)
    """
    # Ensure true and predicted are float tensors for accurate division
    true = true.float()
    predicted = predicted.float()

    # Compute the absolute differences and the squared differences
    abs_diff = torch.abs(true - predicted)
    sq_diff = (true - predicted) ** 2

    # Compute the L1 and L2 norms for the numerator
    l1_norm = torch.sum(abs_diff)
    l2_norm = torch.sqrt(torch.sum(sq_diff))

    # Compute the L1 and L2 norms for the denominator
    l1_denom = torch.sum(torch.abs(true))
    l2_denom = torch.sqrt(torch.sum(true ** 2))

    # Calculate the relative errors
    l1_relative_error = l1_norm / l1_denom
    l2_relative_error = l2_norm / l2_denom

    return l1_relative_error.item(), l2_relative_error.item()


def scale_tensors_to_largest_order(tensors):
    def order_of_magnitude(tensor):
        # Calculate order of magnitude for non-zero elements
        orders = torch.floor(torch.log10(torch.abs(tensor)))
        orders[tensor == 0] = float('-inf')  # Handle zeros gracefully
        return orders

    # Find the largest order of magnitude among all tensors
    max_order = max(order_of_magnitude(tensor).item() for tensor in tensors)

    scaled_tensors = []
    weights = []

    for tensor in tensors:
        # Calculate current order of magnitude for the tensor
        current_order = order_of_magnitude(tensor).item()

        # Calculate the weight to scale up to the largest order
        weight = 10 ** (max_order - current_order)

        # Scale the tensor by the calculated weight (only if needed)
        scaled_tensor = tensor * weight
        weights.append(weight)
        scaled_tensors.append(scaled_tensor)

    return weights


class NonLocalMeansSmoothing(nn.Module):
    def __init__(self, state_size, kernel_size=5, sigma=1.0, h=1.0, device='cpu'):
        """
        Initialize the Non-Local Means Smoothing Layer.
        
        Args:
        state_size (int): Number of channels in the input tensor.
        kernel_size (int): Size of the local neighborhood for patch comparison.
        sigma (float): Standard deviation for the Gaussian kernel.
        h (float): Filtering parameter controlling the decay of the exponential function.
        device (str): Device to use for computations ('cpu' or 'cuda').
        """
        super(NonLocalMeansSmoothing, self).__init__()
        self.state_size = state_size
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.h = h
        self.device = device
        
        self.gaussian_kernel = self._create_gaussian_kernel().to(self.device)
        
    def _create_gaussian_kernel(self):
        x = torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1, device=self.device)
        y = torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))
        return kernel / kernel.sum()
    
    def forward(self, x):
        """
        Apply Non-Local Means smoothing to the input tensor.
        
        Args:
        x (torch.Tensor): Input tensor of shape [nbatch, nx, ny, state_size].
        
        Returns:
        torch.Tensor: Smoothed output tensor of shape [nbatch, nx, ny, state_size].
        """
        x = x.to(self.device)
        nbatch, nx, ny, _ = x.shape
        
        pad = self.kernel_size // 2
        x_padded = F.pad(x.permute(0, 3, 1, 2), (pad, pad, pad, pad), mode='reflect')
        
        # patches: local neighborhoods for each pixel, shape [nbatch, state_size * kernel_size**2, nx * ny]
        patches = F.unfold(x_padded, kernel_size=self.kernel_size, stride=1)
        patches = patches.view(nbatch, self.state_size, self.kernel_size**2, nx * ny)
        
        # x_reshaped: reshaped input for broadcasting, shape [nbatch, state_size, 1, nx * ny]
        x_reshaped = x.permute(0, 3, 1, 2).contiguous().view(nbatch, self.state_size, 1, nx * ny)
        
        # diff: squared difference between central pixel and its neighborhood
        diff = (x_reshaped - patches).pow(2).sum(dim=1)
        
        # weights: similarity weights for each pixel in the neighborhood
        weights = torch.exp(-diff / (self.h**2))
        weights = weights * self.gaussian_kernel.view(-1).unsqueeze(1)
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        # smoothed: weighted sum of neighborhood pixels
        smoothed = (patches * weights.unsqueeze(1)).sum(dim=2)
        smoothed = smoothed.view(nbatch, self.state_size, nx, ny).permute(0, 2, 3, 1)
        
        return smoothed

# class NonLocalMeansSmoothing(nn.Module):
#     def __init__(self, state_size, kernel_size=5, sigma=1.0, h=1.0, device='cpu'):
#         """
#         Initialize the Non-Local Means Smoothing Layer.
        
#         Args:
#         state_size (int): Number of channels in the input tensor.
#         kernel_size (int): Size of the local neighborhood for patch comparison.
#         sigma (float): Standard deviation for the Gaussian kernel.
#         h (float): Filtering parameter controlling the decay of the exponential function.
#         device (str): Device to use for computations ('cpu' or 'cuda').
#         """
#         super(NonLocalMeansSmoothing, self).__init__()
#         self.state_size = state_size
#         self.kernel_size = kernel_size
#         self.sigma = sigma
#         self.h = h
#         self.device = device
        
#         self.gaussian_kernel = self._create_gaussian_kernel().to(self.device)
        
#     def _create_gaussian_kernel(self):
#         x = torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1, device=self.device)
#         y = torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1, device=self.device)
#         xx, yy = torch.meshgrid(x, y, indexing='ij')
#         kernel = torch.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))
#         return kernel / kernel.sum()
    
#     def forward(self, x):
#         """
#         Apply Non-Local Means smoothing to the input tensor.
        
#         Args:
#         x (torch.Tensor): Input tensor of shape [nbatch, nx, ny, state_size].
        
#         Returns:
#         torch.Tensor: Smoothed output tensor of shape [nbatch, nx, ny, state_size].
#         """
#         x = x.to(self.device)
#         nbatch, nx, ny, _ = x.shape
        
#         pad = self.kernel_size // 2
#         x_padded = F.pad(x.permute(0, 3, 1, 2), (pad, pad, pad, pad), mode='reflect')
        
#         # patches: local neighborhoods for each pixel, shape [nbatch, state_size * kernel_size**2, nx * ny]
#         patches = F.unfold(x_padded, kernel_size=self.kernel_size, stride=1)
#         patches = patches.view(nbatch, self.state_size, self.kernel_size**2, nx * ny)
        
#         # x_reshaped: reshaped input for broadcasting, shape [nbatch, state_size, 1, nx * ny]
#         x_reshaped = x.permute(0, 3, 1, 2).contiguous().view(nbatch, self.state_size, 1, nx * ny)
        
#         # diff: squared difference between central pixel and its neighborhood
#         diff = (x_reshaped - patches).pow(2).sum(dim=1)
        
#         # weights: similarity weights for each pixel in the neighborhood
#         weights = torch.exp(-diff / (self.h**2))
#         weights = weights * self.gaussian_kernel.view(-1).unsqueeze(1)
#         weights = weights / weights.sum(dim=1, keepdim=True)
        
#         # smoothed: weighted sum of neighborhood pixels
#         smoothed = (patches * weights.unsqueeze(1)).sum(dim=2)
#         smoothed = smoothed.view(nbatch, self.state_size, nx, ny).permute(0, 2, 3, 1)
        
#         return smoothed

def plot_losses_from_excel(excel_path, lable='Training HNO Loss', Mode='', save_fig=False):
    """
    Plots the training and validation losses from an Excel file, using a logarithmic scale for the y-axis.

    Args:
    excel_path (str): Path to the Excel file containing the loss data.
    """
    # Read the data from Excel
    df = pd.read_excel(excel_path, engine='openpyxl', index_col=0)
    
    # Plotting the data
    plt.figure(figsize=(10, 5))
    plt.plot(df[f'{lable}'], label=f'{lable}')
    plt.plot(df['Train IG loss'], label='Training IG Loss')
    plt.plot(df['Validation Loss'], label='Validation Loss')
    
    # Adding titles and labels
    plt.title(f'Losses Over Epochs  {Mode}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Set y-axis to logarithmic scale
    plt.yscale('log')

    # Show the plot
    plt.grid(True)
    if save_fig:
        plt.savefig(Mode + '.png', dpi=300, bbox_inches='tight')
    plt.show()


def adjust_state_dict(state_dict, model):
    """
    Adjusts the 'module.' prefix in the state_dict keys depending on the model configuration.
    """
    new_state_dict = OrderedDict()
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)  # Checks if model is DDP
    for key, value in state_dict.items():
        if is_ddp and not key.startswith('module.'):
            new_key = 'module.' + key
        elif not is_ddp and key.startswith('module.'):
            new_key = key[7:]  # Strip 'module.' prefix if model is not DDP
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def check_if_from_ddp(checkpoint):

    # Get the state dictionary
    state_dict = checkpoint['model_state_dict']

    # Check if any key starts with 'module.'
    is_ddp = any(key.startswith('module.') for key in state_dict.keys())
    
    return is_ddp

from torch.func import vmap
from torch import vmap

# =============================================================================
# The Function to be Tested
# =============================================================================
def batchJacobian_vmap(y, x, graphed=False):
    """
    Computes the Jacobian of y w.r.t. x for a batch of 2D fields.
    """
    if y.ndim != 3 or x.ndim != 3:
        raise ValueError("Both y and x must be 3D tensors of shape [batch, height, width]")

    nb, nx1, ny1 = y.shape
    _, nx2, ny2 = x.shape
    
    def get_one_jacobian_column(v):
        v_batched = v.expand(nb, -1, -1)
        column, = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=v_batched,
            retain_graph=True,
            create_graph=graphed,
            allow_unused=True
        )
        return column

    I = torch.eye(nx1 * ny1, device=x.device, dtype=x.dtype).view(nx1 * ny1, nx1, ny1)
    jacobian_columns = vmap(get_one_jacobian_column, in_dims=0)(I)
    jacobian = jacobian_columns.permute(1, 0, 2, 3)
    DYDX = jacobian.view(nb, nx1, ny1, nx2, ny2)

    if not graphed:
        DYDX = DYDX.detach()

    return DYDX



def compute_jacobian_blocks(u, k, block_size, graphed=True):
    """
    Compute the Jacobian of tensor u with respect to tensor k in blocks using a provided Jacobian function,
    where each block of u is considered with respect to the entire tensor k.

    Parameters:
    - u: Tensor with shape [nbatch, nx, ny], requires_grad=False.
    - k: Tensor with shape [nbatch, nx11, ny1], requires_grad=True.
    - block_size: Size of the block over which to compute the Jacobian of u.

    Returns:
    - jacobian: Tensor with shape [nbatch, nx, ny, nx1, ny1].
    """
    nbatch, nx, ny = u.shape
    nbatch1, nx1, ny1 = k.shape
    # Initialize the Jacobian matrix
    jacobian = torch.zeros(nbatch, nx, ny, nx1, ny1, device=u.device, dtype=u.dtype)

    # Loop over the blocks of u
    for i in range(0, nx, block_size):
        for j in range(0, ny, block_size):
            # Define the block limits
            i_end = min(i + block_size, nx)
            j_end = min(j + block_size, ny)

            # Compute Jacobian for the current block of u with respect to all k
            u_block = u[:, i:i_end, j:j_end]
            block_jacobian = batchJacobian_vmap(u_block, k, graphed=graphed)  # k is used entirely

            # Place the computed Jacobian in the appropriate slice
            jacobian[:, i:i_end, j:j_end, :, :] = block_jacobian

    return jacobian

def load_and_normalize_datasets(PATH_dataset, PATH_saved_normalizers, case, Mode, device, enable_normalizer=True, batch_size=32):
    
    train_dataset = torch.load(PATH_dataset + f'train_dataset_{case}.pt')
    eval_dataset = torch.load(PATH_dataset + f'eval_dataset_{case}.pt')

    if enable_normalizer:
        # Extract training data tensors
        K_train = train_dataset.tensors[0]
        u_in_train = train_dataset.tensors[1]
        u_out_train = train_dataset.tensors[2]
        jac_train = train_dataset.tensors[3]

        # Initialize normalizers using training data
        u_in_normalizer = UnitGaussianNormalizer(u_in_train)
        u_out_normalizer = UnitGaussianNormalizer(u_out_train)

        # Normalize training data
        u_in_train = u_in_normalizer.encode(u_in_train)
        u_out_train = u_out_normalizer.encode(u_out_train)

        # Create normalized training dataset
        train_dataset = TensorDataset(K_train, u_in_train, u_out_train, jac_train)

        # Clean up memory
        del K_train, u_in_train, u_out_train, jac_train

        # Extract evaluation data tensors
        K_eval = eval_dataset.tensors[0]
        u_in_eval = eval_dataset.tensors[1]
        u_out_eval = eval_dataset.tensors[2]
        jac_eval = eval_dataset.tensors[3]

        # Normalize evaluation data using the same normalizers
        u_in_eval = u_in_normalizer.encode(u_in_eval)
        u_out_eval = u_out_normalizer.encode(u_out_eval)

        # Create normalized evaluation dataset
        eval_dataset = TensorDataset(K_eval, u_in_eval, u_out_eval, jac_eval)

        # Clean up memory
        del K_eval, u_in_eval, u_out_eval, jac_eval

        # Save the normalizer to a file
        with open(PATH_saved_normalizers + f'u_in_normalizer_{Mode}_{train_dataset.tensors[0].shape[0]}.pkl', 'wb') as f:
            pickle.dump(u_in_normalizer, f)
        with open(PATH_saved_normalizers + f'u_out_normalizer_{Mode}_{train_dataset.tensors[0].shape[0]}.pkl', 'wb') as f:
            pickle.dump(u_out_normalizer, f)

        u_in_normalizer.to(device)
        u_out_normalizer.to(device)
    
    print(f'Training with {train_dataset.tensors[0].shape[0]} samples and evaluation with {eval_dataset.tensors[0].shape[0]} samples')

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    if enable_normalizer == False:
        u_in_normalizer, u_out_normalizer = None, None

    return train_loader, eval_loader, u_in_normalizer, u_out_normalizer


def count_parameters(model):
    """
    Counts the number of learnable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: Total number of learnable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_errors(pred, true):
    """
    Calculate the L2 error, R2 score, and Max Abs Error between predictions and true values.
    
    Parameters:
        pred (torch.Tensor): The predicted values with shape [nbatch, nx, ny].
        true (torch.Tensor): The true values with shape [nbatch, nx, ny].
    
    Returns:
        tuple: (l2_error, r2_score, max_abs_error) where l2_error is the root mean squared error (RMSE),
               r2_score is the coefficient of determination, and max_abs_error is the maximum absolute error.
    """
    if not (pred.shape == true.shape):
        raise ValueError("Predictions and true values must have the same shape.")
    
    # Calculate Mean Squared Error (MSE)
    mse = torch.mean((pred - true) ** 2)
    
    # Calculate L2 error (Root Mean Squared Error, RMSE)
    l2_error = torch.sqrt(mse)
    
    # Calculate Total Sum of Squares (TSS)
    tss = torch.sum((true - torch.mean(true)) ** 2)
    
    # Calculate Residual Sum of Squares (RSS)
    rss = torch.sum((true - pred) ** 2)
    
    # Calculate R2 score
    r2_score = 1 - rss / tss

    # Calculate Max Absolute Error
    max_abs_error = torch.max(torch.abs(pred - true))
    
    return l2_error.item(), r2_score.item(), max_abs_error.item()

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")


# def prepare_dataset(path, train_number, eval_number, test_number, device):
#     '''
#     Function to create dataloaders from sub_datasets for Ground_water case.
#     Assumes all parts have the same shape and there are enough parts to meet the dataset size requirements.
#     '''
#     files = [f for f in os.listdir(path) if f.startswith('Ground_water_dataset_part_') and f.endswith('.pt')]
#     if not files:
#         raise ValueError("No dataset files found in the provided path.")

#     # Calculate how many datasets we need based on the desired numbers of training, evaluation, and test sets
#     M = train_number + eval_number + test_number
#     ss = torch.load(os.path.join(path, files[0]), map_location=device).tensors[0].shape[0]
#     n = int(M / ss) + (1 if M % ss != 0 else 0)

#     # Randomly select the files to load
#     random_indices = random.sample(range(len(files)), min(n, len(files)))

#     # Initialize tensors
#     K, Solution, Jac = None, None, None

#     # Load data into tensors with a progress bar
#     for index in tqdm(random_indices, desc="Loading dataset parts"):
#         dt = torch.load(os.path.join(path, files[index]), map_location=device)
#         if K is None:
#             # Initialize tensors with the first dataset
#             K = dt.tensors[0].detach().unsqueeze(0)
#             K = K.to(torch.float16)
#             Solution = dt.tensors[1].detach().unsqueeze(0)
#             Solution = Solution.to(torch.float16)
#             Jac = dt.tensors[2].detach().unsqueeze(0)
#             Jac = Jac.to(torch.float16)
#         else:
#             # Concatenate directly into tensors
#             K = torch.cat((K, dt.tensors[0].detach().unsqueeze(0)), dim=0)
#             K = K.to(torch.float16)
#             Solution = torch.cat((Solution, dt.tensors[1].detach().unsqueeze(0)), dim=0)
#             Solution = Solution.to(torch.float16)
#             Jac = torch.cat((Jac, dt.tensors[2].detach().unsqueeze(0)), dim=0)
#             Jac = Jac.to(torch.float16)
#     # Create the dataset
#     dataset = TensorDataset(K, Solution, Jac)
#     return datase

# def prepare_dataset(path_temp, PATH_norm, train_number, eval_number, test_number, device='cpu', use_normalizer=False):
#     '''
#     Function to create dataloaders from sub_datasets for Ground_water case
#     '''
#     path = path_temp
#     files = [f for f in os.listdir(path) if f.startswith('Ground_water_dataset_part_') and f.endswith('.pt')]
#     first_dataset = torch.load(path + files[0])
#     ss = first_dataset.tensors[0].shape[0]
#     # # Define the sizes of each subset
#     # eval_number = int(0.15 * train_number)  # 15% of the train dataset
#     # test_number = int(0.15 * train_number)  # 15% of the train dataset

#     M = train_number + eval_number + test_number
#     part_train = int(train_number/ss)
#     part_eval = int(eval_number/ss)
#     part_test = int(test_number/ss)

#     random_indices = random.sample(range(1, len(files)+1), part_train + part_eval + part_test)  # Generate n random indices from 1 to 40

#     train_indices = random_indices[:part_train]
#     eval_indices =  random_indices[part_train:(part_train + part_eval)]
#     test_indices = random_indices[(part_train + part_eval):]


#     # Lists to hold the data from each part
#     K_list, u_in_list, u_out_list, Jac_list = [], [], [], []

#     # Loop through each part and load data into lists with progress bar
#     for index in tqdm(train_indices, desc="Preparing train dataset"):
#         # Load each part into CPU memory
#         dt = torch.load(os.path.join(path, f'Ground_water_dataset_part_s_{index}.pt'), map_location=device)
#         # Append each tensor to its respective list after detaching it from the computation graph
#         K_list.append(dt.tensors[0].detach())
#         u_in_list.append(dt.tensors[1][..., 0].detach())
#         u_out_list.append(dt.tensors[1][..., -1].detach())
#         Jac_list.append(dt.tensors[2].detach())

#     # Concatenate all tensors in each list to form single large tensors
#     K = torch.cat(K_list, dim=0)
#     u_in = torch.cat(u_in_list, dim=0)
#     u_out = torch.cat(u_out_list, dim=0)
#     Jac = torch.cat(Jac_list, dim=0)
#     train_dataset = TensorDataset(K, u_in, u_out, Jac)

#     # Lists to hold the data from each part
#     K_list, u_in_list, u_out_list, Jac_list = [], [], [], []

#     # Loop through each part and load data into lists with progress bar
#     for index in tqdm(eval_indices, desc="Preparing evaluation dataset"):
#         # Load each part into CPU memory
#         dt = torch.load(os.path.join(path, f'Ground_water_dataset_part_s_{index}.pt'), map_location=device)
#         # Append each tensor to its respective list after detaching it from the computation graph
#         K_list.append(dt.tensors[0].detach())
#         u_in_list.append(dt.tensors[1][..., 0].detach())
#         u_out_list.append(dt.tensors[1][..., -1].detach())
#         Jac_list.append(dt.tensors[2].detach())

#     # Concatenate all tensors in each list to form single large tensors
#     K = torch.cat(K_list, dim=0)
#     u_in = torch.cat(u_in_list, dim=0)
#     u_out = torch.cat(u_out_list, dim=0)
#     Jac = torch.cat(Jac_list, dim=0)
#     eval_dataset = TensorDataset(K, u_in, u_out, Jac)

#     # Lists to hold the data from each part
#     K_list, u_in_list, u_out_list, Jac_list = [], [], [], []

#     # Loop through each part and load data into lists with progress bar
#     for index in tqdm(test_indices, desc="Preparing test dataset"):
#         # Load each part into CPU memory
#         dt = torch.load(os.path.join(path, f'Ground_water_dataset_part_s_{index}.pt'), map_location=device)
#         # Append each tensor to its respective list after detaching it from the computation graph
#         K_list.append(dt.tensors[0].detach())
#         u_in_list.append(dt.tensors[1][..., 0].detach())
#         u_out_list.append(dt.tensors[1][..., -1].detach())
#         Jac_list.append(dt.tensors[2].detach())

#     # Concatenate all tensors in each list to form single large tensors
#     K = torch.cat(K_list, dim=0)
#     u_in = torch.cat(u_in_list, dim=0)
#     u_out = torch.cat(u_out_list, dim=0)
#     Jac = torch.cat(Jac_list, dim=0)
#     test_dataset = TensorDataset(K, u_in, u_out, Jac)


#     # Extract data for training
#     P_train = train_dataset.tensors[0].to(device)
#     u_in_train = train_dataset.tensors[1].to(device)
#     u_out_train = train_dataset.tensors[2].to(device)
#     du_train = train_dataset.tensors[3].to(device)

#     # Extract data for evaluation
#     P_eval = eval_dataset.tensors[0].to(device)
#     u_in_eval = eval_dataset.tensors[1].to(device)
#     u_out_eval = eval_dataset.tensors[2].to(device)
#     du_eval = eval_dataset.tensors[3].to(device)

#     # Extract data for testing
#     P_test = test_dataset.tensors[0].to(device)
#     u_in_test = test_dataset.tensors[1].to(device)
#     u_out_test = test_dataset.tensors[2].to(device)
#     du_test = test_dataset.tensors[3].to(device)


#     if use_normalizer:
#         u_in_normalizer = UnitGaussianNormalizer(u_in_train)
#         u_out_normalizer = UnitGaussianNormalizer(u_out_train)

#         u_in_train = u_in_normalizer.encode(u_in_train)
#         u_in_test =  u_in_normalizer.encode(u_in_test)
#         u_in_eval =  u_in_normalizer.encode(u_in_eval)

#         u_out_train = u_out_normalizer.encode(u_out_train)
#         u_out_test =  u_out_normalizer.encode(u_out_test)
#         u_out_eval =  u_out_normalizer.encode(u_out_eval)

#         # Check if the directory exists, and create it if it doesn't
#         if not os.path.exists(PATH_norm):
#             os.makedirs(PATH_norm)

#         if save_train:
#             # Save the normalizer to a file
#             with open(PATH_norm + f'u_in_normalizer_{train_number}_{Mode}.pkl', 'wb') as f:
#                 pickle.dump(u_in_normalizer, f)
#             with open(PATH_norm + f'u_out_normalizer_{train_number}_{Mode}.pkl', 'wb') as f:
#                 pickle.dump(u_out_normalizer, f)

#     # Create TensorDatasets
#     train_dataset = TensorDataset(P_train, u_in_train, u_out_train, du_train)
#     eval_dataset = TensorDataset(P_eval, u_in_eval, u_out_eval, du_eval)
#     test_dataset = TensorDataset(P_test, u_in_test, u_out_test, du_test)

#     del u_out_train, u_out_test, u_out_eval, u_in_train, u_in_test, u_in_eval, P_test, P_eval, du_eval, P_train, du_train, du_test
#     torch.cuda.empty_cache()


#     return train_dataset, eval_dataset, test_dataset

def prepare_dataset(Mode, path_temp, PATH_norm, PATH_test_dataset_save, train_number, eval_number, test_number, batch_size, threshold, device1, device2='cpu', use_normalizer=False):
    '''
    Function to create dataloaders from sub_datasets for the Groundwater case.
    Sequentially fills train, then eval, and then test datasets from the files.
    '''
    path = path_temp
    files = [f for f in os.listdir(path) if f.startswith('Ground_water_dataset_part_') and f.endswith('.pt')]
    random.shuffle(files)

    # Lists to hold the data from each part
    K_list_train, u_in_list_train, u_out_list_train, Jac_list_train = [], [], [], []
    K_list_eval, u_in_list_eval, u_out_list_eval, Jac_list_eval = [], [], [], []
    K_list_test, u_in_list_test, u_out_list_test, Jac_list_test = [], [], [], []

    # Current data filling status
    current_dataset = 'train'
    assigned_train, assigned_eval, assigned_test = 0, 0, 0

    # Loop through each part and load data into lists
    for file in tqdm(files, desc="Loading subdatasets"):
        # Load each part into CPU memory
        dt = torch.load(os.path.join(path, file), map_location=device2)

        # Process each item in the dataset
        for i in range(dt.tensors[0].shape[0]):  # Assume dt[0] has the primary index
            # Check thresholds
            if dt.tensors[2][i].max() > threshold or dt.tensors[2][i].min() < -threshold:
                continue  # Skip this sample if it fails threshold checks

            # Check if we should switch to the next dataset
            if current_dataset == 'train' and assigned_train >= train_number:
                K_train = torch.cat(K_list_train, dim=0)
                del K_list_train
                u_in_train = torch.cat(u_in_list_train, dim=0)
                del u_in_list_train
                u_out_train = torch.cat(u_out_list_train, dim=0)
                del u_out_list_train
                Jac_train = torch.cat(Jac_list_train, dim=0)
                del Jac_list_train
                current_dataset = 'eval'  # Switch to filling eval
            if current_dataset == 'eval' and assigned_eval >= eval_number:
                K_eval = torch.cat(K_list_eval, dim=0)
                del K_list_eval
                u_in_eval = torch.cat(u_in_list_eval, dim=0)
                del u_in_list_eval
                u_out_eval = torch.cat(u_out_list_eval, dim=0)
                del u_out_list_eval
                Jac_eval = torch.cat(Jac_list_eval, dim=0)
                del Jac_list_eval
                current_dataset = 'test'  # Switch to filling test
            if current_dataset == 'test' and assigned_test >= test_number:
                break  # Stop processing if test is also filled

            # Add to the appropriate dataset
            if current_dataset == 'train':
                K_list_train.append(dt.tensors[0][i, ...].unsqueeze(0))
                u_in_list_train.append(dt.tensors[1][i, ..., 0].unsqueeze(0))
                u_out_list_train.append(dt.tensors[1][i, ..., -1].unsqueeze(0))
                Jac_list_train.append(dt.tensors[2][i, ...].unsqueeze(0))
                assigned_train += 1
            elif current_dataset == 'eval':
                K_list_eval.append(dt.tensors[0][i, ...].unsqueeze(0))
                u_in_list_eval.append(dt.tensors[1][i, ..., 0].unsqueeze(0))
                u_out_list_eval.append(dt.tensors[1][i, ..., -1].unsqueeze(0))
                Jac_list_eval.append(dt.tensors[2][i, ...].unsqueeze(0))
                assigned_eval += 1
            elif current_dataset == 'test':
                K_list_test.append(dt.tensors[0][i, ...].unsqueeze(0))
                u_in_list_test.append(dt.tensors[1][i, ..., 0].unsqueeze(0))
                u_out_list_test.append(dt.tensors[1][i, ..., -1].unsqueeze(0))
                Jac_list_test.append(dt.tensors[2][i, ...].unsqueeze(0))
                assigned_test += 1

    K_test = torch.cat(K_list_test, dim=0)
    del K_list_test
    u_in_test = torch.cat(u_in_list_test, dim=0)
    del u_in_list_test
    u_out_test = torch.cat(u_out_list_test, dim=0)
    del u_out_list_test
    Jac_test = torch.cat(Jac_list_test, dim=0)
    del Jac_list_test

    # # Change device to computational GPU
    # # Extract data for training
    # K_train = K_train.to(device1)
    # u_in_train = u_in_train.to(device1)
    # u_out_train = u_out_train.to(device1)
    # Jac_train = Jac_train.to(device1)

    # # Extract data for evaluation
    # K_eval = K_eval.to(device1)
    # u_in_eval = u_in_eval.to(device1)
    # u_out_eval = u_out_eval.to(device1)
    # Jac_eval = Jac_eval.to(device1)

    # # Extract data for testing
    # K_test = K_test.to(device1)
    # u_in_test = u_in_test.to(device1)
    # u_out_test = u_out_test.to(device1)
    # Jac_test = Jac_test.to(device1)

    if use_normalizer:
        u_in_normalizer = UnitGaussianNormalizer(u_in_train)
        u_out_normalizer = UnitGaussianNormalizer(u_out_train)

        u_in_train = u_in_normalizer.encode(u_in_train)
        u_in_test =  u_in_normalizer.encode(u_in_test)
        u_in_eval =  u_in_normalizer.encode(u_in_eval)

        u_out_train = u_out_normalizer.encode(u_out_train)
        u_out_test =  u_out_normalizer.encode(u_out_test)
        u_out_eval =  u_out_normalizer.encode(u_out_eval)

        if save_train:
            # Save the normalizer to a file
            with open(PATH_norm + f'u_in_normalizer_{train_number}_{Mode}.pkl', 'wb') as f:
                pickle.dump(u_in_normalizer, f)
            with open(PATH_norm + f'u_out_normalizer_{train_number}_{Mode}.pkl', 'wb') as f:
                pickle.dump(u_out_normalizer, f)

    # Create TensorDatasets
    train_dataset = TensorDataset(K_train, u_in_train, u_out_train, Jac_train)
    eval_dataset =  TensorDataset(K_eval,  u_in_eval,  u_out_eval,  Jac_eval)
    test_dataset =  TensorDataset(K_test,  u_in_test,  u_out_test,  Jac_test)


    del (K_train, u_in_train, u_out_train, Jac_train, 
         K_eval, u_in_eval, u_out_eval, Jac_eval, 
         K_test, u_in_test, u_out_test, Jac_test)

    torch.cuda.empty_cache()

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    eval_loader =  DataLoader(eval_dataset, batch_size=batch_size)
    test_loader =  DataLoader(test_dataset)

    # Save the test dataset  
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(PATH_test_dataset_save):
        os.makedirs(PATH_test_dataset_save)
    torch.save(test_loader, PATH_test_dataset_save + f'test_dataset_{Mode}_{train_number}.pt')
    
    del test_loader
    return train_loader, eval_loader


def prepare_dataset2(Mode, path_temp, PATH_test_dataset_save, train_number, eval_number, test_number, threshold, device='cpu'):
    '''
    Function to create dataloaders from sub_datasets for the Groundwater case.
    Sequentially fills train, then eval, and then test datasets from the files.
    '''
    path = path_temp
    files = [f for f in os.listdir(path) if f.startswith('Ground_water_dataset_part_') and f.endswith('.pt')]
    random.shuffle(files)

    # Lists to hold the data from each part
    K_list_train, u_in_list_train, u_out_list_train, Jac_list_train = [], [], [], []
    K_list_eval, u_in_list_eval, u_out_list_eval, Jac_list_eval = [], [], [], []
    K_list_test, u_in_list_test, u_out_list_test, Jac_list_test = [], [], [], []

    # Current data filling status
    current_dataset = 'train'
    assigned_train, assigned_eval, assigned_test = 0, 0, 0

    # Loop through each part and load data into lists
    for file in tqdm(files, desc="Loading subdatasets"):
        # Load each part into CPU memory
        dt = torch.load(os.path.join(path, file), map_location=device, weights_only=False)

        # Process each item in the dataset
        for i in range(dt.tensors[0].shape[0]):  # Assume dt[0] has the primary index
            # Check thresholds
            if dt.tensors[2][i].max() > threshold or dt.tensors[2][i].min() < -threshold:
                continue  # Skip this sample if it fails threshold checks

            # Check if we should switch to the next dataset
            if current_dataset == 'train' and assigned_train >= train_number:
                K_train = torch.cat(K_list_train, dim=0)
                del K_list_train
                u_in_train = torch.cat(u_in_list_train, dim=0)
                del u_in_list_train
                u_out_train = torch.cat(u_out_list_train, dim=0)
                del u_out_list_train
                Jac_train = torch.cat(Jac_list_train, dim=0)
                del Jac_list_train
                current_dataset = 'eval'  # Switch to filling eval
            if current_dataset == 'eval' and assigned_eval >= eval_number:
                K_eval = torch.cat(K_list_eval, dim=0)
                del K_list_eval
                u_in_eval = torch.cat(u_in_list_eval, dim=0)
                del u_in_list_eval
                u_out_eval = torch.cat(u_out_list_eval, dim=0)
                del u_out_list_eval
                Jac_eval = torch.cat(Jac_list_eval, dim=0)
                del Jac_list_eval
                current_dataset = 'test'  # Switch to filling test
            if current_dataset == 'test' and assigned_test >= test_number:
                break  # Stop processing if test is also filled

            # Add to the appropriate dataset
            if current_dataset == 'train':
                K_list_train.append(dt.tensors[0][i, ...].unsqueeze(0))
                u_in_list_train.append(dt.tensors[1][i, ..., 0].unsqueeze(0))
                u_out_list_train.append(dt.tensors[1][i, ..., -1].unsqueeze(0))
                Jac_list_train.append(dt.tensors[2][i, ...].unsqueeze(0))
                assigned_train += 1
            elif current_dataset == 'eval':
                K_list_eval.append(dt.tensors[0][i, ...].unsqueeze(0))
                u_in_list_eval.append(dt.tensors[1][i, ..., 0].unsqueeze(0))
                u_out_list_eval.append(dt.tensors[1][i, ..., -1].unsqueeze(0))
                Jac_list_eval.append(dt.tensors[2][i, ...].unsqueeze(0))
                assigned_eval += 1
            elif current_dataset == 'test':
                K_list_test.append(dt.tensors[0][i, ...].unsqueeze(0))
                u_in_list_test.append(dt.tensors[1][i, ..., 0].unsqueeze(0))
                u_out_list_test.append(dt.tensors[1][i, ..., -1].unsqueeze(0))
                Jac_list_test.append(dt.tensors[2][i, ...].unsqueeze(0))
                assigned_test += 1

    K_test = torch.cat(K_list_test, dim=0)
    del K_list_test
    u_in_test = torch.cat(u_in_list_test, dim=0)
    del u_in_list_test
    u_out_test = torch.cat(u_out_list_test, dim=0)
    del u_out_list_test
    Jac_test = torch.cat(Jac_list_test, dim=0)
    del Jac_list_test



    # Create TensorDatasets
    train_dataset = TensorDataset(K_train, u_in_train, u_out_train, Jac_train)
    eval_dataset =  TensorDataset(K_eval,  u_in_eval,  u_out_eval,  Jac_eval)
    test_dataset =  TensorDataset(K_test,  u_in_test,  u_out_test,  Jac_test)


    del (K_train, u_in_train, u_out_train, Jac_train, 
         K_eval, u_in_eval, u_out_eval, Jac_eval, 
         K_test, u_in_test, u_out_test, Jac_test)

    torch.cuda.empty_cache()


    # Save the test dataset  
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(PATH_test_dataset_save):
        os.makedirs(PATH_test_dataset_save)

    torch.save(train_dataset, PATH_test_dataset_save +  f'train_dataset_{Mode}_{train_number}.pt')
    torch.save(eval_dataset, PATH_test_dataset_save +   f'eval_dataset_{Mode}_{eval_number}.pt')
    torch.save(test_dataset, PATH_test_dataset_save +   f'test_dataset_{Mode}_{test_number}.pt')



def upscale_tensor(jac, N):
    """
    Upscales a tensor using bilinear interpolation on the 2nd and 3rd dimensions.

    Parameters:
    jac (torch.Tensor): Input tensor with shape [nbatch, nx1, ny1, nx2, ny2].
    N (int): Upscaling factor.

    Returns:
    torch.Tensor: Upscaled tensor with shape [nbatch, N * nx1, N * ny1, nx2, ny2].
    """
    nbatch, nx1, ny1, nx2, ny2 = jac.shape

    # Reshape the tensor to merge dimensions [nx2, ny2] into the batch dimension
    reshaped_tensor = jac.permute(0, 3, 4, 1, 2).reshape(nbatch * nx2 * ny2, 1, nx1, ny1)

    # Perform interpolation
    upscaled_tensor = F.interpolate(reshaped_tensor, scale_factor=N, mode='bilinear', align_corners=False)

    # Reshape the tensor back to the original dimensions
    final_tensor = upscaled_tensor.view(nbatch, nx2, ny2, N * nx1, N * ny1).permute(0, 3, 4, 1, 2)

    return final_tensor


def upscale_tensor_3d(tensor, N):
    """
    Upscales a 3D tensor using bilinear interpolation on the 2nd and 3rd dimensions.

    Parameters:
    tensor (torch.Tensor): Input tensor with shape [nb, nx, ny].
    N (int): Upscaling factor.

    Returns:
    torch.Tensor: Upscaled tensor with shape [nb, N * nx, N * ny].
    """
    # Ensure tensor is in the correct shape for interpolation
    tensor_reshaped = tensor.unsqueeze(1)  # Add a channel dimension, resulting in [nb, 1, nx, ny]

    # Perform interpolation
    upscaled_tensor = F.interpolate(tensor_reshaped, scale_factor=N, mode='bilinear', align_corners=False)

    # Remove the channel dimension added earlier
    final_tensor = upscaled_tensor.squeeze(1)  # Resulting shape [nb, N * nx, N * ny]

    return final_tensor





#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path, mode='r')
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss

# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c
