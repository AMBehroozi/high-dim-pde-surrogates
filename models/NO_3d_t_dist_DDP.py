import sys

sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")



from models.FNO_3d_t_dist_DDP import *
from models.WNO_3d_t_dist_DDP import *
from models.DeepONet_3d_t_dist_DDP import *
from models.CNO_3d_t_dist_DDP import *

def create_model(operator_type, T_in, T_out, nx, ny, 
                 width_CNO=None, depth_CNO=None, kernel_size=None, unet_depth=None,  # CNO inputs
                 mode1=None, mode2=None, mode3=None, width_FNO=None,  # FNO inputs
                 wavelet=None, level=None, layers=None, grid_range=None, width_WNO=None,  # WNO inputs
                 branch_layers=None, trunk_layers=None):  # DeepONet inputs
    """
    Generalized function to create a model based on the specified model_type.
    Irrelevant inputs can be set to None for models that don't require them.

    Args:
        operator_type (str): Type of model to create. Options: 'CNO', 'FNO', 'WNO', 'DeepONet'.
        T_in (int): Number of input time steps.
        T_out (int): Number of output time steps.
        nx (int): Spatial resolution in the x-direction.
        ny (int): Spatial resolution in the y-direction.

        # CNO-specific inputs
        width_CNO (int, optional): Width of the CNO model. Required for 'CNO'.
        depth_CNO (int, optional): Depth of the CNO model. Required for 'CNO'.
        kernel_size (int, optional): Kernel size for the CNO model. Required for 'CNO'.
        unet_depth (int, optional): Depth of the U-Net in the CNO model. Required for 'CNO'.

        # FNO-specific inputs
        mode1 (int, optional): Number of Fourier modes in the x-direction. Required for 'FNO'.
        mode2 (int, optional): Number of Fourier modes in the y-direction. Required for 'FNO'.
        mode3 (int, optional): Number of Fourier modes in the time direction. Required for 'FNO'.
        width (int, optional): Width of the FNO model. Required for 'FNO'.

        # WNO-specific inputs
        wavelet (str, optional): Wavelet basis function (e.g., 'db2'). Required for 'WNO'.
        level (int, optional): Level of wavelet decomposition. Required for 'WNO'.
        layers (int, optional): Number of wavelet layers. Required for 'WNO'.
        grid_range (list, optional): Grid range for the WNO model. Required for 'WNO'.

        # DeepONet-specific inputs
        branch_layers (list, optional): List of branch network layer sizes. Required for 'DeepONet'.
        trunk_layers (list, optional): List of trunk network layer sizes. Required for 'DeepONet'.

    Returns:
        model: An instance of the specified model.

    Raises:
        ValueError: If the operator_type is unknown or required inputs for the specified model are missing.
    """
    
    # CNO Model
    if operator_type == 'CNO':
        # Check if all required inputs for CNO are provided
        if None in [width_CNO, depth_CNO, kernel_size, unet_depth]:
            raise ValueError("CNO model requires width_CNO, depth_CNO, kernel_size, and unet_depth to be specified.")
        # Instantiate the CNO model
        model = UNetCNO2DTime(nx, ny, T_in, T_out, width_CNO, depth_CNO, kernel_size, unet_depth)
    
    # FNO Model
    elif operator_type == 'FNO':
        # Check if all required inputs for FNO are provided
        if None in [mode1, mode2, mode3, width_FNO]:
            raise ValueError("FNO model requires mode1, mode2, mode3, and width to be specified.")
        # Instantiate the FNO model
        model = FNO3d(T_in=T_in, T_out=T_out, modes_x=mode1, modes_y=mode2, modes_t=mode3, width=width_FNO)
    
    # DeepONet Model
    elif operator_type == 'DeepONet':
        # Check if all required inputs for DeepONet are provided
        if None in [branch_layers, trunk_layers]:
            raise ValueError("DeepONet model requires branch_layers and trunk_layers to be specified.")
        # Instantiate the DeepONet model
        model = DNO2DTime(nx, ny, T_in, T_out, branch_layers, trunk_layers)
    
    # WNO Model
    elif operator_type == 'WNO':
        # Check if all required inputs for WNO are provided
        if None in [wavelet, level, layers, grid_range, width_WNO]:
            raise ValueError("WNO model requires wavelet, level, layers, grid_range, and width to be specified.")
        # Instantiate the WNO model
        model = WNO3d(T_in, T_out, width_WNO, level, layers=layers, size=[T_out, nx, ny], wavelet=wavelet, grid_range=grid_range, omega=6)
    
    # Unknown Model Type
    else:
        raise ValueError(f"Unknown operator_type: {operator_type}")
    
    return model