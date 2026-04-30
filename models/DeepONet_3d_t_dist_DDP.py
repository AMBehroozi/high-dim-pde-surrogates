import torch
import torch.nn as nn
import os
import torch
import torch.nn as nn
import os


class DNO2DTime(nn.Module):
    def __init__(self, nx, ny, T_in, T_out, branch_layers, trunk_layers, num_stacks=2):
        super(DNO2DTime, self).__init__()
        self.nx, self.ny = nx, ny
        self.T_in, self.T_out = T_in, T_out
        self.num_stacks = num_stacks  # Number of stacks (e.g., 2 for u0 and parameters)

        # Branch network for u0
        branch_input_size_u0 = nx * ny * T_in  # Input size for u0: [nb, nx, ny, T_in] flattened
        
        branch_layers_u0 = [branch_input_size_u0] + branch_layers
        self.branch_net_u0 = self._build_mlp(branch_layers_u0)

        # Branch network for parameters
        branch_input_size_params = nx * ny * T_in  # Input size for parameters after repeating
        branch_layers_params = [branch_input_size_params] + branch_layers
        self.branch_net_params = self._build_mlp(branch_layers_params)

        # Trunk network for (x, y, t)
        trunk_layers = [3] + trunk_layers  # Input is (x, y, t), so 3 dimensions
        # Adjust the output dimension of the trunk network to be num_stacks * hidden_dim
        trunk_layers[-1] = trunk_layers[-1] * self.num_stacks
        self.trunk_net = self._build_mlp(trunk_layers)

        # Ensure the output dimensions match for the dot product
        assert branch_layers_u0[-1] == branch_layers_params[-1], \
            f"Mismatch: Branch_u0 output {branch_layers_u0[-1]} != Branch_params output {branch_layers_params[-1]}"
        assert branch_layers_u0[-1] == trunk_layers[-1] // self.num_stacks, \
            f"Mismatch: Branch output {branch_layers_u0[-1]} != Trunk output per stack {trunk_layers[-1] // self.num_stacks}"

    def _build_mlp(self, layers):
        """Helper function to build an MLP."""
        mlp = []
        for i in range(len(layers) - 1):
            mlp.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mlp.append(nn.ReLU())  # Activation except last layer
        return nn.Sequential(*mlp)

    def _prepare_grid(self, batch_size, device):
        """Generate spatial-temporal grid (excluding parameters)."""
        x = torch.linspace(0, 1, self.nx, device=device)
        y = torch.linspace(0, 1, self.ny, device=device)
        t = torch.linspace(0, 1, self.T_out, device=device)

        grid_x, grid_y, grid_t = torch.meshgrid(x, y, t, indexing='ij')  # [nx, ny, T_out]

        # Stack spatial-temporal coordinates (x, y, t)
        grid = torch.stack([grid_x, grid_y, grid_t], dim=-1).reshape(1, -1, 3)  # Shape: [1, nx * ny * T_out, 3]
        return grid.repeat(batch_size, 1, 1)  # [batch_size, nx * ny * T_out, 3]

    def forward(self, u0, parameters):
        batch_size = u0.shape[0]
        device = u0.device

        # Process u0 through its branch network
        branch_input_u0 = u0.reshape(batch_size, -1)  # [batch_size, nx * ny * T_in]
        branch_output_u0 = self.branch_net_u0(branch_input_u0)  # [batch_size, hidden_dim]

        # Process parameters through its branch network
        parameters_expanded = parameters.unsqueeze(-1)  # [batch_size, nx, ny, 1]
        parameters_repeated = parameters_expanded.repeat(1, 1, 1, self.T_in)  # [batch_size, nx, ny, T_in]
        branch_input_params = parameters_repeated.reshape(batch_size, -1)  # [batch_size, nx * ny * T_in]
        branch_output_params = self.branch_net_params(branch_input_params)  # [batch_size, hidden_dim]

        # Combine the outputs of the two branch networks (element-wise multiplication)
        branch_output = branch_output_u0 * branch_output_params  # [batch_size, hidden_dim]

        # Process the spatial-temporal grid through the trunk network
        trunk_input = self._prepare_grid(batch_size, device)  # [batch_size, nx * ny * T_out, 3]
        trunk_output = self.trunk_net(trunk_input)  # [batch_size, nx * ny * T_out, hidden_dim * num_stacks]

        # Split the trunk output into num_stacks parts
        hidden_dim = branch_output.shape[-1]  # hidden_dim per stack
        trunk_outputs = torch.split(trunk_output, hidden_dim, dim=-1)  # List of [batch_size, nx * ny * T_out, hidden_dim]

        # Combine the trunk outputs using element-wise multiplication
        trunk_output_combined = trunk_outputs[0]
        for i in range(1, len(trunk_outputs)):
            trunk_output_combined = trunk_output_combined * trunk_outputs[i]  # [batch_size, nx * ny * T_out, hidden_dim]

        # Combine the branch and trunk outputs
        output = torch.einsum('bi,bni->bn', branch_output, trunk_output_combined)  # [batch_size, nx * ny * T_out]
        return output.view(batch_size, self.nx, self.ny, self.T_out)  # [batch_size, nx, ny, T_out]


# class DNO2DTime(nn.Module):
#     def __init__(self, nx, ny, T_in, T_out, branch_layers, trunk_layers):
#         super(DNO2DTime, self).__init__()
#         self.nx, self.ny = nx, ny
#         self.T_in, self.T_out = T_in, T_out

#         # ✅ Fix: Include parameters in branch input
#         branch_input_size = nx * ny * (T_in + 1)  # (T_in + 1) includes parameters
#         branch_layers = [branch_input_size] + branch_layers
#         self.branch_net = self._build_mlp(branch_layers)

#         # ✅ Fix: Only use spatial-temporal grid for trunk net
#         trunk_layers = [3] + trunk_layers  # Only (x, y, t)
#         self.trunk_net = self._build_mlp(trunk_layers)

#         assert branch_layers[-1] == trunk_layers[-1], \
#             f"Mismatch: Branch output {branch_layers[-1]} != Trunk output {trunk_layers[-1]}"

#     def _build_mlp(self, layers):
#         """Helper function to build an MLP."""
#         mlp = []
#         for i in range(len(layers) - 1):
#             mlp.append(nn.Linear(layers[i], layers[i + 1]))
#             if i < len(layers) - 2:
#                 mlp.append(nn.ReLU())  # Activation except last layer
#         return nn.Sequential(*mlp)

#     def _prepare_grid(self, batch_size, device):
#         """Generate spatial-temporal grid (excluding parameters)."""
#         x = torch.linspace(0, 1, self.nx, device=device)
#         y = torch.linspace(0, 1, self.ny, device=device)
#         t = torch.linspace(0, 1, self.T_out, device=device)

#         grid_x, grid_y, grid_t = torch.meshgrid(x, y, t, indexing='ij')  # [nx, ny, T_out]

#         # Stack spatial-temporal coordinates (x, y, t)
#         grid = torch.stack([grid_x, grid_y, grid_t], dim=-1).reshape(1, -1, 3)  # Shape: [1, nx * ny * T_out, 3]
#         return grid.repeat(batch_size, 1, 1)  # [batch_size, nx * ny * T_out, 3]

#     def forward(self, u0, parameters):
#         batch_size = u0.shape[0]
#         device = u0.device
#         parameters_expanded = parameters.unsqueeze(-1)  # Shape: [batch_size, nx, ny, 1]
#         u0_with_params = torch.cat([u0, parameters_expanded], dim=-1)  # Shape: [batch_size, nx, ny, T_in + 1]
#         branch_input = u0_with_params.reshape(batch_size, -1)  # [batch_size, nx * ny * (T_in + 1)]
#         branch_output = self.branch_net(branch_input)  # [batch_size, hidden_dim]
#         trunk_input = self._prepare_grid(batch_size, device)  # [batch_size, nx * ny * T_out, 3]
#         trunk_output = self.trunk_net(trunk_input)  # [batch_size, nx * ny * T_out, trunk_dim]
#         output = torch.einsum('bi,bni->bn', branch_output, trunk_output)  # [batch_size, nx * ny * T_out]
#         return output.view(batch_size, self.nx, self.ny, self.T_out)  # [batch_size, nx, ny, T_out]
