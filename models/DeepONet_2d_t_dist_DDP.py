import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class DNO2DTime(nn.Module):
    def __init__(self, nx, ny, state_size, parameter_size, branch_layers, trunk_layers):
        super(DNO2DTime, self).__init__()
        self.nx = nx
        self.ny = ny
        self.state_size = state_size
        self.parameter_size = parameter_size

        # Compute correct branch input size
        branch_input_size = nx * ny * state_size + nx * ny * parameter_size

        # Branch net
        branch_layers = [branch_input_size] + branch_layers
        self.branch_net = self._build_mlp(branch_layers)

        # Trunk net
        trunk_layers = [2] + trunk_layers  # 2 for x, y coordinates
        self.trunk_net = self._build_mlp(trunk_layers)

        assert branch_layers[-1] == trunk_layers[-1], "Branch and trunk output dimensions must match"
        self.output_dim = branch_layers[-1]

    def _build_mlp(self, layers):
        mlp = nn.ModuleList()
        for i in range(len(layers) - 1):
            mlp.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mlp.append(nn.ReLU())
        return nn.Sequential(*mlp)

    def _prepare_grid(self, x, y, device):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], y.shape[1]
        gridx = x.reshape(batchsize, size_x, 1, 1).repeat([1, 1, size_y, 1])
        gridy = y.reshape(batchsize, 1, size_y, 1).repeat([1, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def forward(self, u, x, y, par):
        batch_size = u.shape[0]

        # Flatten u and parameters properly
        u_flat = u.reshape(batch_size, -1)
        par_flat = par.reshape(batch_size, -1)

        branch_input = torch.cat([u_flat, par_flat], dim=1)
        branch_output = self.branch_net(branch_input)  # Shape: (batch_size, output_dim)

        trunk_input = self._prepare_grid(x, y, u.device)  # Shape: (batch_size, nx, ny, 2)
        trunk_output = self.trunk_net(trunk_input)  # Shape: (batch_size, nx, ny, output_dim)

        # Adjust einsum operation to match shapes
        output = torch.einsum('bi,bxyi->bxy', branch_output, trunk_output)

        return output.unsqueeze(-1)  # Ensure shape: (batch_size, nx, ny, state_size)

