import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y, modes_t):
        super(SpectralConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes_x #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes_y
        self.modes3 = modes_t

        self.scale = (1 / (in_channels * out_channels))
        # Initialize real and imaginary parts separately
        self.weights1_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights1_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        
        self.weights2_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights2_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        
        self.weights3_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights3_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        
        self.weights4_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights4_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))

    def compl_mul3d(self, input_real, input_imag, weights_real, weights_imag):
        # (batch, in_channel, x,y,t), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input_real, weights_real) - \
               torch.einsum("bixyz,ioxyz->boxyz", input_imag, weights_imag), \
               torch.einsum("bixyz,ioxyz->boxyz", input_real, weights_imag) + \
               torch.einsum("bixyz,ioxyz->boxyz", input_imag, weights_real)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])
        x_ft_real, x_ft_imag = x_ft.real, x_ft.imag
        
        # Multiply relevant Fourier modes
        out_ft_real = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, device=x.device)
        out_ft_imag = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, device=x.device)
        
        # First set of modes
        out_ft_real[:, :, :self.modes1, :self.modes2, :self.modes3], \
        out_ft_imag[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft_real[:, :, :self.modes1, :self.modes2, :self.modes3],
                           x_ft_imag[:, :, :self.modes1, :self.modes2, :self.modes3],
                           self.weights1_real, self.weights1_imag)
        
        # Second set of modes
        out_ft_real[:, :, -self.modes1:, :self.modes2, :self.modes3], \
        out_ft_imag[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft_real[:, :, -self.modes1:, :self.modes2, :self.modes3],
                           x_ft_imag[:, :, -self.modes1:, :self.modes2, :self.modes3],
                           self.weights2_real, self.weights2_imag)
        
        # Third set of modes
        out_ft_real[:, :, :self.modes1, -self.modes2:, :self.modes3], \
        out_ft_imag[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft_real[:, :, :self.modes1, -self.modes2:, :self.modes3],
                           x_ft_imag[:, :, :self.modes1, -self.modes2:, :self.modes3],
                           self.weights3_real, self.weights3_imag)
        
        # Fourth set of modes
        out_ft_real[:, :, -self.modes1:, -self.modes2:, :self.modes3], \
        out_ft_imag[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft_real[:, :, -self.modes1:, -self.modes2:, :self.modes3],
                           x_ft_imag[:, :, -self.modes1:, -self.modes2:, :self.modes3],
                           self.weights4_real, self.weights4_imag)
        
        # Combine real and imaginary parts
        out_ft = torch.complex(out_ft_real, out_ft_imag)
        
        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO3d_Full_Sequence(nn.Module):
    def __init__(self, T_in, T_out, modes_x, modes_y, modes_t, width):
        super(FNO3d_Full_Sequence, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """
        self.T_in = T_in
        self.T_out = T_out
        self.modes1 = modes_x
        self.modes2 = modes_y
        self.modes3 = modes_t
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic

        self.fc0 = nn.Linear(self.T_in + 3, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-2).repeat(1, 1, 1, self.T_out, 1)
        grid = self.get_grid(x.shape, x.device)
        
        # torch.Size([2, 64, 64, 40, 10]) torch.Size([2, 64, 64, 40, 3])

        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic SHAPE torch.Size([B, W, H, T, width])
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x).squeeze(-1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

