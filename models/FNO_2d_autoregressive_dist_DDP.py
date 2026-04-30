import torch
import torch.nn as nn
import torch.nn.functional as F
# %%
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights1_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    def compl_mul2d(self, input_real, input_imag, weights_real, weights_imag):
        # (batch, in_channel, x,y), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input_real, weights_real) - \
               torch.einsum("bixy,ioxy->boxy", input_imag, weights_imag), \
               torch.einsum("bixy,ioxy->boxy", input_real, weights_imag) + \
               torch.einsum("bixy,ioxy->boxy", input_imag, weights_real)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        x_ft_real, x_ft_imag = x_ft.real, x_ft.imag
        
        # Multiply relevant Fourier modes
        out_ft_real = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, device=x.device)
        out_ft_imag = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, device=x.device)
        
        out_ft_real[:, :, :self.modes1, :self.modes2], out_ft_imag[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft_real[:, :, :self.modes1, :self.modes2], x_ft_imag[:, :, :self.modes1, :self.modes2], 
                             self.weights1_real, self.weights1_imag)
        
        out_ft_real[:, :, -self.modes1:, :self.modes2], out_ft_imag[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft_real[:, :, -self.modes1:, :self.modes2], x_ft_imag[:, :, -self.modes1:, :self.modes2], 
                             self.weights2_real, self.weights2_imag)
        
        # Combine real and imaginary parts
        out_ft = torch.complex(out_ft_real, out_ft_imag)
        
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, T_in, modes1, modes2, width, state_size, parameter_size):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.state_size = state_size
        self.parameter_size = parameter_size
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.T_in = T_in
        self.fc0 = nn.Linear(self.parameter_size + self.T_in * self.state_size + 2, self.width) # input channel is : (u(x, y)*n_state par(x, y)*n_prameters, x, y)
        
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.state_size)

    def process_tensor(self, u):
        # Initial fully connected layer and reshape
        u = self.fc0(u)
        u = u.permute(0, 3, 1, 2)
        u = F.pad(u, [0, self.padding, 0, self.padding])

        # Process through convolution and weight layers
        layers = [(self.conv0, self.w0), (self.conv1, self.w1), 
                (self.conv2, self.w2), (self.conv3, self.w3)]
        for conv, weight in layers:
            u1 = conv(u)
            u2 = weight(u)
            u = u1 + u2
            if conv != self.conv3:  # Apply activation function except for the last layer
                u = F.gelu(u)

        # Final steps after the loop
        u = u[..., :-self.padding, :-self.padding]
        u = u.permute(0, 2, 3, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u

    def forward(self, u, x, y, par):
        nbatch, s0, s1 = u.shape[0], u.shape[1], u.shape[2]
        grid = self.get_grid(x, y, u.device)
        u = u.reshape(nbatch, s0, s1, self.T_in * self.state_size)
        u = torch.cat((u, par, grid), dim=-1)
        out = self.process_tensor(u)
        return out
    
    def get_grid(self, x, y, device):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], y.shape[1]
        gridx = x.reshape(batchsize, size_x, 1, 1).repeat([1, 1, size_y, 1])
        gridy = y.reshape(batchsize, 1, size_y, 1).repeat([1, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)




# # Usage
# nbatch = 16
# s0 = 64
# s1 = 64
# parameter_size = 1
# state_size = 2
# T_in = 10
# modes1, modes2,  width = 8, 8, 10
# device = 'cuda'
# u_in = torch.rand(nbatch, s0, s1, T_in, state_size).to(device) # [nb, s0, s0, state_size]

# parameters = torch.rand(nbatch, s0, s1, parameter_size).to(device) # [nb, s0, s0, parameter_size]
# x = torch.linspace(0, 1, s0).unsqueeze(0).repeat(nbatch, 1).to(device)
# y = torch.linspace(0, 1, s1).unsqueeze(0).repeat(nbatch, 1).to(device)


# model = FNO2d(T_in, modes1, modes2,  width, state_size, parameter_size).to(device)
# u_out = model(u_in, x, y, parameters)  # [nb, s0, s0, state_size]

 
# print(u_out.shape  )
