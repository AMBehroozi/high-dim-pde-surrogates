import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ResidualConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualConvBlock2D, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.activation = nn.GELU()
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.residual_conv is None else self.residual_conv(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)

class CNO(nn.Module):
    def __init__(self, nx, ny, state_size, parameter_size, width, depth, kernel_size=3, unet_depth=4):
        super(CNO, self).__init__()
        self.nx = nx
        self.ny = ny
        self.state_size = state_size
        self.parameter_size = parameter_size
        self.width = width
        self.depth = depth
        self.kernel_size = kernel_size
        self.unet_depth = unet_depth

        # Input projection
        self.fc_in = nn.Linear((state_size + parameter_size), width - 2)

        # U-Net Encoder
        self.encoder = nn.ModuleList()
        for i in range(unet_depth):
            in_channels = width if i == 0 else width * 2**i
            out_channels = width * 2**(i+1)
            self.encoder.append(ResidualConvBlock2D(in_channels, out_channels, kernel_size))

        # Bottleneck
        self.bottleneck = ResidualConvBlock2D(width * 2**unet_depth, width * 2**unet_depth, kernel_size)

        # U-Net Decoder
        self.decoder = nn.ModuleList()
        for i in range(unet_depth):
            in_channels = width * 2**(unet_depth - i) * 2  # Adjusted for concatenation
            out_channels = width * 2**(unet_depth - i - 1)
            self.decoder.append(ResidualConvBlock2D(in_channels, out_channels, kernel_size))

        # Additional convolutional layers
        self.conv_layers = nn.ModuleList([
            ResidualConvBlock2D(width, width, kernel_size) for _ in range(depth)
        ])

        # Output projection
        self.fc_out = nn.Linear(width, state_size)

    def generate_grid(self, batch_size, device):
        x = torch.linspace(0, 1, self.nx, device=device)
        y = torch.linspace(0, 1, self.ny, device=device)
        x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
        return x_grid.unsqueeze(0).repeat(batch_size, 1, 1), y_grid.unsqueeze(0).repeat(batch_size, 1, 1)

    def forward(self, u0, x, y, P):
        batch_size = u0.shape[0]
        device = u0.device
        x_grid, y_grid = self.generate_grid(batch_size, device)
        P_expanded = P.permute(0, 3, 1, 2)
        u0_interp = u0.permute(0, 3, 1, 2)
        inputs = torch.cat([u0_interp, P_expanded], dim=1)

        batch_size, channels, height, width = inputs.shape
        x = self.fc_in(inputs.permute(0, 2, 3, 1).view(batch_size, -1, channels))
        x = x.view(batch_size, height, width, -1)

        x = torch.cat([x, x_grid.unsqueeze(-1), y_grid.unsqueeze(-1)], dim=-1)
        x = x.permute(0, 3, 1, 2)
        
        # Encoder
        encoder_outputs = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            encoder_outputs.append(x)
            x = F.avg_pool2d(x, (2, 2))
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, decoder_layer in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True)
            encoder_output = encoder_outputs[-(i+1)]
            if x.shape[2:] != encoder_output.shape[2:]:
                diff_x = encoder_output.shape[2] - x.shape[2]
                diff_y = encoder_output.shape[3] - x.shape[3]
                x = F.pad(x, (0, diff_y, 0, diff_x))
            x = torch.cat([x, encoder_output], dim=1)
            x = decoder_layer(x)
        
        # Additional conv layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Output projection
        x = x.permute(0, 2, 3, 1)
        x = self.fc_out(x)
        return x

# if __name__ == "__main__":
#     rank, world_size = setup_ddp()
#     device = torch.device(f"cuda:{rank}")
#     model = CNO(nx=64, ny=64, state_size=1, parameter_size=1, width=16, depth=4).to(device)
#     model = DDP(model, device_ids=[rank])

#     nbatch, s0, s1, state_size, parameter_size = 16, 64, 64, 1, 1
#     u_in = torch.rand(nbatch, s0, s1, state_size).to(device)
#     parameters = torch.rand(nbatch, s0, s1, parameter_size).to(device)
#     x = torch.linspace(0, 1, s0).unsqueeze(0).repeat(nbatch, 1).to(device)
#     y = torch.linspace(0, 1, s1).unsqueeze(0).repeat(nbatch, 1).to(device)

#     u_out = model(u_in, x, y, parameters)
#     print(f"Rank {rank}: {u_out.shape}")
    
#     dist.barrier()
#     dist.destroy_process_group()
