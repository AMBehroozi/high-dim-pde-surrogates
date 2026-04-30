import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetCNO2DTime(nn.Module):
    def __init__(self, nx, ny, T_in, T_out, width, depth, kernel_size=3, unet_depth=4):
        super(UNetCNO2DTime, self).__init__()
        self.nx = nx
        self.ny = ny
        self.T_in = T_in
        self.T_out = T_out
        self.width = width
        self.depth = depth
        self.kernel_size = kernel_size
        self.unet_depth = unet_depth

        # Input projection
        self.fc_in = nn.Linear(T_in + 1, width)  # +1 for the parameter channel

        # U-Net Encoder
        self.encoder = nn.ModuleList()
        for i in range(unet_depth):
            in_channels = width if i == 0 else width * 2**i
            out_channels = width * 2**(i+1)
            self.encoder.append(ResidualConvBlock2D(in_channels, out_channels, kernel_size))

        # U-Net Bottleneck
        self.bottleneck = ResidualConvBlock2D(width * 2**unet_depth, width * 2**unet_depth, kernel_size)

        # U-Net Decoder
        self.decoder = nn.ModuleList()
        for i in range(unet_depth):
            in_channels = width * 2**(unet_depth-i+1)
            out_channels = width * 2**(unet_depth-i-1)
            self.decoder.append(ResidualConvBlock2D(in_channels, out_channels, kernel_size))

        # Additional convolutional layers with residual connections
        self.conv_layers = nn.ModuleList([
            ResidualConvBlock2D(width, width, kernel_size) for _ in range(depth)
        ])

        # Output projection
        self.fc_out = nn.Linear(width, T_out)

    def forward(self, u0, parameter):
        batch_size = u0.shape[0]
        device = u0.device

        # Prepare input features
        # u0: [batch, nx, ny, T_in]
        # parameter: [batch, nx, ny]
        parameter = parameter.unsqueeze(-1)  # [batch, nx, ny, 1]
        inputs = torch.cat([u0, parameter], dim=-1)  # [batch, nx, ny, T_in + 1]

        # Apply input projection
        x = self.fc_in(inputs)  # [batch, nx, ny, width]

        # Permute for 2D convolutions
        x = x.permute(0, 3, 1, 2)  # [batch, width, nx, ny]

        # Encoder
        encoder_outputs = []
        for i, encoder_layer in enumerate(self.encoder):
            x = encoder_layer(x)
            encoder_outputs.append(x)
            if i < self.unet_depth - 1:  # Avoid downsampling the last layer
                x = F.avg_pool2d(x, kernel_size=2)  # Downsample in spatial dimensions

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, decoder_layer in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            encoder_output = encoder_outputs[-i-1]
            
            # Pad if necessary
            if x.size(2) != encoder_output.size(2) or x.size(3) != encoder_output.size(3):
                diff_h = encoder_output.size(2) - x.size(2)
                diff_w = encoder_output.size(3) - x.size(3)
                x = F.pad(x, (0, diff_w, 0, diff_h))
            
            x = torch.cat([x, encoder_output], dim=1)  # Concatenate along channel dimension
            x = decoder_layer(x)

        # Additional convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Output projection
        x = x.permute(0, 2, 3, 1)  # [batch, nx, ny, width]
        x = self.fc_out(x)  # [batch, nx, ny, T_out]

        return x


class ResidualConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualConvBlock2D, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm2d(out_channels)  # Use BatchNorm2d instead of InstanceNorm2d
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        
        # Add a 1x1 convolution for residual connection if channel dimensions don't match
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.residual_conv is None else self.residual_conv(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)

