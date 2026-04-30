try:
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    import ptwt, pywt
    import numpy as np
    from ptwt.conv_transform_3 import wavedec3, waverec3
    from pytorch_wavelets import DWT1D, IDWT1D
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    from pytorch_wavelets import DWT, IDWT 
except ImportError:
    print('Wavelet convolution requires <Pytorch Wavelets>, <PyWavelets>, <Pytorch Wavelet Toolbox> \n \
                    For Pytorch Wavelet Toolbox: $ pip install ptwt \n \
                    For PyWavelets: $ conda install pywavelets \n \
                    For Pytorch Wavelets: $ git clone https://github.com/fbcotter/pytorch_wavelets \n \
                                          $ cd pytorch_wavelets \n \
                                          $ pip install .')
    
""" Def: 3d Wavelet convolutional layer """
class WaveConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet='db4', mode='periodic', omega=6):
        super(WaveConv3d, self).__init__()

        """
        3D Wavelet layer. It does 3D DWT, linear transform, and Inverse dWT.    
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet      : string, Specifies the first level biorthogonal wavelet filters
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights0 : tensor, shape-[in_channels * out_channels * x * y * z]
                        kernel weights for Approximate wavelet coefficients
        self.weights_ : tensor, shape-[in_channels * out_channels * x * y * z]
                        kernel weights for Detailed wavelet coefficients 
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list):
            if len(size) != 3:
                raise Exception('size: WaveConv2dCwt accepts the size of 3D signal in list with 3 elements')
            else:
                self.size = size
        else:
            raise Exception('size: WaveConv2dCwt accepts size of 3D signal is list')
        self.wavelet = wavelet
        self.mode = mode
        dummy_data = torch.randn( [*self.size] ).unsqueeze(0)
        mode_data = wavedec3(dummy_data, pywt.Wavelet(self.wavelet), level=self.level, mode=self.mode)
        self.modes1 = mode_data[0].shape[-3]
        self.modes2 = mode_data[0].shape[-2]
        self.modes3 = mode_data[0].shape[-1]
        self.omega = omega
        self.effective_modes_x = self.modes1//self.omega+1
        self.effective_modes_y = self.modes2//self.omega+1
        self.effective_modes_z = self.modes3//self.omega+1

        self.scale = (1 / (in_channels * out_channels))
        self.weights_a1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_a2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_a3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_a4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_aad1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_aad2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_aad3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_aad4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_ada1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_ada2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_ada3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_ada4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_add1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_add2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_add3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_add4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_daa1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_daa2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_daa3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_daa4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_dad1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_dad2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_dad3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_dad4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_dda1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_dda2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_dda3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_dda4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_ddd1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_ddd2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_ddd3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))
        self.weights_ddd4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, self.effective_modes_z, dtype=torch.cfloat))

    # Convolution
    def mul3d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(in_channel * x * y * z)
                  3D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y * z)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(out_channel * x * y * z)
        """
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
    
    # Spectral Convolution
    def spectralconv(self, waves, weights1, weights2, weights3, weights4):
        """
        Performs spectral convolution using Fourier decomposition

        Input Parameters
        ----------
        waves : tensor, shape-[Batch * Channel * size(x)]
                signal to be convolved, here the wavelet coefficients.
        weights : tensor, shape-[in_channel * out_channel * size(x)]
                The weights/kernel of the neural network.

        Returns
        -------
        convolved signal : tensor, shape-[batch * out_channel * size(x)]

        """
        # Get the frequency componenets
        xw = torch.fft.rfftn(waves, dim=[-3,-2,-1])
        
        # Initialize the output
        conv_out = torch.zeros(waves.shape[0], self.out_channels, waves.shape[-3], waves.shape[-2], waves.shape[-1]//2+1, dtype=torch.cfloat, device=waves.device)
        
        # Perform Element-wise multiplication in spectral doamin
        conv_out[:,:,:self.effective_modes_x,:self.effective_modes_y,:self.effective_modes_z] = self.mul3d(xw[:,:,:self.effective_modes_x,:self.effective_modes_y,:self.effective_modes_z], weights1)
        conv_out[:,:,-self.effective_modes_x:,:self.effective_modes_y,:self.effective_modes_z] = self.mul3d(xw[:,:,-self.effective_modes_x:,:self.effective_modes_y,:self.effective_modes_z], weights2)
        conv_out[:,:,:self.effective_modes_x,-self.effective_modes_y:,:self.effective_modes_z] = self.mul3d(xw[:,:,:self.effective_modes_x,-self.effective_modes_y:,:self.effective_modes_z], weights3)
        conv_out[:,:,-self.effective_modes_x:,-self.effective_modes_y:,:self.effective_modes_z] = self.mul3d(xw[:,:,-self.effective_modes_x:,-self.effective_modes_y:,:self.effective_modes_z], weights4)
        return torch.fft.irfftn(conv_out, s=(waves.shape[-3], waves.shape[-2], waves.shape[-1]))
    
    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x * y * z]
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x * y * z]
        """ 
        if x.shape[-1] > self.size[-1]:
            factor = int(np.log2(x.shape[-1] // self.size[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            x_coeff = wavedec3(x, pywt.Wavelet(self.wavelet), level=self.level+factor, mode=self.mode)
        
        elif x.shape[-1] < self.size[-1]:
            factor = int(np.log2(self.size[-1] // x.shape[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            x_coeff = wavedec3(x, pywt.Wavelet(self.wavelet), level=self.level-factor, mode=self.mode)        
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            x_coeff = wavedec3(x, pywt.Wavelet(self.wavelet), level=self.level, mode=self.mode)
        
        # Convert tuple to list
        x_coeff = list(x_coeff)
        
        # Multiply relevant Wavelet modes
        x_coeff[0] = self.spectralconv(x_coeff[0].clone(), self.weights_a1, self.weights_a2, self.weights_a3, self.weights_a4)
        x_coeff[1]['aad'] = self.spectralconv(x_coeff[1]['aad'].clone(), self.weights_aad1, self.weights_aad2, self.weights_aad3, self.weights_aad4)
        x_coeff[1]['ada'] = self.spectralconv(x_coeff[1]['ada'].clone(), self.weights_ada1, self.weights_ada2, self.weights_ada3, self.weights_ada4)
        x_coeff[1]['add'] = self.spectralconv(x_coeff[1]['add'].clone(), self.weights_add1, self.weights_add2, self.weights_add3, self.weights_add4)
        x_coeff[1]['daa'] = self.spectralconv(x_coeff[1]['daa'].clone(), self.weights_daa1, self.weights_daa2, self.weights_daa3, self.weights_daa4)
        x_coeff[1]['dad'] = self.spectralconv(x_coeff[1]['dad'].clone(), self.weights_dad1, self.weights_dad2, self.weights_dad3, self.weights_dad4)
        x_coeff[1]['dda'] = self.spectralconv(x_coeff[1]['dda'].clone(), self.weights_dda1, self.weights_dda2, self.weights_dda3, self.weights_dda4)
        x_coeff[1]['ddd'] = self.spectralconv(x_coeff[1]['ddd'].clone(), self.weights_ddd1, self.weights_ddd2, self.weights_ddd3, self.weights_ddd4)
            
        # Instantiate higher level coefficients as zeros
        for jj in range(2, self.level + 1):
            x_coeff[jj] = {key: torch.zeros([*x_coeff[jj][key].shape], device=x.device)
                            for key in x_coeff[jj].keys()}
        
        # Convert back to the tuple
        x_coeff = tuple(x_coeff)
        
        # Return to physical space        
        x = waverec3(x_coeff, pywt.Wavelet(self.wavelet))
        return x


class WNO3d(nn.Module):
    def __init__(self, T_in, T_out, width, level, layers, size, wavelet, grid_range, omega, padding=6):
        super(WNO3d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 4-channel tensor, Input at t0 and location (a(x,y,t), t, x, y)
              : shape: (batchsize * t=time * x=width * x=height * c=4)
        Output: Solution of a later timestep (u(x, T_in+1))
              : shape: (batchsize * t=time * x=width * x=height * c=1)
        
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : list with 3 elements (for 3D), the 3D volume size
        wavelet   : string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: list with 3 elements (for 3D), right supports of the 3D domain
        padding   : scalar, size of zero padding
        """
        self.T_in=T_in
        self.T_out=T_out
        self.level = level
        self.width = width
        self.size = size
        self.wavelet = wavelet
        self.omega = omega
        self.layers = layers
        self.grid_range = grid_range 
        self.padding = padding
                
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(self.T_in + 4, self.width) # input channel is 3: (a(x, y), x, y)
        for i in range(self.layers):
            self.conv.append(WaveConv3d(self.width, self.width, self.level, size=self.size, 
                                        wavelet=self.wavelet, omega=self.omega))
            self.w.append(nn.Conv3d(self.width, self.width, 1))

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, p):
        p = p.unsqueeze(1).repeat(1, self.T_out, 1, 1).unsqueeze(-1)
        x = x.unsqueeze(1).repeat(1, self.T_out, 1, 1, 1) # [nb, T_out, nx, nx, T_in]

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid, p), dim=-1)
        x = self.fc0(x)                 # Shape: Batch * x * y * z * Channel
        x = x.permute(0, 4, 3, 1, 2)    # Shape: Batch * Channel * z * x * y 
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding, 0,self.padding]) # do padding, if required
        

        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.mish(x)                # Shape: Batch * Channel * x * y
            
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding, :-self.padding] # remove padding, when required
        x = x.permute(0, 3, 4, 2, 1)        # Shape: Batch * x * y * z * Channel 
        x = self.fc2(F.mish(self.fc1(x)))   # Shape: Batch * x * y * z 
        return x.squeeze(-1).permute(0, 2, 3, 1)
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, self.grid_range[2], size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
