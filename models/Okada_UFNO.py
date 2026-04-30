from typing import Tuple, Optional, Literal
import torch
import torch.nn as nn

# ----------------------------- Fourier pieces -----------------------------
class SpectralConv2d(nn.Module):
    """
    2D Fourier layer: FFT -> multiply low-frequency modes with learned complex weights -> iFFT
    modes_x/modes_y: number of kept modes along each dimension.
    width: channel dimension of the feature maps.
    """
    def __init__(self, width: int, modes_x: int, modes_y: int):
        super().__init__()
        self.width = width
        self.modes_x = modes_x
        self.modes_y = modes_y
        # Complex weights as two real tensors
        self.weight_real = nn.Parameter(torch.randn(width, width, modes_x, modes_y) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(width, width, modes_x, modes_y) * 0.02)

    def complex_mul2d(self, a, b_real, b_imag):
        # a: [B, C_in, Mx, My] complex; b_*: [C_out, C_in, Mx, My] real
        a_real, a_imag = a.real, a.imag
        out_real = torch.einsum("bcmn,ocmn->bomn", a_real, b_real) - torch.einsum("bcmn,ocmn->bomn", a_imag, b_imag)
        out_imag = torch.einsum("bcmn,ocmn->bomn", a_real, b_imag) + torch.einsum("bcmn,ocmn->bomn", a_imag, b_real)
        return torch.complex(out_real, out_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] -> returns: [B, C, H, W]
        """
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, s=(H, W), norm="ortho")  # [B, C, H, W//2+1] complex
        out_ft = torch.zeros(B, C, H, W // 2 + 1, dtype=torch.complex64, device=x.device)

        mx = min(self.modes_x, H)
        my = min(self.modes_y, W // 2 + 1)

        x_ft_low = x_ft[:, :, :mx, :my]                            # [B,C,mx,my]
        w_real = self.weight_real[:, :, :mx, :my]                  # [C,C,mx,my]
        w_imag = self.weight_imag[:, :, :mx, :my]
        out_ft[:, :, :mx, :my] = self.complex_mul2d(x_ft_low, w_real, w_imag)
        y = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return y

class FiLM(nn.Module):
    """
    Per-layer FiLM conditioning: from parameters -> (gamma, beta) in R^{C}
    Applies y = (1+gamma) * x + beta (broadcast over H,W) for safe initialization.
    """
    def __init__(self, in_dim: int, out_channels: int, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, 2 * out_channels)]
        self.net = nn.Sequential(*layers)

    def forward(self, params: torch.Tensor):
        # params: [B, in_dim]
        gb = self.net(params)  # [B, 2C]
        B = gb.shape[0]
        gamma, beta = gb.chunk(2, dim=-1)  # [B,C], [B,C]
        return gamma.view(B, -1, 1, 1), beta.view(B, -1, 1, 1)

class FNOBlock(nn.Module):
    """
    One FNO layer:
      y = SpectralConv(x) + 1x1 Conv(x)
      y = FiLM(gamma,beta) on y
      y = activation(y) -> dropout
    """
    def __init__(self, width: int, modes_x: int, modes_y: int,
                 film_in: int, film_hidden: int = 64, film_layers: int = 2,
                 dropout_p: float = 0.0):
        super().__init__()
        self.spectral = SpectralConv2d(width, modes_x, modes_y)
        self.w = nn.Conv2d(width, width, kernel_size=1)
        self.film = FiLM(film_in, width, hidden=film_hidden, n_layers=film_layers)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

        nn.init.kaiming_normal_(self.w.weight, nonlinearity='relu')
        if self.w.bias is not None:
            nn.init.zeros_(self.w.bias)

    def forward(self, x, params):
        y = self.spectral(x) + self.w(x)
        gamma, beta = self.film(params)  # [B,C,1,1] each
        y = y * (1 + gamma) + beta
        y = self.act(y)
        y = self.drop(y)
        return y

# ----------------------------- U-Net refiner ------------------------------
class UNetConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GELU(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x): return self.net(x)

class UNetRefiner(nn.Module):
    """
    A small 3-level U-Net:
      in_ch  -> enc( base -> 2*base -> 4*base )
              -> dec with skip connections
              -> out_channels (predicts Δy)
    Uses bilinear upsample to avoid checkerboards.
    """
    def __init__(self, in_ch: int, out_ch: int, base: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = UNetConv(in_ch, base)          # H,W
        self.enc2 = UNetConv(base, base*2)         # H/2,W/2
        self.enc3 = UNetConv(base*2, base*4)       # H/4,W/4
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = UNetConv(base*4 + base*2, base*2)
        self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = UNetConv(base*2 + base, base)

        self.head = nn.Conv2d(base, out_ch, kernel_size=1)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity='relu')
        if self.head.bias is not None: nn.init.zeros_(self.head.bias)

    def forward(self, x):  # x: [B,in_ch,H,W]
        e1 = self.enc1(x)                  # [B,base,H,W]
        e2 = self.enc2(self.pool(e1))      # [B,2b,H/2,W/2]
        e3 = self.enc3(self.pool(e2))      # [B,4b,H/4,W/4]

        d2 = self.up2(e3)                  # -> H/2,W/2
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)                 # [B,2b,H/2,W/2]

        d1 = self.up1(d2)                  # -> H,W
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)                 # [B,base,H,W]

        return self.head(d1)               # Δy

# ----------------------- Dual-mode main model -----------------------------
class Param2FieldDecoder128(nn.Module):
    """
    Dual-mode model with FNO trunk + U-Net residual refiner AND inverse encoder.

    Forward behavior:
      - If input x has shape [B, n_parameters] (2D): treats x as parameters and returns field [B, outC, H, W].
      - If input x has shape [B, 1, H, W] (4D): treats x as field and returns parameters [B, n_parameters].

    Args:
      n_parameters: size of parameter vector
      target_hw: (H, W) output/input spatial resolution for the field
      base_ch: width (channels) of the FNO trunk (also used as default for inverse)
      n_layers: number of FNO blocks
      modes_x/modes_y: kept Fourier modes per dimension
      param_embed_dim: channels for broadcast param embedding
      unet_base: base channels for the U-Net refiner
      block_dropout_p: dropout prob for spectral blocks
      ... other args kept for signature compatibility; ignored where noted
    """
    def __init__(self, n_parameters: int,
                 start_hw: Tuple[int,int]=(4,4),   # ignored for FNO path
                 n_stages: int=5,                  # ignored
                 base_ch: int=32,                  # width for spectral trunk
                 ch_growth: float=1.5,             # ignored
                 n_convs_per_stage: int=1,         # ignored
                 up_method="bilinear",             # ignored
                 norm="in", act="gelu", residual=True,  # ignored here
                 out_channels: int=1,
                 target_hw: Tuple[int,int]=(100,100),
                 proj_dropout_p: float = 0.0,
                 block_dropout_p: float = 0.0,
                 # FNO (params->field) knobs
                 width: Optional[int] = None,
                 n_layers: int = 4,
                 modes_x: int = 8,
                 modes_y: int = 8,
                 param_embed_dim: int = 8,
                 film_hidden: int = 128,
                 film_layers: int = 2,
                 # U-Net head
                 unet_base: int = 32,
                 # Inverse (field->params) encoder knobs
                 inv_width: Optional[int] = None,
                 inv_layers: int = 3,
                 inv_modes_x: Optional[int] = None,
                 inv_modes_y: Optional[int] = None,
                 inv_mlp_hidden: int = 128):
        super().__init__()
        self.H, self.W = target_hw
        self.n_parameters = n_parameters
        self.width = width or base_ch
        self.param_embed_dim = param_embed_dim

        # ---------- Shared: cached coord grid ----------
        self._cached_grid = None

        # ---------- PARAMS -> FIELD (FNO trunk + U-Net refiner) ----------
        in_ch = 2 + 1 + param_embed_dim  # [x,y,1,param_embed]
        self.param_embed = nn.Linear(n_parameters, param_embed_dim)
        self.input_lift = nn.Conv2d(in_ch, self.width, kernel_size=1)

        # FNO blocks with FiLM conditioning
        self.blocks = nn.ModuleList([
            FNOBlock(
                width=self.width, modes_x=modes_x, modes_y=modes_y,
                film_in=n_parameters, film_hidden=film_hidden, film_layers=film_layers,
                dropout_p=block_dropout_p
            ) for _ in range(n_layers)
        ])

        # First-pass field head and U-Net residual refiner
        self.fno_head = nn.Conv2d(self.width, out_channels, kernel_size=1)
        unet_in_ch = self.width + 3 + param_embed_dim + out_channels  # feats + coords + p_map + y_fno
        self.unet = UNetRefiner(in_ch=unet_in_ch, out_ch=out_channels, base=unet_base)

        # ---------- FIELD -> PARAMS (inverse encoder) ----------
        inv_width_val = inv_width or self.width
        inv_modes_x = inv_modes_x or modes_x
        inv_modes_y = inv_modes_y or modes_y

        # Stem consumes [field, x, y, 1]
        self.inv_stem = nn.Sequential(
            nn.Conv2d(1 + 3, inv_width_val, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(inv_width_val, inv_width_val, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # Spectral residual blocks (no FiLM for inverse)
        self.inv_spec = nn.ModuleList([
            nn.Sequential(
                SpectralConv2d(inv_width_val, inv_modes_x, inv_modes_y),
                nn.Conv2d(inv_width_val, inv_width_val, kernel_size=1),
                nn.GELU(),
                nn.Dropout2d(p=block_dropout_p) if block_dropout_p > 0 else nn.Identity()
            ) for _ in range(inv_layers)
        ])
        # Global average pool -> MLP to parameters
        self.inv_head = nn.Sequential(
            nn.Linear(inv_width_val, inv_mlp_hidden),
            nn.GELU(),
            nn.Linear(inv_mlp_hidden, n_parameters),
        )

        # ---------- Inits ----------
        nn.init.kaiming_normal_(self.input_lift.weight, nonlinearity='relu')
        if self.input_lift.bias is not None: nn.init.zeros_(self.input_lift.bias)
        nn.init.kaiming_normal_(self.param_embed.weight, nonlinearity='relu')
        if self.param_embed.bias is not None: nn.init.zeros_(self.param_embed.bias)
        nn.init.kaiming_normal_(self.fno_head.weight, nonlinearity='relu')
        if self.fno_head.bias is not None: nn.init.zeros_(self.fno_head.bias)

        for m in self.inv_stem:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
        for blk in self.inv_spec:
            for m in blk:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None: nn.init.zeros_(m.bias)
        for m in self.inv_head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    # --------------------- helpers ---------------------
    def _coord_grid(self, B, device, dtype):
        key = (self.H, self.W, device, dtype)
        if self._cached_grid is not None and self._cached_grid[0] == key:
            return self._cached_grid[1].expand(B, -1, -1, -1)
        xs = torch.linspace(-1.0, 1.0, self.W, device=device, dtype=dtype)
        ys = torch.linspace(-1.0, 1.0, self.H, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        ones = torch.ones_like(xx)
        grid = torch.stack([xx, yy, ones], dim=0).unsqueeze(0)  # [1,3,H,W]
        self._cached_grid = (key, grid)
        return grid.expand(B, -1, -1, -1)

    # ---------------------- forward --------------------
    def forward(self, x: torch.Tensor, add_input_noise_std: float = 0.0) -> torch.Tensor:
        """
        - If x.ndim == 2: treat x as parameters [B, n_parameters] and output field [B, outC, H, W].
        - If x.ndim == 4: treat x as field     [B, 1, H, W]     and output parameters [B, n_parameters].
        """
        if x.ndim == 2:
            # ---------------- PARAMS -> FIELD ----------------
            if self.training and add_input_noise_std > 0:
                x = x + add_input_noise_std * torch.randn_like(x)

            B = x.size(0)
            grid = self._coord_grid(B, device=x.device, dtype=x.dtype)             # [B,3,H,W]
            p_emb = self.param_embed(x).unsqueeze(-1).unsqueeze(-1)                # [B,P,1,1]
            p_map = p_emb.expand(-1, -1, self.H, self.W)                           # [B,P,H,W]
            inp = torch.cat([grid, p_map], dim=1)                                  # [B,3+P,H,W]
            feats = self.input_lift(inp)                                           # [B,width,H,W]

            for blk in self.blocks:
                feats = blk(feats, x)                                              # FiLM conditioning

            y_fno = self.fno_head(feats)                                           # [B,outC,H,W]
            unet_in = torch.cat([feats, grid, p_map, y_fno], dim=1)
            delta = self.unet(unet_in)
            return y_fno + delta                                                   # [B,outC,H,W]

        elif x.ndim == 4:
            # ---------------- FIELD -> PARAMS ----------------
            # Expect x: [B, 1, H, W] (normalized field). Append coords.
            B, C, H, W = x.shape
            assert C == 1 and H == self.H and W == self.W, \
                f"Expected field [B,1,{self.H},{self.W}], got {list(x.shape)}"
            grid = self._coord_grid(B, device=x.device, dtype=x.dtype)             # [B,3,H,W]
            z = torch.cat([x, grid], dim=1)                                        # [B,1+3,H,W]
            z = self.inv_stem(z)                                                   # [B,inv_width,H,W]
            for blk in self.inv_spec:
                spec = blk[0](z)                                                   # spectral
                res  = blk[1](z)                                                   # 1x1 conv
                z = blk[2](spec + res)                                             # GELU
                z = blk[3](z)                                                      # Dropout (maybe)
            z = z.mean(dim=(2, 3))                                                 # GAP -> [B,inv_width]
            return self.inv_head(z)                                                # [B, n_parameters]

        else:
            raise ValueError("Input rank not supported. Pass [B,n_parameters] or [B,1,H,W].")

