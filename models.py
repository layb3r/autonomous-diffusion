from torch import nn
import torch
from torch.nn import functional as F
from helper import DoubleConv, ResidualBlock, UpSampleBlock


class SmallUNet(nn.Module):
    def __init__(self, in_channels=1, base_dim=64, time_dim=128):
        super().__init__()
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        # FiLM adapters for each double conv block (scale and shift)
        self.film1 = nn.Linear(time_dim, base_dim * 2)
        self.film2 = nn.Linear(time_dim, base_dim * 4)
        self.film3 = nn.Linear(time_dim, (base_dim*4) * 2)
        self.film4 = nn.Linear(time_dim, (base_dim*4) * 2)
        self.film5 = nn.Linear(time_dim, base_dim * 4)
        self.film6 = nn.Linear(time_dim, base_dim * 2)

        # Encoder
        self.inc = DoubleConv(in_channels, base_dim) # 64
        self.down1 = nn.Conv2d(base_dim, base_dim*2, 3, stride=2, padding=1) # 128
        self.dc1 = DoubleConv(base_dim*2, base_dim*2) # 128
        self.down2 = nn.Conv2d(base_dim*2, base_dim*4, 3, stride=2, padding=1) # 256
        self.dc2 = DoubleConv(base_dim*4, base_dim*4) # 256

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_dim*4, base_dim*2, 4, stride=2, padding=1) # 128
        self.dc3 = DoubleConv(base_dim*4, base_dim*2)
        self.up2 = nn.ConvTranspose2d(base_dim*2, base_dim, 4, stride=2, padding=1) # 64
        self.dc4 = DoubleConv(base_dim*2, base_dim)
        self.outc = nn.Conv2d(base_dim, in_channels, 3, padding=1)

    def film(self, x, gamma, beta):
        return x * gamma + beta

    def forward(self, x, t):
        # time embedding and FiLM parameters
        if t.ndim == 1:
            t = t[:, None]
        temb = self.time_mlp(t)

        # Encoder
        x1 = self.inc(x) # 64
        g1, b1 = self.film1(temb).chunk(2, dim=1)
        x1 = self.film(x1, g1[:,:,None,None], b1[:,:,None,None])

        x2 = self.down1(x1) # 128
        x2 = self.dc1(x2)
        g2, b2 = self.film2(temb).chunk(2, dim=1)
        x2 = self.film(x2, g2[:,:,None,None], b2[:,:,None,None])

        x3 = self.down2(x2) # 256
        x3 = self.dc2(x3)
        g3, b3 = self.film3(temb).chunk(2, dim=1)
        x3 = self.film(x3, g3[:,:,None,None], b3[:,:,None,None])

        # Decoder
        x = self.up1(x3) # 128
        # Concatenate with skip connection x2 (also 128)
        x = torch.cat([x, x2], dim=1) # 256
        g4, b4 = self.film4(temb).chunk(2, dim=1)
        x = self.film(x, g4[:,:,None,None], b4[:,:,None,None])
        x = self.dc3(x) # 128

        x = self.up2(x) # 64
        x = torch.cat([x, x1], dim=1) # 128 (64+64)
        g5, b5 = self.film5(temb).chunk(2, dim=1)
        x = self.film(x, g5[:,:,None,None], b5[:,:,None,None])
        x = self.dc4(x) # 64

        x = self.outc(x)
        return x


class NoiseAgnosticSmallUNet(nn.Module):
    def __init__(self, in_channels=1, base_dim=64):
        super().__init__()

        self.inc = DoubleConv(in_channels, base_dim)
        self.down1 = nn.Conv2d(base_dim, base_dim * 2, 3, stride=2, padding=1)
        self.dc1 = DoubleConv(base_dim * 2, base_dim * 2)
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 4, 3, stride=2, padding=1)
        self.dc2 = DoubleConv(base_dim * 4, base_dim * 4)

        self.up1 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 4, stride=2, padding=1)
        self.dc3 = DoubleConv(base_dim * 4, base_dim * 2)
        self.up2 = nn.ConvTranspose2d(base_dim * 2, base_dim, 4, stride=2, padding=1)
        self.dc4 = DoubleConv(base_dim * 2, base_dim)
        self.outc = nn.Conv2d(base_dim, in_channels, 3, padding=1)

    def forward(self, x, t=None):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x2 = self.dc1(x2)

        x3 = self.down2(x2)
        x3 = self.dc2(x3)

        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.dc3(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dc4(x)

        return self.outc(x)
    
    
class TimeMLP(nn.Module):
    def __init__(self, data_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x, t):
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat((x, t), dim=1))


class TimeConvNet(nn.Module):
    def __init__(self, channels=1, base_dim=64, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        self.to_scale1 = nn.Linear(time_dim, base_dim)
        self.to_scale2 = nn.Linear(time_dim, base_dim)

        self.conv_in = nn.Conv2d(channels, base_dim, 3, padding=1)
        self.conv_mid = nn.Conv2d(base_dim, base_dim, 3, padding=1)
        self.conv_out = nn.Conv2d(base_dim, channels, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, base_dim)
        self.norm2 = nn.GroupNorm(8, base_dim)

    def forward(self, x, t):
        if t.ndim == 1:
            t = t[:, None]

        temb = self.time_mlp(t)
        s1 = self.to_scale1(temb)[:, :, None, None]
        s2 = self.to_scale2(temb)[:, :, None, None]

        h = self.conv_in(x)
        h = self.norm1(h)
        h = F.silu(h + s1)

        h = self.conv_mid(h)
        h = self.norm2(h)
        h = F.silu(h + s2)

        return self.conv_out(h)