import torch
import torch.nn as nn
from einops import rearrange


class TemporalTransformer(nn.Module):
    def __init__(self, dim, depth=1, heads=4, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=inner_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=depth
        )

    def forward(self, x):
        # x: (B, F, C, H, W) -> (B*H*W, F, C)
        B, F, C, H, W = x.shape
        x = rearrange(x, 'b f c h w -> b h w f c')
        x = x.reshape(B * H * W, F, C)
        x = self.attn(self.norm(x))
        x = x.reshape(B, H, W, F, C).permute(0, 3, 4, 1, 2)  # (B, F, C, H, W)
        return x


class STCEncoder(nn.Module):
    def __init__(self, in_channels=3, feature_dim=64, transformer_depth=1):
        super().__init__()
        self.spatial_net = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.SiLU()
        )
        self.temporal_transformer = TemporalTransformer(
            dim=feature_dim,
            depth=transformer_depth,
            heads=4,
            dim_head=feature_dim // 4
        )

    def forward(self, video):
        # video: (B, C, F, H, W)
        B, C, F, H, W = video.shape
        video = rearrange(video, 'b c f h w -> (b f) c h w')  # treat each frame
        features = self.spatial_net(video)  # (B*F, C', H', W')
        C_, H_, W_ = features.shape[1:]  # updated dims
        features = features.view(B, F, C_, H_, W_)  # (B, F, C', H', W')
        features = self.temporal_transformer(features)  # STC encoded features
        return features  # (B, F, C', H', W')
