import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class NoiseShaper(nn.Module):
    def __init__(self, in_channels=64, beta=0.7):
        super().__init__()
        self.beta = beta
        self.flow_predictor = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)  # (dx, dy)

    def forward(self, stc_feat, noise_first_frame, noise_ind):
        """
        Args:
            stc_feat: (B, T, C_feat, Hs, Ws) from STC encoder
            noise_first_frame: (B, C, H, W) - shared base noise
            noise_ind: (B, C, T, H, W) - per-frame independent noise
        Returns:
            Shaped noise tensor: (B, C, T, H, W)
        """
        B, T, C_feat, Hs, Ws = stc_feat.shape
        _, C, H, W = noise_first_frame.shape

        # Predict flow from STC features
        stc_feat_flat = rearrange(stc_feat, 'b t c h w -> (b t) c h w')
        flow = self.flow_predictor(stc_feat_flat)  # (B*T, 2, Hs, Ws)
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        flow = rearrange(flow, '(b t) c h w -> b t c h w', b=B, t=T)

        # Build normalized coordinate grid for warping
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=flow.device),
            torch.linspace(-1, 1, W, device=flow.device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
        grid = grid.unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1, 1)  # (B, T, H, W, 2)

        # Normalize flow to [-1, 1]
        flow_norm = flow.permute(0, 1, 3, 4, 2)  # (B, T, H, W, 2)
        flow_norm[..., 0] /= W / 2
        flow_norm[..., 1] /= H / 2
        sampling_grid = grid + flow_norm  # (B, T, H, W, 2)

        # Warp base noise
        noise_base = repeat(noise_first_frame, 'b c h w -> b t c h w', t=T)
        noise_base = rearrange(noise_base, 'b t c h w -> (b t) c h w')
        sampling_grid = rearrange(sampling_grid, 'b t h w c -> (b t) h w c')
        warped_noise = F.grid_sample(noise_base, sampling_grid, align_corners=False, mode='bilinear', padding_mode='border')
        warped_noise = rearrange(warped_noise, '(b t) c h w -> b c t h w', b=B, t=T)

        # Per-pixel normalization across channel (approximate 1/sqrt(|Omega_p|) from NTE)
        warped_mean = warped_noise.mean(dim=1, keepdim=True)  # (B, 1, T, H, W)
        warped_std = warped_noise.std(dim=1, keepdim=True)    # (B, 1, T, H, W)
        warped_noise = (warped_noise - warped_mean) / (warped_std + 1e-6)

        # Combine with independent noise
        eps = self.beta * warped_noise + torch.sqrt(torch.tensor(1.0 - self.beta**2, device=warped_noise.device)) * noise_ind

        # Optional: normalize total eps to std = 1 (global fix)
        eps_flat = eps.reshape(B, -1)
        mean = eps_flat.mean(dim=1, keepdim=True).unsqueeze(-1)
        std = eps_flat.std(dim=1, keepdim=True).unsqueeze(-1)
        eps = (eps - mean) / (std + 1e-6)

        return eps
