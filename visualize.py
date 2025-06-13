import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
from einops import rearrange, repeat
import os

class NoiseShaper(nn.Module):
    def __init__(self, in_channels=64, beta=0.999):
        super().__init__()
        self.beta = beta
        self.flow_predictor = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)

    def forward(self, stc_feat, noise_first_frame, noise_ind):
        B, T, C_feat, Hs, Ws = stc_feat.shape
        _, C, H, W = noise_first_frame.shape

        stc_feat_flat = rearrange(stc_feat, 'b t c h w -> (b t) c h w')
        flow = self.flow_predictor(stc_feat_flat)
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        flow = rearrange(flow, '(b t) c h w -> b t c h w', b=B, t=T)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1, 1)
        flow_norm = flow.permute(0, 1, 3, 4, 2)
        flow_norm[..., 0] /= W / 2
        flow_norm[..., 1] /= H / 2
        sampling_grid = grid + flow_norm

        noise_base = repeat(noise_first_frame, 'b c h w -> b t c h w', t=T)
        noise_base = rearrange(noise_base, 'b t c h w -> (b t) c h w')
        sampling_grid = rearrange(sampling_grid, 'b t h w c -> (b t) h w c')
        warped_noise = F.grid_sample(noise_base, sampling_grid, align_corners=False, mode='bilinear', padding_mode='border')
        warped_noise = rearrange(warped_noise, '(b t) c h w -> b c t h w', b=B, t=T)

        warped_mean = warped_noise.mean(dim=1, keepdim=True)
        warped_std = warped_noise.std(dim=1, keepdim=True)
        warped_noise = (warped_noise - warped_mean) / (warped_std + 1e-6)

        eps = self.beta * warped_noise + torch.sqrt(torch.tensor(1.0 - self.beta**2)) * noise_ind

        eps_flat = eps.reshape(B, -1)
        mean = eps_flat.mean(dim=1, keepdim=True).unsqueeze(-1)
        std = eps_flat.std(dim=1, keepdim=True).unsqueeze(-1)
        eps = (eps - mean) / (std + 1e-6)

        return eps

def load_video_frames(video_path, start_frame=0, end_frame=4, size=(48, 48)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    # Clamp values to valid range
    start_frame = max(0, start_frame)
    end_frame = min(end_frame, total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = T.ToTensor()(frame)
        frames.append(tensor)
    cap.release()

    video_tensor = torch.stack(frames, dim=1)  # (C, T, H, W)
    return video_tensor.unsqueeze(0)  # (1, C, T, H, W)


def visualize_latents(z_tensor, title, save_path):
    B, C, T, H, W = z_tensor.shape
    z_mean = z_tensor[0].mean(dim=0)  # (T, H, W)

    fig, axs = plt.subplots(T, 1, figsize=(6, T*2))
    for t in range(T):
        axs[t].imshow(z_mean[t].cpu().detach().numpy(), cmap='plasma')
        axs[t].axis('off')
        axs[t].set_title(f"{title} - Frame {t}")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    
def visualize_and_save_each_frame(z_tensor, prefix, output_dir="heatmaps"):
    """
    Save each frame of z_tensor (mean over channels) as a heatmap image.
    Args:
        z_tensor: (B, C, T, H, W)
        prefix: one of "z0", "zt_gaussian", "zt_shaped"
        output_dir: folder to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    B, C, T, H, W = z_tensor.shape
    z_mean = z_tensor[0].mean(dim=0)  # (T, H, W)

    for t in range(T):
        frame = z_mean[t].cpu().detach().numpy()
        plt.figure(figsize=(H / 20, W / 20), dpi=100)
        plt.imshow(frame, cmap='plasma')
        plt.axis('off')
        plt.tight_layout(pad=0)
        fname = os.path.join(output_dir, f"{prefix}_{t}.png")
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {fname}")
        
def main():
    video_path = '/media/hdd1/ngoc/NAS/Original_vid_2s/Train/Class_D/BasketballPass_416x240_50.mp4'  # Replace with your video path
    start_frame = 1           # <<< your desired start frame
    end_frame = 5                # <<< your desired end frame

    video_tensor = load_video_frames(video_path, start_frame=start_frame, end_frame=end_frame)
    B, C, T, H, W = video_tensor.shape

    z0 = video_tensor
    eps_gaussian = torch.randn_like(z0)
    zt = z0 + eps_gaussian

    noise_base = torch.randn(B, C, H, W)
    stc_feat = torch.randn(B, T, 64, H//4, W//4)

    noise_shaper = NoiseShaper()
    eps_shaped = noise_shaper(stc_feat, noise_base, eps_gaussian)
    zt_prime = z0 + eps_shaped

    #visualize_and_save_each_frame(z0, "z0")
    #visualize_and_save_each_frame(zt, "zt_gaussian")
    visualize_and_save_each_frame(zt_prime, "zt_shaped_10")

if __name__ == "__main__":
    main()

