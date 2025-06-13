import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.cuda.amp import autocast, GradScaler
from diffusers import DDIMScheduler

from models.autoencoder import AutoencoderKL
from models.unet_sd import UNetSD_temporal
from models.stc_encoder import STCEncoder
from models.noise_shaper import NoiseShaper
from models.text_encoder import TextEncoder
from utils.latent_guard import latent_guard, latent_penalty
from utils.load_unet_pretrained import load_stable_diffusion_pretrained

def train(cfg, dataset):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()

    scale_factor = 0.18215

    autoencoder = AutoencoderKL(ddconfig=cfg.ddconfig, embed_dim=cfg.embed_dim, ckpt_path=cfg.vae_ckpt).to(device).eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    text_encoder = TextEncoder(model_name=cfg.text_encoder.model, device=cfg.text_encoder.device)
    stc_encoder = STCEncoder().to(device)
    noise_shaper = NoiseShaper(in_channels=64, beta=0.9).to(device)

    unet = UNetSD_temporal(
        cfg=cfg,
        in_dim=4,
        dim=320,
        y_dim=512,
        context_dim=1024,
        out_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_heads=8,
        head_dim=64,
        num_res_blocks=2,
        attn_scales=[1/2, 1/4, 1/8],
        dropout=0.0,
        temporal_attention=False,
        temporal_attn_times=1,
        use_checkpoint=False,
        use_image_dataset=False,
        use_fps_condition=False,
        use_sim_mask=False,
        video_compositions=['text'],
        p_all_zero=0.0,
        p_all_keep=1.0,
        zero_y=torch.zeros(1, 77, 1024).to(device),
        black_image_feature=torch.zeros(1, 1, 1024).to(device)
    ).to(device)

    sd_raw = torch.load(cfg.unet_ckpt, map_location="cpu")
    sd = sd_raw["state_dict"] if "state_dict" in sd_raw else sd_raw
    sd = load_stable_diffusion_pretrained(sd, temporal_attention=False)
    unet.load_state_dict(sd, strict=False)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False

    for p in stc_encoder.parameters():
        p.requires_grad = True
    for p in noise_shaper.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(list(stc_encoder.parameters()), lr=cfg.lr)

    scheduler = DDIMScheduler(
    num_train_timesteps=cfg.timesteps,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    prediction_type="epsilon"  # For noise prediction
    )
    scheduler.set_timesteps(cfg.timesteps)

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    for epoch in range(cfg.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch in pbar:
            video = batch['video'].to(device)
            mask = batch['mask'].to(device)
            captions = batch['caption']
            text_embed = text_encoder(captions).to(device)

            if video.shape[2] > 30:
                video = video[:, :, :30]
                mask = mask[:, :, :30]

            stc_feat = stc_encoder(video)

            with torch.no_grad():
                B, _, T, H, W = video.shape
                video_flat = video.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)
                chunks = torch.chunk(video_flat, chunks=64, dim=0)
                z_list = [autoencoder.encode(chunk).sample() * scale_factor for chunk in chunks]
                z_flat = torch.cat(z_list, dim=0)
                z_0 = z_flat.view(B, T, -1, H // 8, W // 8).permute(0, 2, 1, 3, 4)
                z_0 = latent_guard(z_0, name="z_0", normalize=True)
                _, C_latent, _, H_z, W_z = z_0.shape
                noise_first = torch.randn(B, C_latent, H_z, W_z, device=device)
                noise_ind = torch.randn_like(z_0)

            noise_shaped = noise_shaper(stc_feat, noise_first, noise_ind)

            t = torch.randint(0, cfg.timesteps, (B,), device=device).long()
            z_t = scheduler.add_noise(z_0, noise_shaped, t)

            with autocast():
                pred_noise = unet(z_t, t, stc_feat=stc_feat)
                loss = nn.MSELoss()(pred_noise, noise_shaped)
                loss += 0.01 * latent_penalty(z_0)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=loss.item())
            torch.cuda.empty_cache()

        if (epoch + 1) % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'stc_encoder': stc_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    print("Training finished.")
