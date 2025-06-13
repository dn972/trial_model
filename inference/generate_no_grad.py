from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from models.autoencoder import AutoencoderKL
from models.unet_sd import UNetSD_temporal
from models.stc_encoder import STCEncoder
from models.noise_shaper import NoiseShaper
from models.text_encoder import TextEncoder
from utils.load_unet_pretrained import load_stable_diffusion_pretrained

import matplotlib.pyplot as plt
import os

def load_video(path, resize, max_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and len(frames) >= max_frames):
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    frames = np.stack(frames).astype(np.float32) / 127.5 - 1
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)
    return frames

def load_caption(caption_file, video_path):
    path = Path(video_path)
    class_name = path.parent.name
    raw_name = path.stem.split("_")[0]
    key = f"{class_name}/{raw_name}"
    with open(caption_file, 'r') as f:
        for line in f:
            if line.startswith(key + "|"):
                return line.strip().split("|", 1)[-1]
    return ""

def chunk_video_tensor(video, chunk_size=30):
    B, C, T, H, W = video.shape
    chunks = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = video[:, :, start:end]
        actual_len = end - start
        if actual_len < chunk_size:
            pad = chunk_size - actual_len
            chunk = F.pad(chunk, (0, 0, 0, 0, 0, pad))
        chunks.append((chunk, actual_len))
    return chunks

def generate(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resize = tuple(cfg.resize)
    
    # Load only 1 frame from the video
    degraded = load_video(cfg.input_video, resize, max_frames=1).to(device)
    caption = load_caption(cfg.caption_file, cfg.input_video)
    print("Caption:", caption)

    text_encoder = TextEncoder(cfg.text_encoder.model, cfg.text_encoder.device)
    text_embed = text_encoder([caption]).to(device)
    if text_embed.dim() == 2:
        text_embed = text_embed.unsqueeze(1)
    if text_embed.shape[-1] < 1024:
        text_embed = F.pad(text_embed, (0, 1024 - text_embed.shape[-1]))

    scale_factor = 0.18215
    autoencoder = AutoencoderKL(ddconfig=cfg.ddconfig, embed_dim=cfg.embed_dim, ckpt_path=cfg.vae_ckpt).to(device).eval()
    stc_encoder = STCEncoder().to(device).eval()
    noise_shaper = NoiseShaper(in_channels=64, beta=0.9).to(device)

    unet = UNetSD_temporal(
        cfg=cfg, in_dim=4, dim=320, y_dim=512, context_dim=1024, out_dim=4,
        dim_mult=[1, 2, 4, 4], num_heads=8, head_dim=64, num_res_blocks=2,
        attn_scales=[1/2, 1/4, 1/8], dropout=0.0, temporal_attention=False,
        temporal_attn_times=1, use_checkpoint=False, use_image_dataset=False,
        use_fps_condition=False, use_sim_mask=False, video_compositions=["text"],
        p_all_zero=0.0, p_all_keep=1.0,
        zero_y=torch.zeros(1, 77, 1024).to(device),
        black_image_feature=torch.zeros(1, 1, 1024).to(device)
    ).to(device).eval()

    unet.load_state_dict(load_stable_diffusion_pretrained(torch.load(cfg.temporal_unet_ckpt, map_location="cpu"), temporal_attention=False), strict=False)
    stc_ckpt = torch.load(cfg.checkpoint, map_location=device)
    stc_encoder.load_state_dict(stc_ckpt["stc_encoder"])

    chunks = chunk_video_tensor(degraded, chunk_size=30)
    restored_video = []

    with torch.no_grad():
        scheduler = DDPMScheduler(num_train_timesteps=cfg.timesteps)
        scheduler.set_timesteps(cfg.timesteps, device=device)

        for i, (chunk, actual_len) in enumerate(tqdm(chunks, desc="Chunks")):
            B, C, T, H, W = chunk.shape

            stc_feat = stc_encoder(chunk)
            chunk_flat = chunk.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)
            z_flat = autoencoder.encode(chunk_flat).mode() * scale_factor
            del chunk_flat
            torch.cuda.empty_cache()

            z_0 = z_flat.view(B, T, -1, H // 8, W // 8).permute(0, 2, 1, 3, 4)
            del z_flat
            torch.cuda.empty_cache()

            noise_first = torch.randn(B, 4, H // 8, W // 8, device=device)
            noise_ind = torch.randn_like(z_0)
            #noise = noise_shaper(stc_feat, noise_first, noise_ind)
            noise = torch.randn_like(z_0)
            z_t = scheduler.add_noise(z_0, noise, scheduler.timesteps[0].repeat(B))
            del noise_first, noise_ind, noise
            torch.cuda.empty_cache()

            for t in scheduler.timesteps:
                t_tensor = torch.tensor([t.item()], device=device)
                noise_pred = unet(z_t, t_tensor, stc_feat=stc_feat, y=text_embed)
                z_t = scheduler.step(model_output=noise_pred, timestep=t, sample=z_t).prev_sample
                del noise_pred
                torch.cuda.empty_cache()

            z_final_flat = z_t.permute(0, 2, 1, 3, 4).reshape(B * T, 4, H // 8, W // 8)
            del z_t
            torch.cuda.empty_cache()

            recon_chunks = []
            for j in range(z_final_flat.size(0)):
                z_chunk = z_final_flat[j:j+1] / scale_factor
                with torch.cuda.amp.autocast():
                    x_chunk = autoencoder.decode(z_chunk)
                    if hasattr(x_chunk, "sample"):
                        x_chunk = x_chunk.sample
                recon_chunks.append(x_chunk.cpu())
                del z_chunk, x_chunk
                torch.cuda.empty_cache()

            x_cat = torch.cat(recon_chunks, dim=0)  # (T_decoded, 3, H, W)
            T_decoded = x_cat.size(0) // B
            x_hat = x_cat.view(B, T_decoded, 3, H, W).permute(0, 2, 1, 3, 4)[0]
            x_hat = x_hat[:, :actual_len]
            restored_video.append(x_hat)

        x_out = torch.cat(restored_video, dim=1)
        x_out = ((x_out + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 3, 0).numpy()

        out = cv2.VideoWriter(cfg.output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, resize)
        for frame in x_out:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print("Saved:", cfg.output_path)
