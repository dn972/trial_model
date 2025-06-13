import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
from models.autoencoder import AutoencoderKL
from models.unet_sd import UNetSD_temporal as UNetSTC
from models.stc_encoder import STCEncoder
from models.noise_shaper import NoiseShaper
from models.text_encoder import TextEncoder
from utils.latent_guard import latent_guard
from utils.load_unet_pretrained import load_stable_diffusion_pretrained
from VVC_Python import VVC
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

def load_video(path, resize):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
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

def encode_like_latent(image_tensor, autoencoder, scale_factor):
    with torch.no_grad():
        z = autoencoder.encode(image_tensor).sample() * scale_factor
    return z

def grad_VVC(x_in, x_lr, maskBG, batchIdx, curStep):
    x_in = x_in[batchIdx]
    x_lr = x_lr[batchIdx]
    maskBG = maskBG[batchIdx]
    h = 16 / 255.0
    perturb = torch.ones_like(x_in).cuda()
    x_fw = torch.clamp(x_in + h * perturb, -1, 1)
    x_bw = torch.clamp(x_in - h * perturb, -1, 1)
    save_dir = f"./vvc_temp/batch_{batchIdx}"
    os.makedirs(save_dir, exist_ok=True)
    x_deg = VVC.vvc_func(((x_in + 1) / 2), curStep, save_dir)
    x_fw_deg = VVC.vvc_func_h(((x_fw + 1) / 2), curStep, save_dir)
    x_bw_deg = VVC.vvc_func_h(((x_bw + 1) / 2), curStep, save_dir)
    grad_Dx = 2 * maskBG * (x_deg - (x_lr + 1) / 2)
    grad_x = (x_fw_deg - x_bw_deg) * perturb / (2 * h)
    grad = grad_Dx * grad_x
    return grad, x_deg * 2 - 1

def generate(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    degraded = load_video(cfg.input_video, tuple(cfg.resize)).to(device)
    mask = load_video(cfg.input_mask, tuple(cfg.resize)).to(device)[:, :1] > 0.5
    caption = load_caption(cfg.caption_file, cfg.input_video)
    print("Caption:", caption)

    text_encoder = TextEncoder(cfg.text_encoder.model, cfg.text_encoder.device)
    text_embed = text_encoder([caption]).to(device)

    autoencoder = AutoencoderKL(ddconfig=cfg.vae_config, embed_dim=cfg.embed_dim, ckpt_path=cfg.vae_path).to(device).eval()
    stc_encoder = STCEncoder().to(device)
    noise_shaper = NoiseShaper(64, 4, beta=cfg.beta).to(device)

    unet = UNetSTC(cfg).to(device)
    unet.load_state_dict(load_stable_diffusion_pretrained(torch.load(cfg.temporal_unet_ckpt, map_location="cpu"), temporal_attention=False), strict=False)
    unet.eval()

    stc_ckpt = torch.load(cfg.checkpoint, map_location=device)
    stc_encoder.load_state_dict(stc_ckpt["stc_encoder"])
    stc_encoder.eval()

    scheduler = DDPMScheduler(num_train_timesteps=cfg.timesteps, beta_schedule="linear")
    scheduler.set_timesteps(cfg.timesteps, device=device)

    with torch.no_grad():
        B, _, T, H, W = degraded.shape
        degraded_flat = degraded.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)
        z_flat = autoencoder.encode(degraded_flat).sample() * cfg.scale_factor
        z_0 = z_flat.view(B, T, -1, H // 8, W // 8).permute(0, 2, 1, 3, 4)
        z_0 = latent_guard(z_0, name="z_0", normalize=True)

        stc_feat = stc_encoder(degraded)
        noise_first = torch.randn_like(z_0[:, :, 0])
        noise_ind = torch.randn_like(z_0)
        noise = noise_shaper(stc_feat, noise_first, noise_ind)
        noise = latent_guard(noise, name="noise_shaped", normalize=False)

        z_t = scheduler.add_noise(z_0, noise, scheduler.timesteps[0])

    z_cur = z_t.clone()
    for i, t in enumerate(tqdm(scheduler.timesteps, desc="Sampling w/ Correction")):
        with torch.no_grad():
            model_pred = unet(z_cur, torch.tensor([t] * B, device=device), y=text_embed, depth=stc_feat)
            z_pred = scheduler.step(model_pred, t, z_cur).prev_sample

            z_pred_flat = z_pred.permute(0, 2, 1, 3, 4).reshape(B * T, 4, H // 8, W // 8)
            x_img = autoencoder.decode(z_pred_flat)
            if hasattr(x_img, "sample"): x_img = x_img.sample
            x_img = x_img.view(B, T, 3, H, W).permute(0, 2, 1, 3, 4)

        grad_all = torch.zeros_like(x_img)
        for f in range(T):
            grad_f, _ = grad_VVC(x_img[:, :, f], degraded[:, :, f], ~mask[:, :, f], 0, int(t))
            grad_all[:, :, f] = grad_f

        grad_latent = encode_like_latent(grad_all.view(B * T, 3, H, W), autoencoder, cfg.scale_factor)
        grad_latent = grad_latent.view(B, T, 4, H // 8, W // 8).permute(0, 2, 1, 3, 4)
        z_cur = z_pred - cfg.guidance_scale * grad_latent

    with torch.no_grad():
        z_cur_flat = z_cur.permute(0, 2, 1, 3, 4).reshape(B * T, 4, H // 8, W // 8)
        x_hat = autoencoder.decode(z_cur_flat)
        if hasattr(x_hat, "sample"): x_hat = x_hat.sample
        x_hat = x_hat.view(B, T, 3, H, W).permute(0, 2, 1, 3, 4)[0]
        x_hat = ((x_hat + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 3, 0).cpu().numpy()

    out = cv2.VideoWriter(cfg.output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (cfg.resize[0], cfg.resize[1]))
    for frame in x_hat:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print("Saved:", cfg.output_path)
