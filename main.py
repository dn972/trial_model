import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import argparse
import yaml
from easydict import EasyDict
from trainer.train import train
from inference.generate import generate as generate_with_grad
from inference.generate_no_grad import generate as generate_no_grad
from data.video_dataset import VideoDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'inference_no_grad'], required=True)
    args = parser.parse_args()

    # === Load config ===
    with open(args.config, 'r') as f:
        cfg_raw = yaml.safe_load(f)
    cfg_all = EasyDict(cfg_raw)

    # === Extract sub-config for the selected mode ===
    if args.mode == 'train':
        cfg = cfg_all.train
        cfg.text_encoder = cfg_all.text_encoder

        # Postprocess
        cfg.lr = float(cfg.lr)
        cfg.scale_factor = float(cfg.scale_factor)
        cfg.lambda_bg = float(cfg.lambda_bg)
        cfg.ddim_steps = int(cfg.timesteps)

        dataset = VideoDataset(
            root_original=cfg.root_original,
            root_train=cfg.root_train,
            root_mask=cfg.root_mask,
            caption_file=cfg.caption_file,
            resize=tuple(cfg.resize))
        train(cfg, dataset)

    elif args.mode == 'inference':
        cfg = cfg_all.inference
        cfg.text_encoder = cfg_all.text_encoder
        cfg.ddim_steps = int(cfg.timesteps)
        generate_with_grad(cfg)

    elif args.mode == 'inference_no_grad':
        cfg = cfg_all.inference
        cfg.text_encoder = cfg_all.text_encoder
        cfg.ddim_steps = int(cfg.timesteps)
        generate_no_grad(cfg)

    
