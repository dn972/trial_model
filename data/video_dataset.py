import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from models.text_encoder import TextEncoder


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.stack(frames, axis=0)  # (F, H, W, C)


def preprocess(frames, size=None):
    if size:
        frames = [cv2.resize(f, size, interpolation=cv2.INTER_LINEAR) for f in frames]
    frames = np.stack(frames, axis=0)
    frames = frames.astype(np.float32) / 127.5 - 1.0
    frames = frames.transpose(3, 0, 1, 2)
    return torch.from_numpy(frames)


def preprocess_mask(frames, size=None):
    masks = []
    for f in frames:
        if f.ndim == 3 and f.shape[-1] == 3:
            f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        f = (f > 127).astype(np.float32)
        if size:
            f = cv2.resize(f, size, interpolation=cv2.INTER_NEAREST)
        masks.append(f)
    masks = np.stack(masks, axis=0)
    masks = masks[None, ...]
    return torch.from_numpy(masks)


class VideoDataset(Dataset):
    def __init__(self, root_original, root_train, root_mask, caption_file, resize=None, device="cuda"):
        self.resize = resize
        self.device = device
        self.encoder = TextEncoder(device=device)
        self.data = []
        self.caption_dict = {}

        with open(caption_file, 'r') as f:
            for line in f:
                key, caption = line.strip().split("|", 1)
                self.caption_dict[key] = caption

        for cls in sorted(os.listdir(root_original)):
            p_ori = os.path.join(root_original, cls)
            p_train = os.path.join(root_train, cls)
            p_mask = os.path.join(root_mask, cls)
            for file in sorted(os.listdir(p_ori)):
                if file.endswith('.mp4'):
                    video_name = file[:-4]
                    base_name = video_name.split('_')[0]  # remove _640x360_24
                    key = f"{cls}/{base_name}"

                    if key in self.caption_dict:
                        if os.path.exists(os.path.join(p_train, file)) and os.path.exists(os.path.join(p_mask, file)):
                            self.data.append({
                                'ori': os.path.join(p_ori, file),
                                'train': os.path.join(p_train, file),
                                'mask': os.path.join(p_mask, file),
                                'caption': self.caption_dict[key]
                            })
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x_0 = preprocess(read_video(item['ori']), size=self.resize)
        degraded = preprocess(read_video(item['train']), size=self.resize)
        mask = preprocess_mask(read_video(item['mask']), size=self.resize)
        caption = item['caption']


        return {
            'video': x_0,
            'train': degraded,
            'mask': mask,
            'caption': caption
        }
