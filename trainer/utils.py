import os
import torch
import shutil

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }, path)


def load_checkpoint(model, optimizer, path):
    if not os.path.exists(path):
        print("Checkpoint not found.")
        return 0
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch']


def cleanup_logs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
