import imageio
import numpy as np

def save_gif(tensor, path, fps=10):
    """
    tensor: (C, F, H, W), in [-1, 1]
    """
    video = ((tensor.permute(1, 2, 3, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
    imageio.mimsave(path, list(video), fps=fps)
