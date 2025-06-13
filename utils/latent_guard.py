# === latent_guard.py ===
import torch

def log_tensor_stats(name, tensor):
    mean = tensor.mean().item()
    std = tensor.std().item()
    

def normalize_latent(tensor, target_mean=0.0, target_std=1.0):
    tensor = tensor - tensor.mean()
    tensor = tensor / (tensor.std() + 1e-6)
    return tensor * target_std + target_mean

def latent_guard(tensor, name="latent", normalize=False, target_mean=0.0, target_std=1.0, verbose=True):
    if verbose:
        log_tensor_stats(name, tensor)
    if normalize:
        tensor = normalize_latent(tensor, target_mean, target_std)
        if verbose:
            log_tensor_stats(name + " (normalized)", tensor)
    return tensor

def latent_penalty(tensor, target_mean=0.0, target_std=1.0):
    mean_penalty = (tensor.mean() - target_mean).pow(2)
    std_penalty = (tensor.std() - target_std).pow(2)
    return mean_penalty + std_penalty

# === Aliases for compatibility with imports in train.py or generate.py ===
latent_checker = latent_guard
latent_normalizer = normalize_latent
