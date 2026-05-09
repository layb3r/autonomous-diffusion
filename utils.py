from pathlib import Path

import numpy as np
import torch
import yaml
import os
import glob
import re
import random

def make_concentric_circles(n_points=100, radii=(0.5, 1.0), noise=0.02):
    """Generate concentric circles dataset in 2D."""
    points = []
    for r in radii:
        theta = np.linspace(0, 2 * np.pi, n_points // len(radii), endpoint=False)
        x = r * np.cos(theta) + noise * np.random.randn(len(theta))
        y = r * np.sin(theta) + noise * np.random.randn(len(theta))
        points.append(np.stack([x, y], axis=1))
    return np.vstack(points)


def make_swiss_roll_2d(n_points=100, noise=0.02):
    """Swiss roll projected to 2D (a coiled curve)."""
    t = 1.5 * np.pi * (1 + 2 * np.linspace(0, 1, n_points))
    x = t * np.cos(t) / (4 * np.pi) + noise * np.random.randn(n_points)
    y = t * np.sin(t) / (4 * np.pi) + noise * np.random.randn(n_points)
    data = np.stack([x, y], axis=1)
    return (data - data.mean(0)) / data.std()  # normalise


def make_figure_eight(n_points=100, noise=0.02):
    """Figure-eight (lemniscate) curve in 2D."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = np.sin(t) + noise * np.random.randn(n_points)
    y = np.sin(t) * np.cos(t) + noise * np.random.randn(n_points)
    data = np.stack([x, y], axis=1)
    return data / np.std(data)


def embed_in_high_dim(data_2d, D):
    """
    Embed 2D data into R^D using a random orthogonal projection.
    P in R^{D x 2} with P^T P = I.
    """
    if D == 2:
        return data_2d, np.eye(2)
    
    # Random orthogonal matrix: QR decomposition of random Gaussian
    G = np.random.randn(D, D)
    Q, _ = np.linalg.qr(G)
    P = Q[:, :2]  # D x 2 column-orthonormal matrix
    
    data_high = data_2d @ P.T  # (N, D)
    return data_high, P


def project_back_to_2d(samples_high_dim, P):
    """Project high-dim samples back to 2D via P^T."""
    return samples_high_dim @ P  # (N, 2)


def make_concentric_circle_images(
    n_samples=4096,
    image_size=32,
    radii=(0.35, 0.65),
    thickness=0.05,
    noise=0.03,
    center_jitter=0.08,
):
    grid = torch.linspace(-1.0, 1.0, image_size)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")

    images = []
    for _ in range(n_samples):
        cx, cy = (torch.rand(2) * 2 - 1) * center_jitter
        rr = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        img = torch.zeros_like(rr)
        for r in radii:
            img = img + torch.exp(-((rr - r) ** 2) / (2 * thickness ** 2))

        img = img / img.max().clamp(min=1e-6)
        img = (img + noise * torch.randn_like(img)).clamp(0.0, 1.0)
        images.append(img.unsqueeze(0))

    return torch.stack(images, dim=0)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg, config_path):
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(ckpt_path, device="cpu"):
    return torch.load(ckpt_path, map_location=device)


def instantiate_from_config(module, cfg):
    cls = getattr(module, cfg["type"])
    kwargs = {key: value for key, value in cfg.items() if key != "type"}
    return cls(**kwargs)


def clean_state_dict_prefix(state_dict, prefix="model."):
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict

def get_new_run_path(base_dir='train-log', experiment_name='run'):
    """Creates a unique directory, handling increments correctly even if folders are deleted."""
    base_path = Path(base_dir)
    prefix = experiment_name or 'run'
    
    base_path.mkdir(parents=True, exist_ok=True)

    # Find the highest existing ID
    existing_ids = []
    for folder in base_path.glob(f"{prefix}_*"):
        match = re.search(rf"{prefix}_(\d+)$", folder.name)
        if match:
            existing_ids.append(int(match.group(1)))

    next_id = max(existing_ids, default=0) + 1
    new_run_dir = base_path / f"{prefix}_{next_id}"

    new_run_dir.mkdir()
    return str(new_run_dir)