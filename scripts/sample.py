"""python -m scripts.sample --checkpoint <path_to_checkpoint> --config <path_to_config> --output <output_file>"""
import argparse
from pathlib import Path

import torch
import torchvision.utils as tv_utils

import data.dataset as dataset_module
import models as model_module
from unifed_diffusion import UnifiedDiffusion
from utils import clean_state_dict_prefix, instantiate_from_config, load_checkpoint, load_config


def save_samples(samples, output_path, data_shape):
    actions = {
        3: lambda: (
            print(f"+ Saving samples to: {output_path}"),
            tv_utils.save_image(samples.clamp(0, 1), output_path, nrow=8),
        ),
        1: lambda: print(f"+ Samples shape: {samples.shape}\n  Min: {samples.min().item():.4f}, Max: {samples.max().item():.4f}"),
    }
    actions[len(data_shape)]()


def create_model_from_config(ckpt_data, cfg, device):
    model = instantiate_from_config(model_module, cfg["model"])
    model.load_state_dict(clean_state_dict_prefix(ckpt_data["model"]))
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Sample from trained diffusion model")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--config", default=None, help="Path to config")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--output", default="samples.png", help="Output file")
    parser.add_argument("--device", default=None, help="Device override")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"+ Loading checkpoint: {args.checkpoint}")
    ckpt_data = load_checkpoint(args.checkpoint, device=device)

    config_path = Path(args.config) if args.config else Path(args.checkpoint).with_name("config.yaml")
    cfg = load_config(config_path)
    data = instantiate_from_config(dataset_module, cfg["data"])

    print(f"+ Creating {cfg['model']['type']} diffusion model")
    model = create_model_from_config(ckpt_data, cfg, device)
    
    diffusion = UnifiedDiffusion(
        model,
        model_type=cfg["diffusion"]["model_type"],
        data_shape=data.sample_shape
    )
    diffusion.eval()

    sampling_cfg = cfg.get("sampling", {})
    n_samples = args.n_samples if args.n_samples is not None else cfg["training"]["sample_size"]
    print(f"+ Sampling {n_samples} samples...")
    with torch.inference_mode():
        samples = diffusion.sample(
            n_samples=n_samples,
            t_min=sampling_cfg.get("t_min", 1e-3),
            stochastic=sampling_cfg.get("stochastic"),
            eta=sampling_cfg.get("eta", 1.0),
        )

    save_samples(samples, args.output, data.sample_shape)

    print("+ Done!")


if __name__ == "__main__":
    main()
