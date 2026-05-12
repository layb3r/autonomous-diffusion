"""python -m scripts.arbitrary_sample --checkpoint <path_to_checkpoint> --config <path_to_config> --output <output_file>"""
import argparse
from pathlib import Path

import torch
import torchvision.utils as tv_utils

import data.dataset as dataset_module
import models as model_module
from unifed_diffusion import UnifiedDiffusion
from utils import instantiate_from_config, load_checkpoint, load_config


def save_samples(samples, output_path, data_shape):
    if len(data_shape) == 3:
        print(f"+ Saving samples to: {output_path}")
        tv_utils.save_image(samples.clamp(0, 1), output_path, nrow=8)
    elif len(data_shape) == 1:
        print(
            f"+ Samples shape: {samples.shape}\n"
            f"  Min: {samples.min().item():.4f}, Max: {samples.max().item():.4f}"
        )
    else:
        print(f"+ Samples shape: {samples.shape}")


def save_observation(observation, output_path, data_shape):
    if len(data_shape) == 3:
        output_path = Path(output_path)
        obs_path = output_path.with_name(f"{output_path.stem}_observation{output_path.suffix}")
        print(f"+ Saving noisy observation to: {obs_path}")
        tv_utils.save_image(observation.clamp(0, 1), str(obs_path), nrow=8)


def create_diffusion_from_config(ckpt_data, cfg, data_shape, device):
    model = instantiate_from_config(model_module, cfg["model"])

    predictor = None
    predictor_cfg = cfg["diffusion"].get("noise_level_predictor")
    if predictor_cfg is not None and predictor_cfg.get("enabled", True):
        predictor_cfg = {k: v for k, v in predictor_cfg.items() if k != "enabled"}
        predictor = instantiate_from_config(model_module, predictor_cfg)

    diffusion = UnifiedDiffusion(
        model,
        model_type=cfg["diffusion"]["model_type"],
        data_shape=data_shape,
        noise_level_predictor=predictor,
    )

    missing, unexpected = diffusion.load_state_dict(ckpt_data["model"], strict=False)
    if missing:
        print(f"! Missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"! Unexpected checkpoint keys: {len(unexpected)}")

    diffusion.to(device)
    diffusion.eval()
    return diffusion


def main():
    parser = argparse.ArgumentParser(description="Sample from arbitrary noisy observation")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--config", default=None, help="Path to config")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--output", default="arbitrary_samples.png", help="Output file")
    parser.add_argument("--device", default=None, help="Device override")
    parser.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help="Optional start noise level t in (0, 1). If omitted, a random t is used.",
    )
    parser.add_argument(
        "--use-predicted-noise-level",
        action="store_true",
        help="Use the noise-level predictor to estimate t from the noisy observation.",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"+ Loading checkpoint: {args.checkpoint}")
    ckpt_data = load_checkpoint(args.checkpoint, device=device)

    config_path = Path(args.config) if args.config else Path(args.checkpoint).with_name("config.yaml")
    cfg = load_config(config_path)
    data = instantiate_from_config(dataset_module, cfg["data"])

    n_samples = args.n_samples if args.n_samples is not None else cfg["training"]["sample_size"]
    sampling_cfg = cfg.get("sampling", {})

    print(f"+ Creating {cfg['model']['type']} diffusion model")
    diffusion = create_diffusion_from_config(ckpt_data, cfg, data.sample_shape, device)

    if args.noise_level is None:
        true_t = float(torch.empty(1).uniform_(0.1, 0.95).item())
        print(f"+ Random start noise level t: {true_t:.4f}")
    else:
        true_t = float(max(1e-6, min(1.0 - 1e-6, args.noise_level)))
        print(f"+ User-provided start noise level t: {true_t:.4f}")

    x0_seed = torch.randn(n_samples, *data.sample_shape, device=device)
    t_batch = torch.full((n_samples,), true_t, device=device)

    with torch.inference_mode():
        x_obs, _ = diffusion.forward_diffusion(x0_seed, t_batch)

        if diffusion.noise_level_predictor is not None:
            predicted_t = diffusion.noise_level_predictor(x_obs).squeeze(-1)
            print(
                f"+ Predictor t stats -> mean: {predicted_t.mean().item():.4f}, "
                f"std: {predicted_t.std(unbiased=False).item():.4f}"
            )

        use_predicted_t = args.use_predicted_noise_level and diffusion.noise_level_predictor is not None
        if args.noise_level is None and diffusion.noise_level_predictor is not None:
            # Default behavior for random observation: estimate t from predictor.
            use_predicted_t = True

        if use_predicted_t:
            print("+ Denoising from predictor-estimated start noise level")
            samples = diffusion.sample_from_observation(
                x_obs,
                t_start=None,
                t_min=sampling_cfg.get("t_min", 1e-3),
                stochastic=sampling_cfg.get("stochastic"),
                eta=sampling_cfg.get("eta", 1.0),
            )
        else:
            print("+ Denoising from provided/ground-truth start noise level")
            samples = diffusion.sample_from_observation(
                x_obs,
                t_start=true_t,
                t_min=sampling_cfg.get("t_min", 1e-3),
                stochastic=sampling_cfg.get("stochastic"),
                eta=sampling_cfg.get("eta", 1.0),
            )

    save_observation(x_obs, args.output, data.sample_shape)
    save_samples(samples, args.output, data.sample_shape)
    print("+ Done!")


if __name__ == "__main__":
    main()
"""python -m scripts.arbitrary_sample --checkpoint train-log\concentric_circles_10\model-final.pt --config configs\concentric_circles.yaml"""