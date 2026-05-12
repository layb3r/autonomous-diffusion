"""python -m scripts.train --config <path_to_config>"""
import argparse
from pathlib import Path

import data.dataset as dataset_module
import models as model_module
from unifed_diffusion import Trainer, UnifiedDiffusion
from utils import get_new_run_path, instantiate_from_config, load_config, save_config, seed_everything


def main():
    parser = argparse.ArgumentParser(description="Train unified diffusion model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--device", default=None, help="Device override")
    parser.add_argument("--seed", type=int, default=None, help="Seed override")
    parser.add_argument(
        "--noise-level-predictor-type",
        default=None,
        help="Optional class name in models.py for noise-level predictor override",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    device = cfg["training"]["device"] if args.device is None else args.device

    seed_everything(seed)
    print(f"+ Seed set to {seed}")

    print(f"+ Creating {cfg['data']['type']} dataset...")
    data = instantiate_from_config(dataset_module, cfg["data"])
    print(f"  Data shape: {data.sample_shape}")

    print(f"+ Creating {cfg['model']['type']} model...")
    model = instantiate_from_config(model_module, cfg["model"])

    diffusion_cfg = cfg["diffusion"]
    noise_level_predictor = None
    predictor_cfg = diffusion_cfg.get("noise_level_predictor")

    if predictor_cfg is None and args.noise_level_predictor_type is not None:
        predictor_cfg = {"type": args.noise_level_predictor_type}

    if predictor_cfg is not None:
        predictor_cfg = dict(predictor_cfg)

        if args.noise_level_predictor_type is not None:
            predictor_cfg["type"] = args.noise_level_predictor_type

        if predictor_cfg.get("enabled", True):
            if "type" not in predictor_cfg:
                raise ValueError(
                    "diffusion.noise_level_predictor must include a 'type' field when enabled"
                )

            predictor_cfg.pop("enabled", None)
            print(f"+ Creating {predictor_cfg['type']} noise-level predictor...")
            noise_level_predictor = instantiate_from_config(model_module, predictor_cfg)

    diffusion = UnifiedDiffusion(
        model,
        model_type=diffusion_cfg["model_type"],
        data_shape=data.sample_shape,
        noise_level_predictor=noise_level_predictor,
    )
    print(f"+ Diffusion model type: {diffusion.model_type}")

    train_cfg = cfg["training"]
    out_dir = get_new_run_path(
        base_dir="train-log",
        experiment_name=cfg["experiment_name"],
    )
    print(f"+ Results will be saved to: {out_dir}")

    trainer = Trainer(
        diffusion,
        data,
        train_batch_size=train_cfg["batch_size"],
        train_lr=train_cfg["learning_rate"],
        train_num_steps=train_cfg["num_steps"],
        gradient_accumulate_every=train_cfg.get("gradient_accumulate_every", 1),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        amp=train_cfg.get("amp", False),
        num_workers=train_cfg.get("num_workers", 0),
        save_and_sample_every=train_cfg["save_and_sample_every"],
        sample_size=train_cfg["sample_size"],
        results_folder=str(out_dir),
        device=device,
    )

    config_save_path = Path(out_dir) / "config.yaml"
    save_config(cfg, config_save_path)
    print(f"+ Config saved to: {config_save_path}")

    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    trainer.train()

    print("\n" + "="*50)
    print(f"+ Training complete! Results in: {out_dir}")
    print("="*50)


if __name__ == "__main__":
    main()

"""python -m scripts.train --config ./configs/concentric_circles.yaml"""