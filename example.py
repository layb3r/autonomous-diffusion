from models import TimeMLP, SmallUNet
from unifed_diffusion import UnifiedDiffusion, Trainer
import torch
from utils import make_concentric_circle_images, get_new_run_path
import torchvision.utils as tv_utils

def test_1d_gaussian():
    # 1D data: mixture of two Gaussians
    data = torch.cat([torch.randn(1000)*0.3 - 3, torch.randn(1000)*0.3 + 2])
    data = data.unsqueeze(1)

    # Simple MLP
    model = TimeMLP(data_dim=1, hidden_dim=64)
    diff = UnifiedDiffusion(model, model_type="EDM", data_shape=(1,), T=1000)
    out_dir = get_new_run_path(experiment_name="1d_gaussian")

    trainer = Trainer(
        diff,
        data,
        train_batch_size=128,
        train_lr=2e-4,
        train_num_steps=10000,
        save_and_sample_every=250,
        sample_size=64,
        results_folder=str(out_dir),
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    trainer.train()

    # Sample and plot histogram
    samples = diff.sample(n_samples=2000, n_steps=200)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.hist(data.cpu().numpy(), bins=50, alpha=0.5, label="Data")
    plt.hist(samples.cpu().numpy(), bins=50, alpha=0.5, label="Samples")
    plt.legend()
    plt.title("1D Mixture of Gaussians")
    plt.savefig(out_dir / "final_samples_hist.png")

def test_centric_circles():
    images = make_concentric_circle_images(
        n_samples=4096,
        image_size=32,
        radii=(0.35, 0.65),
        thickness=0.05,
        noise=0.03,
    )

    out_dir = get_new_run_path(experiment_name="concentric_circles")
    tv_utils.save_image(images[:64], str(out_dir + "/train_data_preview.png"), nrow=8)

    # 3) Train unified diffusion on images.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    denoiser = SmallUNet(noise_agnostic=True)
    # denoiser = TimeConvNet(channels=1, base_dim=64, time_dim=128)
    diffusion = UnifiedDiffusion(
        denoiser,
        model_type="EDM",
        data_shape=(1, 32, 32),
    )

    trainer = Trainer(
        diffusion,
        images,
        train_batch_size=128,
        train_lr=1e-4,
        train_num_steps=1000,
        save_and_sample_every=50,
        sample_size=64,
        results_folder=str(out_dir),
        device=device,
    )
    trainer.train()

    diffusion.eval()
    with torch.inference_mode():
        sampled = diffusion.sample(n_samples=64, n_steps=200)
    tv_utils.save_image(sampled.clamp(0, 1), str(out_dir + "/final_samples.png"), nrow=8)
    print(f"saved outputs to: {out_dir}")

if __name__ == "__main__":
    # test_1d_gaussian()
    test_centric_circles()