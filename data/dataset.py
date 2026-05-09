from torch.utils.data import Dataset
import torch

from utils import make_concentric_circle_images


class ConcentricCirclesDataset(Dataset):
    def __init__(
        self,
        n_samples=4096,
        image_size=32,
        radii=(0.35, 0.65),
        thickness=0.05,
        noise=0.03,
        center_jitter=0.08,
    ):
        self.data = make_concentric_circle_images(
            n_samples=n_samples,
            image_size=image_size,
            radii=radii,
            thickness=thickness,
            noise=noise,
            center_jitter=center_jitter,
        ).float()
        self.sample_shape = tuple(self.data.shape[1:])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]


class OneDGaussianDataset(Dataset):
    def __init__(self, n_samples=2000, noise_std=0.3, modes=(-3.0, 2.0)):
        counts = [n_samples // len(modes)] * len(modes)
        for index in range(n_samples % len(modes)):
            counts[index] += 1

        chunks = [torch.randn(count) * noise_std + mode for count, mode in zip(counts, modes)]
        self.data = torch.cat(chunks).unsqueeze(1).float()
        self.sample_shape = tuple(self.data.shape[1:])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]