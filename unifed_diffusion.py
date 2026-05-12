from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import utils as tv_utils
from tqdm.auto import tqdm
from utils import get_new_run_path
from model_policies import (
    POLICY_REGISTRY,
    BaseModelPolicy,
    create_policy,
)


def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)


def cycle(dl):
    while True:
        for data in dl:
            yield data


def expand_like(coef, x):
    # Broadcast [B] scalars across [B, ...] tensors.
    return coef.view(coef.shape[0], *([1] * (x.ndim - 1)))


class UnifiedDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        model_type="DDPM",
        policy=None,
        policy_kwargs=None,
        data_shape,
        eps=1e-6,
        noise_level_predictor=None,
    ):
        super().__init__()
        self.model = model
        self.data_shape = tuple(data_shape)
        self.eps = float(eps)
        self.noise_level_predictor = noise_level_predictor

        if exists(policy):
            self.policy = policy
        else:
            policy_kwargs = default(policy_kwargs, {})
            self.policy = create_policy(model_type, eps=eps, **policy_kwargs)

    @property
    def model_type(self):
        return self.policy.name

    @property
    def device(self):
        return next(self.model.parameters()).device

    def set_model_type(self, model_type):
        # Keep current policy hyperparameters when switching type, unless incompatible.
        current = self.policy.__dict__.copy()
        current.pop("solver", None)
        current.pop("eps", None)

        policy_cls = POLICY_REGISTRY[model_type.upper()]
        accepted = policy_cls.__dataclass_fields__.keys()
        kwargs = {k: v for k, v in current.items() if k in accepted}
        self.policy = create_policy(model_type, eps=self.eps, **kwargs)

    def schedule(self, t):
        return self.policy.schedule(t)

    def forward_diffusion(self, x0, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        a, b, _, _ = self.schedule(t)
        return expand_like(a, x0) * x0 + expand_like(b, x0) * noise, noise

    def field_from_x0(self, x_t, x0, t):
        a, b, c, d = self.schedule(t)
        b_safe = b.clamp(min=self.eps)
        k1 = expand_like(d / b_safe, x_t)
        k2 = expand_like(c - d * a / b_safe, x_t)
        return k1 * x_t + k2 * x0

    def _sampler_coefficients(self, t):
        return self.policy.sampler_coefficients(t)

    def p_losses(self, x0):
        bsz = x0.shape[0]
        t = torch.rand(bsz, device=x0.device).clamp(self.eps, 1.0 - self.eps)

        x_t, noise = self.forward_diffusion(x0, t)
        out = self.model(x_t, t)

        _, _, c, d = self.schedule(t)
        target = expand_like(c, x0) * x0 + expand_like(d, x0) * noise
        loss = F.mse_loss(out, target)
        
        # Add noise level prediction loss if predictor is specified
        if exists(self.noise_level_predictor):
            t_pred = self.noise_level_predictor(x_t).squeeze(-1)
            t_pred_loss = F.mse_loss(t_pred, t)
            loss = loss + t_pred_loss
        
        return loss

    def forward(self, x0):
        assert x0.shape[1:] == self.data_shape, (
            f"expected [batch, {self.data_shape}], got {tuple(x0.shape)}"
        )
        return self.p_losses(x0)

    @torch.inference_mode()
    def sample(
        self,
        n_samples=128,
        n_steps=None,
        t_min=1e-3,
        stochastic=None,
        eta=1.0,
        solver=None,
        verbose=False,
    ):
        stochastic = default(stochastic, self.policy.default_stochastic)
        solver = default(solver, self.policy.solver)
        n_steps = default(n_steps, self.policy.n_steps)
        # print(f"sampling with {solver} solver, stochastic={stochastic}, eta={eta}, n_steps={n_steps}")

        x = torch.randn(n_samples, *self.data_shape, device=self.device)
        t_grid = self.policy.discretize_timesteps(n_steps, t_min, device=self.device)
        for i in range(n_steps):
            t_cur = t_grid[i].expand(n_samples)
            t_next = t_grid[i + 1].expand(n_samples)

            def drift_fn(x_state, t_state):
                out = self.model(x_state, t_state)
                mu, nu = self._sampler_coefficients(t_state)
                return expand_like(mu, x_state) * x_state + expand_like(nu, x_state) * out

            if solver == "heun":
                x = self.policy.ode_step(x, t_cur, t_next, drift_fn)
            else:
                # Temporary override: keep ability to force Euler even for EDM.
                original_solver = self.policy.solver
                self.policy.solver = solver
                x = self.policy.ode_step(x, t_cur, t_next, drift_fn)
                self.policy.solver = original_solver

            if stochastic:
                x = self.policy.stochastic_step(x, t_cur, t_next, eta=eta)

        return x

    @torch.inference_mode()
    def sample_from_observation(
        self,
        x_obs,
        *,
        t_start=None,
        n_steps=None,
        t_min=1e-3,
        stochastic=None,
        eta=1.0,
        solver=None,
    ):
        assert x_obs.shape[1:] == self.data_shape, (
            f"expected [batch, {self.data_shape}], got {tuple(x_obs.shape)}"
        )

        stochastic = default(stochastic, self.policy.default_stochastic)
        solver = default(solver, self.policy.solver)
        n_steps = default(n_steps, self.policy.n_steps)

        bsz = x_obs.shape[0]
        if t_start is None:
            if not exists(self.noise_level_predictor):
                raise ValueError("noise_level_predictor is required when t_start is not provided")
            t_pred = self.noise_level_predictor(x_obs).squeeze(-1)
            t_scalar = t_pred.mean().clamp(self.eps, 1.0 - self.eps)
        else:
            t_scalar = torch.as_tensor(t_start, device=self.device, dtype=x_obs.dtype)
            t_scalar = t_scalar.clamp(self.eps, 1.0 - self.eps)

        if t_scalar.ndim != 0:
            raise ValueError("t_start must be a scalar in [0, 1] when provided")

        t_full = self.policy.discretize_timesteps(n_steps, t_min, device=self.device)
        t_tail = t_full[t_full < t_scalar]
        t_grid = torch.cat([t_scalar[None], t_tail], dim=0)

        if t_grid.shape[0] < 2:
            return x_obs

        x = x_obs
        for i in range(t_grid.shape[0] - 1):
            t_cur = t_grid[i].expand(bsz)
            t_next = t_grid[i + 1].expand(bsz)

            def drift_fn(x_state, t_state):
                out = self.model(x_state, t_state)
                mu, nu = self._sampler_coefficients(t_state)
                return expand_like(mu, x_state) * x_state + expand_like(nu, x_state) * out

            if solver == "heun":
                x = self.policy.ode_step(x, t_cur, t_next, drift_fn)
            else:
                original_solver = self.policy.solver
                self.policy.solver = solver
                x = self.policy.ode_step(x, t_cur, t_next, drift_fn)
                self.policy.solver = original_solver

            if stochastic:
                x = self.policy.stochastic_step(x, t_cur, t_next, eta=eta)

        return x



class TensorDatasetND(TorchDataset):
    def __init__(self, data):
        super().__init__()
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        data = data.float()
        if data.ndim < 2:
            raise ValueError(f"expected at least 2D [N, ...], got {tuple(data.shape)}")
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]


class Trainer:
    def __init__(
        self,
        diffusion_model,
        data,
        *,
        train_batch_size=256,
        train_lr=1e-4,
        train_num_steps=20000,
        gradient_accumulate_every=1,
        max_grad_norm=1.0,
        amp=False,
        num_workers=0,
        save_and_sample_every=1000,
        sample_size=256,
        results_folder=None,
        device=None,
    ):
        self.model = diffusion_model
        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm
        self.save_and_sample_every = save_and_sample_every
        self.sample_size = sample_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        self.ds = data if isinstance(data, TorchDataset) else TensorDatasetND(data)
        dl = DataLoader(
            self.ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=(self.device.type == "cuda"),
            num_workers=num_workers,
        )
        self.dl = cycle(dl)

        self.opt = Adam(self.model.parameters(), lr=train_lr)
        self.use_amp = amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler("cuda", enabled=self.use_amp)

        self.results_folder = default(results_folder, lambda: get_new_run_path("train-log", "unified"))
        Path(self.results_folder).mkdir(parents=True, exist_ok=True)
        self.step = 0

    def save(self, milestone):
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "opt": self.opt.state_dict(),
                "scaler": self.scaler.state_dict() if self.use_amp else None,
                "model_type": self.model.model_type,
            },
            str(self.results_folder + f"/model-{milestone}.pt"),
        )

    def load(self, milestone):
        data = torch.load(self.results_folder + f"/model-{milestone}.pt", map_location=self.device)
        self.model.load_state_dict(data["model"])
        self.opt.load_state_dict(data["opt"])
        self.step = data["step"]

        if exists(data.get("model_type")):
            self.model.set_model_type(data["model_type"])
        if self.use_amp and exists(data.get("scaler")):
            self.scaler.load_state_dict(data["scaler"])

    def train(self):
        pbar = tqdm(range(self.step, self.train_num_steps), desc="training")

        while self.step < self.train_num_steps:
            self.model.train()
            total_loss = 0.0
            self.opt.zero_grad(set_to_none=True)

            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dl).to(self.device)

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    loss = self.model(batch)
                    loss = loss / self.gradient_accumulate_every

                total_loss += loss.item()

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            if self.use_amp:
                self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if self.use_amp:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()

            self.step += 1
            pbar.set_description(f"training | loss: {total_loss:.4f}")
            pbar.update(1)

            if self.step % self.save_and_sample_every == 0:
                self.model.eval()
                with torch.inference_mode():
                    samples = self.model.sample(n_samples=self.sample_size)

                # save_path = self.results_folder + f"/samples-{self.step}.pt"
                # torch.save(samples.cpu(), save_path)
                # self.save(self.step)

                if samples.ndim == 4:
                    img_path = self.results_folder + f"/samples-{self.step}.png"
                    tv_utils.save_image(samples.clamp(0, 1), str(img_path), nrow=8)

        # final save at end of training
        self.save("final")

        pbar.close()
        print("training complete")