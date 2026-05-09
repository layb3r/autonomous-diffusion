from dataclasses import dataclass

import torch


class ODESolver:
    @staticmethod
    def _expand_like(coef, x):
        return coef.view(coef.shape[0], *([1] * (x.ndim - 1)))

    @classmethod
    def euler_step(cls, x, t_cur, t_next, drift_fn):
        dt = (t_cur - t_next).clamp(min=0.0)
        d_cur = drift_fn(x, t_cur)
        return x - cls._expand_like(dt, x) * d_cur

    @classmethod
    def heun_step(cls, x, t_cur, t_next, drift_fn):
        dt = (t_cur - t_next).clamp(min=0.0)
        d_cur = drift_fn(x, t_cur)

        x_pred = x - cls._expand_like(dt, x) * d_cur
        d_next = drift_fn(x_pred, t_next)

        return x - 0.5 * cls._expand_like(dt, x) * (d_cur + d_next)


@dataclass
class BaseModelPolicy:
    eps: float = 1e-6
    solver: str = "euler"

    @property
    def name(self):
        return self.__class__.__name__.replace("Policy", "").upper()

    @property
    def default_stochastic(self):
        return False

    def schedule(self, t):
        raise NotImplementedError

    def discretize_timesteps(self, n_steps, t_min, *, device):
        return torch.linspace(1.0, float(t_min), n_steps + 1, device=device)

    def sampler_coefficients(self, t, deriv_dt=1e-4):
        t1 = t.clamp(self.eps, 1.0 - self.eps)
        t2 = (t1 + deriv_dt).clamp(self.eps, 1.0 - self.eps)

        a, b, c, d = self.schedule(t1)
        a2, b2, _, _ = self.schedule(t2)

        a_dot = (a2 - a) / deriv_dt
        b_dot = (b2 - b) / deriv_dt

        denom = (a * d - b * c).clamp(min=self.eps)
        mu = (a_dot * d - b_dot * c) / denom
        nu = (b_dot * a - a_dot * b) / denom
        return mu, nu

    def stochastic_step(self, x, t_cur, t_next, *, eta=1.0):
        return x

    def ode_step(self, x, t_cur, t_next, drift_fn):
        if self.solver == "heun":
            return ODESolver.heun_step(x, t_cur, t_next, drift_fn)
        return ODESolver.euler_step(x, t_cur, t_next, drift_fn)


@dataclass
class DDPMPolicy(BaseModelPolicy):
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    n_steps: int = 100
    solver: str = "euler"

    @property
    def default_stochastic(self):
        return True

    def _beta_t(self, t):
        return self.beta_start + t * (self.beta_end - self.beta_start)

    def _alpha_bar(self, t):
        t_scaled = t * self.T
        log_alpha_bar = -t_scaled * (
            self.beta_start + 0.5 * t_scaled * (self.beta_end - self.beta_start) / self.T
        )
        return torch.exp(log_alpha_bar)

    def schedule(self, t):
        t = t.clamp(self.eps, 1.0 - self.eps)
        alpha_bar = self._alpha_bar(t)
        a = torch.sqrt(alpha_bar)
        b = torch.sqrt((1.0 - alpha_bar).clamp(min=self.eps))
        c = torch.zeros_like(t)
        d = torch.ones_like(t)
        return a, b, c, d

    def stochastic_step(self, x, t_cur, t_next, *, eta=1.0):
        dt = (t_cur - t_next).clamp(min=self.eps)
        beta_t = self._beta_t(t_cur).clamp(min=self.eps)
        noise_std = torch.sqrt(beta_t * dt) * eta
        noise_std = noise_std.view(noise_std.shape[0], *([1] * (x.ndim - 1)))
        return x + noise_std * torch.randn_like(x)


@dataclass
class EDMPolicy(BaseModelPolicy):
    edm_rho: float = 7.0
    solver: str = "heun"
    n_steps: int = 18
    # https://github.com/NVlabs/edm/blob/main/generate.py

    def schedule(self, t):
        t = t.clamp(self.eps, 1.0 - self.eps)
        a = torch.ones_like(t)
        b = t.clamp(min=self.eps)
        c = torch.ones_like(t)
        d = torch.zeros_like(t)
        return a, b, c, d

    def sampler_coefficients(self, t, deriv_dt=1e-4):
        # mu = 1/t, nu = -1/t
        mu = 1.0 / t.clamp(min=self.eps)
        nu = -mu
        # print(f"EDM coefficients at t={t[0].item():.4f}: mu={mu[0].item():.4f}, nu={nu[0].item():.4f}")
        return mu, nu

    def discretize_timesteps(self, n_steps, t_min, *, device):
        step_indices = torch.arange(n_steps + 1, dtype=torch.float32, device=device)
        sigma_max = 1
        sigma_min = 0.002
        return (
            sigma_max ** (1.0 / self.edm_rho)
            + step_indices / n_steps * (sigma_min ** (1.0 / self.edm_rho) - sigma_max ** (1.0 / self.edm_rho))
        ) ** self.edm_rho


@dataclass
class FMPolicy(BaseModelPolicy):
    solver: str = "euler"
    n_steps: int = 50

    def schedule(self, t):
        t = t.clamp(self.eps, 1.0 - self.eps)
        a = 1.0 - t
        b = t.clamp(min=self.eps)
        c = -torch.ones_like(t)
        d = torch.ones_like(t)
        return a, b, c, d


@dataclass
class EQMPolicy(BaseModelPolicy):
    solver: str = "euler"
    n_steps: int = 100
    def schedule(self, t):
        t = t.clamp(self.eps, 1.0 - self.eps)
        a = 1.0 - t
        b = t.clamp(min=self.eps)
        c = -t
        d = t
        return a, b, c, d


POLICY_REGISTRY = {
    "DDPM": DDPMPolicy,
    "EDM": EDMPolicy,
    "FM": FMPolicy,
    "EQM": EQMPolicy,
}


def create_policy(model_type, **kwargs):
    key = model_type.upper()
    if key not in POLICY_REGISTRY:
        raise ValueError(f"Unknown model_type {model_type}. Available: {sorted(POLICY_REGISTRY)}")
    return POLICY_REGISTRY[key](**kwargs)
