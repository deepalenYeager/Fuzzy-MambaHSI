import math
import torch
from torch import nn
from torch.nn import functional as F

try:
    from mamba_ssm import Mamba2 as _Mamba2
    _MAMBA2_IMPORTED = True
except ImportError:
    _MAMBA2_IMPORTED = False


def _cuda_ext_available() -> bool:
    """Return True only when the causal_conv1d CUDA extension is present and loadable."""
    if not _MAMBA2_IMPORTED:
        return False
    try:
        import causal_conv1d_cuda  # noqa: F401
        return causal_conv1d_cuda is not None
    except Exception:
        return False


_USE_MAMBA2 = _cuda_ext_available()


class _PureTorchMamba(nn.Module):
    """Pure-PyTorch SSM used as a drop-in replacement for Mamba2 on CPU or when
    the causal_conv1d CUDA extension is unavailable."""

    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, headdim=64, ngroups=1, **_):
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv,
                                padding=d_conv - 1, groups=d_inner, bias=True)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = self.act(x_conv)
        out = self.norm(x_conv * self.act(z))
        return self.out_proj(out)


def _make_mamba(d_model, d_state, d_conv, expand, headdim, ngroups):
    """Return a Mamba2 instance when CUDA extensions are available, else _PureTorchMamba."""
    if _USE_MAMBA2:
        return _Mamba2(
            d_model=d_model, d_state=d_state, d_conv=d_conv,
            expand=expand, headdim=headdim, ngroups=ngroups,
        )
    return _PureTorchMamba(
        d_model=d_model, d_state=d_state, d_conv=d_conv,
        expand=expand, headdim=headdim, ngroups=ngroups,
    )


def _choose_mamba2_kwargs(d_model, expand=2, preferred_headdim=64, preferred_dstate=64, ngroups=1):
    """Pick (headdim, d_state) so causal_conv1d_cuda accepts the fast path.

    Two divisibility constraints from mamba_ssm + causal_conv1d:
      1. conv_dim = d_inner + 2*ngroups*d_state must be divisible by 8
         ("causal_conv1d only supports channel dimension divisible by 8").
      2. xBC stride(2) after transpose, i.e. d_in_proj = 2*d_inner +
         2*ngroups*d_state + nheads, must also be divisible by 8
         ("causal_conv1d with channel last layout requires strides ...
         multiples of 8"). Because d_inner is even and constraint 1 forces
         2*ngroups*d_state divisible by 8 (when d_inner % 8 == 0), this
         reduces to nheads divisible by 8.
    """
    inner = d_model * expand

    # Largest headdim that divides d_inner AND yields nheads divisible by 8.
    headdim = min(preferred_headdim, max(inner // 8, 1))
    while headdim > 1 and (inner % headdim != 0 or (inner // headdim) % 8 != 0):
        headdim -= 1
    if headdim < 1:
        headdim = 1

    d_state = preferred_dstate
    while d_state > 8 and d_state > inner:
        d_state //= 2

    # Round d_state up so conv_dim = inner + 2*ngroups*d_state is divisible
    # by 8 (worst case 4 bumps, since 2*ngroups*d_state moves in steps of 2).
    while (inner + 2 * ngroups * d_state) % 8 != 0:
        d_state += 1

    return headdim, d_state


class FuzzySpectralGrouping(nn.Module):
    """Raw-band Gaussian soft grouping (Eq. 8-10, 14)."""

    def __init__(self, channels, num_groups):
        super().__init__()
        self.channels = channels
        self.num_groups = num_groups

        centers = torch.linspace(0, channels - 1, num_groups)
        fwhm = 1.3 * (channels / num_groups)
        sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        log_sigmas = torch.full((num_groups,), math.log(sigma))

        self.centers = nn.Parameter(centers)
        self.log_sigmas = nn.Parameter(log_sigmas)

        band_idx = torch.arange(channels, dtype=torch.float32)
        self.register_buffer('band_idx', band_idx, persistent=False)

    def membership(self):
        sigma = torch.exp(self.log_sigmas)
        diff = self.band_idx.unsqueeze(0) - self.centers.unsqueeze(1)
        logits = -(diff ** 2) / (2.0 * sigma.unsqueeze(1) ** 2)
        return F.softmax(logits, dim=0)

    def forward(self, x):
        # x: [B, C, H, W]
        # Returning alpha (shape [G, C]) rather than a materialised
        # [B, G, C, H, W] tensor avoids an 8x memory blow-up on large images.
        alpha = self.membership()
        phi = alpha.mean(dim=1)
        phi_bar = phi / (phi.sum() + 1e-8)
        return alpha, phi_bar


class FSpaMB(nn.Module):
    """Fuzzy Spatial Mamba Block (wrap-output variant)."""

    def __init__(self, d_model, num_rules=4, d_prime=32, group_num=4, use_residual=True):
        super().__init__()
        self.d_model = d_model
        self.num_rules = num_rules
        self.d_prime = d_prime
        self.use_residual = use_residual
        self.warmup = 1.0

        headdim, d_state = _choose_mamba2_kwargs(d_model)
        self.mamba = _make_mamba(
            d_model=d_model, d_state=d_state, d_conv=4,
            expand=2, headdim=headdim, ngroups=1,
        )

        self.proj = nn.Linear(d_model, d_prime, bias=False)
        self.rule_centers = nn.Parameter(torch.randn(num_rules, d_prime) * 0.5)
        self.rule_log_sigmas = nn.Parameter(torch.zeros(num_rules, d_prime))
        self.rule_deltas = nn.Parameter(torch.zeros(num_rules, d_model))
        nn.init.normal_(self.rule_deltas, std=0.02)

        self.norm = nn.GroupNorm(group_num, d_model)
        self.act = nn.SiLU()

    def set_warmup(self, lam):
        self.warmup = float(lam)

    def _fuzzy_delta(self, hf):
        # hf: [B, L, D] -> [B, L, D]
        z = self.proj(hf)
        sigma = torch.exp(self.rule_log_sigmas)
        # [B, L, R, D']
        diff = z.unsqueeze(2) - self.rule_centers.view(1, 1, self.num_rules, self.d_prime)
        log_mu = -(diff ** 2) / (2.0 * sigma.view(1, 1, self.num_rules, self.d_prime) ** 2)
        log_f = log_mu.sum(dim=-1)
        f_bar = F.softmax(log_f, dim=-1)
        return f_bar @ self.rule_deltas

    def forward(self, x):
        # x: [B, D, H, W]
        B, D, H, W = x.shape
        hf = x.flatten(2).transpose(1, 2).contiguous()
        y_mamba = self.mamba(hf)
        y_fuzzy = self._fuzzy_delta(hf)
        y = y_mamba + self.warmup * y_fuzzy
        y = y.transpose(1, 2).reshape(B, D, H, W).contiguous()
        y = self.act(self.norm(y))
        return y + x if self.use_residual else y


class FSpeMB(nn.Module):
    """Fuzzy Spectral Mamba Block with TSK defuzzification (Eq. 13-15)."""

    def __init__(self, num_groups, m_dim, d_out, group_num=4, use_residual=True):
        super().__init__()
        self.num_groups = num_groups
        self.m_dim = m_dim
        self.d_out = d_out
        self.use_residual = use_residual

        headdim, d_state = _choose_mamba2_kwargs(m_dim, preferred_headdim=min(64, m_dim * 2))
        self.mamba = _make_mamba(
            d_model=m_dim, d_state=d_state, d_conv=4,
            expand=2, headdim=headdim, ngroups=1,
        )

        self.norm = nn.GroupNorm(1, num_groups * m_dim)
        self.act = nn.SiLU()

        self.group_up = nn.Parameter(torch.empty(num_groups, m_dim, d_out))
        self.group_bias = nn.Parameter(torch.zeros(num_groups, d_out))
        nn.init.kaiming_uniform_(self.group_up, a=math.sqrt(5))

    # Mamba2 launches a (X, batch*nchunks, nheads) Triton grid, and CUDA caps
    # the y dim at 65535. Pseudo-batch B*H*W (207k for Pavia U) overflows it
    # and Triton aborts with "invalid argument", so chunk to stay under the cap.
    _MAMBA_MAX_BATCH = 32768

    def forward(self, h_grouped, phi_bar):
        # h_grouped: [B, G, M, H, W]
        B, G, M, H, W = h_grouped.shape
        hf = h_grouped.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, G, M)
        N = hf.shape[0]
        if N <= self._MAMBA_MAX_BATCH:
            hr = self.mamba(hf)
        else:
            chunks = [self.mamba(hf[i:i + self._MAMBA_MAX_BATCH])
                      for i in range(0, N, self._MAMBA_MAX_BATCH)]
            hr = torch.cat(chunks, dim=0)
        hr_flat = hr.view(B, H, W, G * M).permute(0, 3, 1, 2).contiguous()
        hr_flat = self.act(self.norm(hr_flat))
        hr = hr_flat.view(B, G, M, H, W)

        # TSK defuzz: sum_g phi_g * (W_g * hr_g + b_g)
        # hr: [B, G, M, H, W], group_up: [G, M, D]
        out = torch.einsum('bgmhw,gmd->bgdhw', hr, self.group_up) + self.group_bias.view(1, G, self.d_out, 1, 1)
        out = (out * phi_bar.view(1, G, 1, 1, 1)).sum(dim=1)

        if self.use_residual:
            h_flat = h_grouped.reshape(B, G * M, H, W)
            out = out + h_flat
        return out


class TSSFM(nn.Module):
    """TSK Fuzzy Spatial-Spectral Fusion Module (Eq. 16-19)."""

    def __init__(self, d_model, num_rules=4, d_z=32, use_warmup=True):
        super().__init__()
        self.d_model = d_model
        self.num_rules = num_rules
        self.d_z = d_z
        self.use_warmup = use_warmup
        self.warmup = 1.0

        self.antecedent = nn.Conv2d(2 * d_model, d_z, kernel_size=1)
        self.rule_centers = nn.Parameter(torch.randn(num_rules, d_z) * 0.5)
        self.rule_log_sigmas = nn.Parameter(torch.zeros(num_rules, d_z))
        self.consequent_w = nn.Parameter(torch.empty(num_rules, d_z, d_model))
        self.consequent_b = nn.Parameter(torch.zeros(num_rules, d_model))
        nn.init.kaiming_uniform_(self.consequent_w, a=math.sqrt(5))

    def set_warmup(self, lam):
        self.warmup = float(lam)

    def forward(self, h_spa, h_spe):
        B, D, H, W = h_spa.shape
        z = self.antecedent(torch.cat([h_spa, h_spe], dim=1))
        z_tokens = z.flatten(2).transpose(1, 2).contiguous()

        sigma = torch.exp(self.rule_log_sigmas)
        diff = z_tokens.unsqueeze(2) - self.rule_centers.view(1, 1, self.num_rules, self.d_z)
        log_mu = -(diff ** 2) / (2.0 * sigma.view(1, 1, self.num_rules, self.d_z) ** 2)
        log_f = log_mu.sum(dim=-1)
        f_bar = F.softmax(log_f, dim=-1)

        # consequents: y_r = W_r z + b_r, shape [B, L, R, D]
        y_r = torch.einsum('blz,rzd->blrd', z_tokens, self.consequent_w) + self.consequent_b.view(1, 1, self.num_rules, D)
        fused = (f_bar.unsqueeze(-1) * y_r).sum(dim=2)
        fused = fused.transpose(1, 2).reshape(B, D, H, W).contiguous()

        gate = self.warmup if self.use_warmup else 1.0
        return h_spa + h_spe + gate * fused
