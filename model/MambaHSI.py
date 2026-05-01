import torch
from torch import nn

from model.fuzzy_modules import (
    FSpaMB,
    FSpeMB,
    TSSFM,
)


class FuzzyEncoderBlock(nn.Module):
    def __init__(self, d_model, num_groups, m_dim, group_num=4,
                 r_spa=4, r_fus=4, d_prime=32, d_z=32, use_residual=True):
        super().__init__()
        self.num_groups = num_groups
        self.m_dim = m_dim

        self.fspa = FSpaMB(
            d_model=d_model,
            num_rules=r_spa,
            d_prime=d_prime,
            group_num=group_num,
            use_residual=use_residual,
        )
        self.fspe = FSpeMB(
            num_groups=num_groups,
            m_dim=m_dim,
            d_out=d_model,
            group_num=group_num,
            use_residual=use_residual,
        )
        self.tssfm = TSSFM(
            d_model=d_model,
            num_rules=r_fus,
            d_z=d_z,
            use_warmup=True,
        )

    def set_warmup(self, lam):
        self.fspa.set_warmup(lam)
        self.tssfm.set_warmup(lam)

    def forward(self, h_flat, phi_bar):
        B, D, H, W = h_flat.shape
        h_grouped = h_flat.view(B, self.num_groups, self.m_dim, H, W)

        h_spa = self.fspa(h_flat)
        h_spe = self.fspe(h_grouped, phi_bar)
        return self.tssfm(h_spa, h_spe)


class MambaHSI(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=128, num_classes=10,
                 use_residual=True, group_num=4,
                 G=8, R_spa=4, R_f=4, D_prime=32, D_z=32, n_enc=3,
                 mamba_type='both', token_num=4, use_att=True):
        super().__init__()
        del mamba_type, token_num, use_att  # legacy kwargs accepted but unused

        assert hidden_dim % G == 0, "hidden_dim must be divisible by G"
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_groups = G
        self.m_dim = hidden_dim // G

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([
            FuzzyEncoderBlock(
                d_model=hidden_dim,
                num_groups=G,
                m_dim=self.m_dim,
                group_num=group_num,
                r_spa=R_spa,
                r_fus=R_f,
                d_prime=D_prime,
                d_z=D_z,
                use_residual=use_residual,
            )
            for _ in range(n_enc)
        ])

        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=1),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def set_warmup(self, lam):
        for block in self.blocks:
            block.set_warmup(lam)

    def fuzzy_centers(self):
        centers = []
        for block in self.blocks:
            centers.append(block.fspa.rule_centers)
            centers.append(block.tssfm.rule_centers)
        return centers

    def forward(self, x):
        G = self.num_groups
        phi_bar = torch.full((G,), 1.0 / G, device=x.device)

        h = self.patch_embedding(x)

        for block in self.blocks:
            h = block(h, phi_bar)

        return self.cls_head(h)
