"""Orthogonalized Bipartite Sparse Autoencoder (OB-SAE), proposal Sec. 4.1-4.2.

A ReLU SAE whose dictionary is split in two, D = [D_d  D_p]: domain features and persona features.
    z = [z_d; z_p] = ReLU(W_enc (h - b_dec) + b_enc)          h_hat = D_d z_d + D_p z_p + b_dec
    L = L_recon + lambda_s * L_sparse + lambda_o * ||D_d^T D_p||_F^2                (Eq. 5, 6)
Decoder columns are kept unit-norm, so the orthogonality term is a sum of squared cosines between
domain atoms and persona atoms, and the L1 penalty cannot be dodged by shrinking the features.

Two activation streams train it: ordinary ("domain") activations and persona-expressing
("behavioral") ones. How the streams use the two halves is the `routing` option:
  mask   the domain stream may only use z_d; the behavioral stream uses everything. D_p can then
         only earn its keep on what D_d cannot explain, which is what a persona subspace should be.
         With `detach_domain`, the behavioral loss also does not move D_d, so D_d is learned from
         the domain stream alone and D_p from the residual.
  split  each stream may only use its own half: domain -> z_d, behavioral -> z_p.
  none   no restriction: both streams reconstruct from the whole dictionary (the literal reading of
         the proposal's Figure 2, where the streams differ only in the data).
  paired the two streams are the same tokens read under a careful and a harmful system prompt
         (paired_streams.py), row i of one matching row i of the other. The domain stream is reconstructed
         from z_d alone. The persona half reconstructs what the harmful framing adds to the activation,
         h_harmful - h_careful, from z_p of the harmful reading, and must stay silent (reconstruct zero) on
         the careful reading. So D_d learns the content and D_p learns only the persona framing.

The orthogonality term (`orth_mode`):
  dict   Eq. 6, ||D_d^T D_p||_F^2 = sum_k d_k^T (D_d D_d^T) d_k over persona atoms d_k: overlap with the domain
         *dictionary*, every domain atom counting the same.
  data   sum_k d_k^T S d_k, with S the second moment of the domain-stream activations, scaled so an average
         direction counts 1: overlap with the domain *data*, each direction weighted by how much ordinary content
         actually uses it. Same form as Eq. 6 with D_d D_d^T replaced by S. Pointing a persona atom along a direction
         normal text barely uses costs little; along a busy one (topic, style, phrasing) it costs a lot.

The persona subspace is the span of the (alive) persona atoms, as an orthonormal basis U_p (Eq. 7).
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HP:
    n_domain: int = 2048
    n_persona: int = 128
    l1: float = 5.0            # lambda_s; inputs are normalised so that E||h||^2 = d
    orth: float = 1.0          # lambda_o
    orth_mode: str = "dict"    # "dict" (Eq. 6) or "data" (weighted by domain-stream usage), see module docstring
    lr: float = 2e-4
    steps: int = 20000
    batch: int = 4096          # tokens per stream per step
    routing: str = "mask"      # "mask", "split", "none" or "paired", see module docstring
    detach_domain: bool = True
    l1_warmup: float = 0.1     # fraction of steps over which lambda_s ramps up from 0
    seed: int = 0
    l1_persona: float = -1.0   # lambda_s for the persona half under "paired" routing; < 0 means same as l1


class OBSAE(nn.Module):
    def __init__(self, d: int, n_domain: int, n_persona: int, seed: int = 0):
        super().__init__()
        self.d, self.n_domain, self.n_persona = d, n_domain, n_persona
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(d, n_domain + n_persona, generator=g)
        W = W / W.norm(dim=0, keepdim=True)
        self.W_dec = nn.Parameter(W)                       # [d, n], unit-norm columns
        self.W_enc = nn.Parameter(W.T.clone())             # [n, d], starts as the decoder transpose
        self.b_enc = nn.Parameter(torch.zeros(n_domain + n_persona))
        self.b_dec = nn.Parameter(torch.zeros(d))

    @property
    def D_d(self) -> torch.Tensor:
        return self.W_dec[:, : self.n_domain]

    @property
    def D_p(self) -> torch.Tensor:
        return self.W_dec[:, self.n_domain:]

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec /= self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return F.relu((h - self.b_dec) @ self.W_enc.T + self.b_enc)

    def forward(self, h: torch.Tensor, persona: bool, routing: str = "mask", detach_domain: bool = True):
        """Reconstruct a batch from one stream (`persona` = it is the behavioral stream)."""
        z = self.encode(h)
        z_d, z_p = z[:, : self.n_domain], z[:, self.n_domain:]
        if routing == "paired":
            if persona:                        # the persona half alone, reconstructing a difference: no bias
                return z_p @ self.D_p.T, torch.zeros_like(z_d), z_p
            return z_d @ self.D_d.T + self.b_dec, z_d, torch.zeros_like(z_p)
        if routing == "none":
            pass
        elif routing == "mask":
            if not persona:
                z_p = torch.zeros_like(z_p)
        elif routing == "split":
            if persona:
                z_d = torch.zeros_like(z_d)
            else:
                z_p = torch.zeros_like(z_p)
        else:
            raise ValueError(f"unknown routing {routing!r}")
        D_d = self.D_d.detach() if (detach_domain and persona and routing == "mask") else self.D_d
        return z_d @ D_d.T + z_p @ self.D_p.T + self.b_dec, z_d, z_p

    def terms(self, h: torch.Tensor, persona: bool, hp: HP) -> dict:
        h_hat, z_d, z_p = self(h, persona, hp.routing, hp.detach_domain)
        err = (h - h_hat).pow(2).sum(-1)
        with torch.no_grad():
            fvu = err.sum() / (h - h.mean(0)).pow(2).sum()              # fraction of variance unexplained
            l0 = ((z_d > 0).sum(-1) + (z_p > 0).sum(-1)).float().mean()
        return {"recon": err.mean(), "l1": (z_d.sum(-1) + z_p.sum(-1)).mean(), "fvu": fvu, "l0": l0}

    def paired_terms(self, h_care: torch.Tensor, h_harm: torch.Tensor, hp: HP) -> tuple[dict, dict]:
        """Losses for "paired" routing on matching rows of the two streams (careful, harmful readings)."""
        td = self.terms(h_care, False, hp)
        delta = h_harm - h_care
        d_hat, _, z_harm = self(h_harm, True, "paired")
        quiet, _, z_care = self(h_care, True, "paired")
        err = (delta - d_hat).pow(2).sum(-1)
        with torch.no_grad():
            fvu = err.sum() / delta.pow(2).sum()          # share of the framing difference left unexplained
            l0 = (z_harm > 0).sum(-1).float().mean()
        tb = {"recon": err.mean() + quiet.pow(2).sum(-1).mean(),
              "l1": z_harm.sum(-1).mean() + z_care.sum(-1).mean(), "fvu": fvu, "l0": l0}
        return td, tb

    def orth_loss(self, content: torch.Tensor | None = None) -> torch.Tensor:
        """||D_d^T D_p||_F^2 (Eq. 6), or with `content` = S [d, d], sum_k d_k^T S d_k ("data" mode)."""
        if content is None:
            return (self.D_d.T @ self.D_p).pow(2).sum()
        return (self.D_p * (content @ self.D_p)).sum()

    @torch.no_grad()
    def firing_rate(self, h: torch.Tensor, persona: bool, hp: HP, batch: int = 8192) -> torch.Tensor:
        """Fraction of tokens on which each persona latent is active (under the training routing)."""
        on = torch.zeros(self.n_persona)
        for i in range(0, len(h), batch):
            _, _, z_p = self(h[i: i + batch].to(self.b_dec.device).float(), persona, hp.routing, hp.detach_domain)
            on += (z_p > 0).float().sum(0).cpu()
        return on / len(h)

    @torch.no_grad()
    def persona_basis(self, alive: torch.Tensor | None = None, tol: float = 1e-3) -> torch.Tensor:
        """Orthonormal basis U_p [d, r] of the span of the persona atoms (Eq. 7), optionally only the
        `alive` ones. Same subspace as the QR factor of D_p; computed by SVD because that drops
        directions D_p does not actually span, where an unpivoted QR would invent arbitrary ones."""
        D_p = self.D_p if alive is None else self.D_p[:, alive.to(self.D_p.device)]
        return orthonormal_basis(D_p, tol)

    @torch.no_grad()
    def domain_basis(self, tol: float = 1e-3) -> torch.Tensor:
        return orthonormal_basis(self.D_d, tol)

    def save(self, path: Path, **extra) -> None:
        torch.save({"d": self.d, "n_domain": self.n_domain, "n_persona": self.n_persona,
                    "state": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: Path, device="cpu") -> tuple["OBSAE", dict]:
        blob = torch.load(path, map_location=device, weights_only=False)
        sae = cls(blob["d"], blob["n_domain"], blob["n_persona"]).to(device)
        sae.load_state_dict(blob["state"])
        return sae, blob


def orthonormal_basis(M: torch.Tensor, tol: float = 1e-3) -> torch.Tensor:
    """Orthonormal basis of the column span of M, dropping singular values below tol * the largest."""
    U, S, _ = torch.linalg.svd(M.float(), full_matrices=False)
    return U[:, : int((S > tol * S[0]).sum())]


def subspace_overlap(U_a: torch.Tensor, U_b: torch.Tensor) -> torch.Tensor:
    """Cosines of the principal angles between two subspaces (1 = shared direction, 0 = orthogonal)."""
    return torch.linalg.svdvals(U_a.T @ U_b)


def fit(sae: OBSAE, dom: torch.Tensor, beh: torch.Tensor, hp: HP, log=print, log_every: int = 500) -> list[dict]:
    """Train on two activation tensors [N, d] (already normalised). Returns the logged history."""
    dev = sae.b_dec.device
    g = torch.Generator().manual_seed(hp.seed)
    with torch.no_grad():
        sae.b_dec.copy_(dom[: 65536].float().mean(0))
    opt = torch.optim.Adam(sae.parameters(), lr=hp.lr, betas=(0.9, 0.999))
    history = []
    paired = hp.routing == "paired"
    if paired and len(dom) != len(beh):
        raise ValueError("paired routing needs row-matched streams (paired_streams.py)")
    l1_p = hp.l1 if hp.l1_persona < 0 else hp.l1_persona
    content = content_moment(dom, dev) if hp.orth_mode == "data" else None
    if hp.orth_mode not in ("dict", "data"):
        raise ValueError(f"unknown orth_mode {hp.orth_mode!r}")
    for step in range(1, hp.steps + 1):
        ramp = min(1.0, step / max(1, hp.l1_warmup * hp.steps))
        idx = torch.randint(len(dom), (hp.batch,), generator=g)
        hd = dom[idx].to(dev).float()
        hb = beh[idx if paired else torch.randint(len(beh), (hp.batch,), generator=g)].to(dev).float()
        if paired:
            td, tb = sae.paired_terms(hd, hb, hp)
            l1_term = ramp * (hp.l1 * td["l1"] + l1_p * tb["l1"])
        else:
            td, tb = sae.terms(hd, False, hp), sae.terms(hb, True, hp)
            l1_term = ramp * hp.l1 * (td["l1"] + tb["l1"])
        orth = sae.orth_loss(content) if hp.orth != 0 else torch.zeros((), device=dev)   # skip the cost when unused
        loss = td["recon"] + tb["recon"] + l1_term + hp.orth * orth
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.normalize_decoder()
        if step % log_every == 0 or step == hp.steps:
            row = {"step": step, "loss": loss.item(), "orth": (orth if hp.orth != 0 else sae.orth_loss(content).detach()).item(),   # the real overlap, even when unpenalised
                   "fvu_dom": td["fvu"].item(), "fvu_beh": tb["fvu"].item(),
                   "l0_dom": td["l0"].item(), "l0_beh": tb["l0"].item()}
            history.append(row)
            log("[sae] " + "  ".join(f"{k}={v:.4g}" if k != "step" else f"step={v}" for k, v in row.items()))
    return history


@torch.no_grad()
def content_moment(acts: torch.Tensor, device, n: int = 200_000, chunk: int = 20_000) -> torch.Tensor:
    """Second moment E[h h^T] of (up to n rows of) the domain stream, scaled so its trace is d: a direction
    ordinary content uses an average amount gets weight 1."""
    rows = acts[torch.randperm(len(acts), generator=torch.Generator().manual_seed(0))[:n]]
    S = torch.zeros(acts.shape[1], acts.shape[1], device=device)
    for i in range(0, len(rows), chunk):
        h = rows[i: i + chunk].to(device).float()
        S += h.T @ h
    S /= len(rows)
    return S * (S.shape[0] / torch.trace(S))


def hp_dict(hp: HP) -> dict:
    return asdict(hp)
