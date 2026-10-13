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

  topk   sum_k ||V^T d_k||^2, with V the top `orth_k` principal directions of the domain-stream activations: overlap
         with the *content subspace*, the low-dimensional subspace that holds most of the energy of ordinary text.
         Unlike "dict" (whose dictionary spans the whole space once it has d atoms, so nothing is orthogonal to it)
         this is well-posed: the complement of V has d - orth_k dimensions, the penalty can reach exactly zero, and
         its value / n_persona is the share of each persona atom lying inside the content subspace. `orth_center`
         takes the principal directions of the centred activations (content variation only, the mean direction is
         then not counted as content). `orth_hard` additionally projects D_p onto the complement after every step,
         so the two subspaces are exactly orthogonal by construction.

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
    orth_mode: str = "dict"    # "dict" (Eq. 6), "data" (weighted by domain-stream usage) or "topk", see docstring
    orth_k: int = 128          # "topk": dimension of the content subspace the persona atoms must avoid
    orth_center: bool = False  # "topk": principal directions of the centred activations (mean not counted as content)
    orth_hard: bool = False    # "topk": project D_p onto the complement after every step (exact orthogonality)
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
            # ||D_d^T D_p||_F^2 = sum_k d_k^T (D_d D_d^T) d_k: the same number through a d x d matrix, so a
            # 50k x 50k dictionary does not need a 50k x 50k product
            content = self.D_d @ self.D_d.T
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


@torch.no_grad()
def separation_probe(sae: OBSAE, care: torch.Tensor, harm: torch.Tensor, chunk: int = 1024) -> dict:
    """Are persona and content separated? Measured on held-out paired rows (the same tokens read under a careful
    and a harmful system prompt), with delta = harm - care the pure persona effect:
      delta_by_persona   share of delta explained by the persona half, D_p (z_p(harm) - z_p(care))   want high
      delta_by_domain    share of delta explained by the domain half,  D_d (z_d(harm) - z_d(care))   want low
      persona_on_plain   energy the persona half writes on the careful reading, relative to the content  want ~0
      l0_persona_harm/care  persona latents active per token under each reading             want harm >> care
      alive_domain/persona  latents that fired at least once on the probe (dead-dictionary check)
      atom_overlap       mean squared cosine between domain and persona atoms, in units of the random level 1/d
      cos_mean_delta     cosine between the mean persona-half reconstruction of delta and the mean delta"""
    dev, n = sae.b_dec.device, sae.n_domain
    e_delta = e_per = e_dom = e_plain = e_content = 0.0
    l0_h = l0_c = 0.0
    fired = torch.zeros(sae.n_domain + sae.n_persona, dtype=torch.bool, device=dev)
    mean_delta, mean_rec = torch.zeros(sae.d, device=dev), torch.zeros(sae.d, device=dev)
    for i in range(0, len(care), chunk):
        c, h = care[i: i + chunk].to(dev).float(), harm[i: i + chunk].to(dev).float()
        zc, zh = sae.encode(c), sae.encode(h)
        delta = h - c
        rec_p = (zh[:, n:] - zc[:, n:]) @ sae.D_p.T
        rec_d = (zh[:, :n] - zc[:, :n]) @ sae.D_d.T
        e_delta += delta.pow(2).sum().item()
        e_per += (delta - rec_p).pow(2).sum().item()
        e_dom += (delta - rec_d).pow(2).sum().item()
        e_plain += (zc[:, n:] @ sae.D_p.T).pow(2).sum().item()
        e_content += (c - sae.b_dec).pow(2).sum().item()
        l0_h += (zh[:, n:] > 0).sum().item()
        l0_c += (zc[:, n:] > 0).sum().item()
        fired |= (zc > 0).any(0) | (zh > 0).any(0)
        mean_delta += delta.sum(0)
        mean_rec += rec_p.sum(0)
    cos = F.cosine_similarity(mean_delta, mean_rec, dim=0).item() if mean_rec.norm() > 0 else 0.0
    overlap = sae.orth_loss().item() / (sae.n_domain * sae.n_persona) * sae.d
    return {"delta_by_persona": 1 - e_per / e_delta, "delta_by_domain": 1 - e_dom / e_delta,
            "persona_on_plain": e_plain / e_content, "l0_persona_harm": l0_h / len(care),
            "l0_persona_care": l0_c / len(care), "alive_domain": int(fired[:n].sum()),
            "alive_persona": int(fired[n:].sum()), "atom_overlap": overlap, "cos_mean_delta": cos}


def fit(sae: OBSAE, dom: torch.Tensor, beh: torch.Tensor, hp: HP, log=print, log_every: int = 500,
        probe: tuple[torch.Tensor, torch.Tensor] | None = None) -> list[dict]:
    """Train on two activation tensors [N, d] (already normalised). Returns the logged history. `probe` =
    (careful, harmful) held-out paired rows: logs separation_probe() on them at every log step."""
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
    if hp.orth_mode == "topk":
        content, held = content_projector(dom, dev, hp.orth_k, hp.orth_center)
        log(f"[orth] content subspace: top {hp.orth_k} principal directions of the "
            f"{'centred ' if hp.orth_center else ''}domain activations, holding {100 * held:.1f}% of their energy"
            f"{'; D_p is projected onto its complement every step' if hp.orth_hard else ''}")
    if hp.orth_mode not in ("dict", "data", "topk"):
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
        if hp.orth_mode == "topk" and hp.orth_hard:
            with torch.no_grad():
                sae.W_dec[:, sae.n_domain:] -= content @ sae.W_dec[:, sae.n_domain:]
        sae.normalize_decoder()
        if step % log_every == 0 or step == hp.steps:
            row = {"step": step, "loss": loss.item(), "orth": (orth if hp.orth != 0 else sae.orth_loss(content).detach()).item(),   # the real overlap, even when unpenalised
                   "fvu_dom": td["fvu"].item(), "fvu_beh": tb["fvu"].item(),
                   "l0_dom": td["l0"].item(), "l0_beh": tb["l0"].item()}
            log("[sae] " + "  ".join(f"{k}={v:.4g}" if k != "step" else f"step={v}" for k, v in row.items()))
            if hp.orth_mode == "topk":
                inside = sae.orth_loss(content).item() / sae.n_persona
                row["persona_in_content"] = inside
                log(f"[orth] persona atoms lie {100 * inside:.3f}% inside the content subspace "
                    f"(random atoms: {100 * hp.orth_k / sae.d:.1f}%)")
            if probe is not None:
                p = separation_probe(sae, *probe)
                row["probe"] = p
                log(f"[probe] delta explained by persona half {100 * p['delta_by_persona']:.0f}% / by domain half "
                    f"{100 * p['delta_by_domain']:.0f}%  |  persona half on plain text "
                    f"{100 * p['persona_on_plain']:.2f}% of content energy  |  persona L0 harmful "
                    f"{p['l0_persona_harm']:.1f} vs careful {p['l0_persona_care']:.1f}  |  alive: domain "
                    f"{p['alive_domain']:,}/{sae.n_domain:,}, persona {p['alive_persona']:,}/{sae.n_persona:,}  |  "
                    f"atom overlap {p['atom_overlap']:.2f}x random  |  cos(persona mean, mean delta) "
                    f"{p['cos_mean_delta']:.2f}")
            history.append(row)
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


@torch.no_grad()
def content_projector(acts: torch.Tensor, device, k: int, center: bool = False, n: int = 200_000,
                      chunk: int = 20_000) -> tuple[torch.Tensor, float]:
    """P = V V^T [d, d], the projector onto the top-k principal directions of (up to n rows of) the domain stream,
    and the share of the stream's energy those directions hold. Uncentred by default, so the mean activation (the
    single largest direction) counts as content."""
    rows = acts[torch.randperm(len(acts), generator=torch.Generator().manual_seed(0))[:n]]
    d = acts.shape[1]
    S, mu = torch.zeros(d, d, device=device, dtype=torch.float64), torch.zeros(d, device=device, dtype=torch.float64)
    for i in range(0, len(rows), chunk):
        h = rows[i: i + chunk].to(device).double()
        S += h.T @ h
        mu += h.sum(0)
    S, mu = S / len(rows), mu / len(rows)
    if center:
        S -= torch.outer(mu, mu)
    ev, V = torch.linalg.eigh(S)
    V = V[:, -k:]
    return (V @ V.T).float(), (ev[-k:].sum() / ev.sum()).item()


def hp_dict(hp: HP) -> dict:
    return asdict(hp)
