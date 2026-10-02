"""CPU tests for the OB-SAE maths. No model, no GPU, no data: run with

    python ob_sae/tests/test_obsae.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from common import Projection, capture, project_out  # noqa: E402
from obsae import HP, OBSAE, fit, orthonormal_basis  # noqa: E402

torch.set_num_threads(4)


def test_decoder_unit_norm():
    sae = OBSAE(32, 20, 6)
    assert torch.allclose(sae.W_dec.norm(dim=0), torch.ones(26), atol=1e-6)
    with torch.no_grad():
        sae.W_dec *= 3.0
    sae.normalize_decoder()
    assert torch.allclose(sae.W_dec.norm(dim=0), torch.ones(26), atol=1e-6)


def test_orth_loss_is_sum_of_squared_cosines():
    sae = OBSAE(16, 8, 4)
    cos = sae.D_d.T @ sae.D_p                              # unit columns -> cosines
    assert torch.allclose(sae.orth_loss(), cos.pow(2).sum())
    Q, _ = torch.linalg.qr(torch.randn(16, 12))            # block-orthogonal dictionaries -> zero loss
    with torch.no_grad():
        sae.W_dec.copy_(Q)
    assert sae.orth_loss().item() < 1e-10


def test_mask_routing_gradients():
    sae, hp = OBSAE(16, 8, 4), HP(routing="mask", detach_domain=True)
    h = torch.randn(64, 16)
    sae.zero_grad()
    sae.terms(h, persona=False, hp=hp)["recon"].backward()  # domain stream: persona half must be untouched
    assert sae.W_dec.grad[:, 8:].abs().sum() == 0 and sae.W_enc.grad[8:].abs().sum() == 0
    assert sae.W_dec.grad[:, :8].abs().sum() > 0
    sae.zero_grad()
    sae.terms(h, persona=True, hp=hp)["recon"].backward()   # behavioral stream, detach_domain: D_d frozen
    assert sae.W_dec.grad[:, :8].abs().sum() == 0 and sae.W_dec.grad[:, 8:].abs().sum() > 0
    sae.zero_grad()
    sae.terms(h, persona=True, hp=HP(detach_domain=False))["recon"].backward()
    assert sae.W_dec.grad[:, :8].abs().sum() > 0


def test_split_routing_uses_own_half_only():
    sae, hp = OBSAE(16, 8, 4), HP(routing="split")
    h = torch.randn(32, 16)
    _, z_d, z_p = sae(h, persona=False, routing="split")
    assert z_p.abs().sum() == 0 and z_d.abs().sum() > 0
    _, z_d, z_p = sae(h, persona=True, routing="split")
    assert z_d.abs().sum() == 0 and z_p.abs().sum() > 0


def test_none_routing_uses_the_whole_dictionary_on_both_streams():
    sae, hp = OBSAE(16, 8, 4), HP(routing="none")
    h = torch.randn(64, 16)
    for persona in (False, True):
        _, z_d, z_p = sae(h, persona=persona, routing="none")
        assert z_d.abs().sum() > 0 and z_p.abs().sum() > 0
    sae.zero_grad()
    sae.terms(h, persona=False, hp=hp)["recon"].backward()  # even the domain stream now trains the persona half
    assert sae.W_dec.grad[:, 8:].abs().sum() > 0


def test_persona_basis_is_orthonormal_and_spans_the_atoms():
    sae = OBSAE(32, 10, 5)
    U = sae.persona_basis()
    assert U.shape == (32, 5) and torch.allclose(U.T @ U, torch.eye(5), atol=1e-5)
    assert torch.allclose(U @ (U.T @ sae.D_p), sae.D_p, atol=1e-5)            # span(U) contains every atom
    with torch.no_grad():                                                       # rank-deficient: 2 duplicated atoms
        sae.W_dec[:, 12] = sae.W_dec[:, 10]
        sae.W_dec[:, 13] = sae.W_dec[:, 11]
    assert sae.persona_basis().shape[1] == 3
    alive = torch.tensor([True, False, True, False, False])
    assert sae.persona_basis(alive).shape[1] == 1                               # atoms 10 and 12 coincide


def test_clamping_holds_the_subspace_at_the_base_value_with_the_same_gradient():
    d, r = 24, 5
    U = orthonormal_basis(torch.randn(d, r))
    h, base = torch.randn(7, d, requires_grad=True), torch.randn(7, d)
    out = project_out(h, U, base)
    assert torch.allclose(out @ U, base @ U, atol=1e-5)                         # span(U) part = the base model's
    P = torch.eye(d) - U @ U.T
    assert torch.allclose(out @ P, h.detach() @ P, atol=1e-5)                   # the rest = the finetuned model's
    w = torch.randn(d)
    (out * w).sum().backward()
    assert torch.allclose(h.grad, (P @ w).expand(7, d), atol=1e-5)              # same gradient as projecting to zero


def test_projection_removes_subspace_and_projects_gradients():
    d, r = 24, 5
    U = orthonormal_basis(torch.randn(d, r))
    h = torch.randn(7, d, requires_grad=True)
    hp_ = project_out(h, U)
    assert (hp_ @ U).abs().max() < 1e-5                                         # U^T h' = 0     (Eq. 10)
    assert torch.allclose(project_out(hp_, U), hp_, atol=1e-5)                  # idempotent
    w = torch.randn(d)
    loss = ((hp_ * w).sum())
    loss.backward()
    P = torch.eye(d) - U @ U.T
    assert torch.allclose(h.grad, (P @ w).expand(7, d), atol=1e-5)              # grad_h L = P grad_h' L (Eq. 11)


def test_projection_hook_on_a_toy_model():
    class Layer(torch.nn.Module):
        def forward(self, x):
            return (x * 2.0,)                                                   # transformers decoder layers return tuples

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Layer(), Layer()])
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            for layer in self.model.layers:
                x = layer(x)[0]
            return x

    m, U = Toy(), orthonormal_basis(torch.randn(8, 2))
    x = torch.randn(3, 8)
    plain = m(x)
    proj = Projection(m, 0, U)
    out = m(x)
    assert torch.allclose(out, 2 * project_out(2 * x, U), atol=1e-5)
    with capture(m, 0) as box:                                                  # a hook added later sees the projected output
        m(x)
    assert (box["h"] @ U).abs().max() < 1e-5
    proj.enabled = False
    assert torch.allclose(m(x), plain)
    proj.remove()
    assert torch.allclose(m(x), plain)


def _planted(d=64, n_dom=24, n_pers=4, n=20000, seed=0):
    """Domain activations = sparse mixtures of `n_dom` directions; behavioral = the same plus
    1-2 of `n_pers` persona directions that are exactly orthogonal to the domain span."""
    g = torch.Generator().manual_seed(seed)
    Qd = orthonormal_basis(torch.randn(d, n_dom, generator=g))
    R = torch.randn(d, n_pers, generator=g)
    Qp = orthonormal_basis(R - Qd @ (Qd.T @ R))

    def mix(Q, k, n):
        idx = torch.stack([torch.randperm(Q.shape[1], generator=g)[:k] for _ in range(n)])
        coef = 1 + torch.rand(n, k, generator=g)
        return torch.einsum("nk,nkd->nd", coef, Q.T[idx])

    dom = mix(Qd, 3, n) + 0.02 * torch.randn(n, d, generator=g)
    beh = mix(Qd, 3, n) + mix(Qp, 2, n) + 0.02 * torch.randn(n, d, generator=g)
    s = d ** 0.5 / dom.norm(dim=1).mean()                                       # normalise as train_sae.py does
    return dom * s, beh * s, Qd, Qp


def _train_planted(**overrides):
    """Train a small OB-SAE on the planted data; return (share of the planted persona span found,
    share of the found basis lying in the domain span)."""
    dom, beh, Qd, Qp = _planted()
    hp = HP(**{**dict(n_domain=32, n_persona=6, l1=1.0, orth=5.0, lr=1e-3, steps=3000, batch=512,
                      l1_warmup=0.2), **overrides})
    torch.manual_seed(0)
    sae = OBSAE(64, hp.n_domain, hp.n_persona)
    fit(sae, dom, beh, hp, log=lambda s: None)
    U = sae.persona_basis(sae.firing_rate(beh, True, hp) > 0.01, tol=0.05)
    captured = ((U.T @ Qp).pow(2).sum() / Qp.shape[1]).item()
    leaked = ((U.T @ Qd).pow(2).sum() / U.shape[1]).item()
    print(f"    {overrides or 'defaults'}: rank {U.shape[1]}, captured {captured:.2f}, leaked {leaked:.3f}")
    return captured, leaked


def test_recovers_planted_persona_subspace():
    captured, leaked = _train_planted()
    assert captured > 0.9 and leaked < 0.05, (captured, leaked)


def test_orthogonality_penalty_reduces_leakage():
    _, leaked_with = _train_planted(orth=5.0)
    _, leaked_without = _train_planted(orth=0.0)
    assert leaked_without > 3 * leaked_with, (leaked_with, leaked_without)


def test_stopping_the_domain_gradient_keeps_persona_out_of_the_domain_half():
    """Without it the behavioral loss lets D_d absorb the persona directions, leaving D_p empty-handed."""
    with_stop, _ = _train_planted(detach_domain=True)
    without, _ = _train_planted(detach_domain=False)
    assert with_stop - without > 0.2, (with_stop, without)


def _train_paired(routing, d=64, n=20000, **overrides):
    """Row-matched streams as paired_streams.py makes them: the same content read twice, the harmful reading
    adding one of 4 persona directions (random, so partly inside the domain span, as on a real model) at a
    tenth of the content's size. Returns the share of the planted persona span found."""
    g = torch.Generator().manual_seed(1)
    Qd = orthonormal_basis(torch.randn(d, 24, generator=g))
    Qp = orthonormal_basis(torch.randn(d, 4, generator=g))
    idx = torch.stack([torch.randperm(24, generator=g)[:3] for _ in range(n)])
    content = torch.einsum("nk,nkd->nd", 1 + torch.rand(n, 3, generator=g), Qd.T[idx])
    persona = Qp.T[torch.randint(4, (n,), generator=g)] * 0.15 * (1 + torch.rand(n, 1, generator=g))
    care = content + 0.02 * torch.randn(n, d, generator=g)
    harm = care + persona
    s = d ** 0.5 / care.norm(dim=1).mean()
    hp = HP(**{**dict(n_domain=32, n_persona=6, l1=1.0, orth=5.0, lr=1e-3, steps=3000, batch=512,
                      l1_warmup=0.2, routing=routing), **overrides})
    torch.manual_seed(0)
    sae = OBSAE(d, hp.n_domain, hp.n_persona)
    fit(sae, care * s, harm * s, hp, log=lambda s: None)
    U = sae.persona_basis(sae.firing_rate(harm * s, True, hp) > 0.01, tol=0.05)
    captured = ((U.T @ Qp).pow(2).sum() / Qp.shape[1]).item()
    print(f"    {routing} {overrides}: rank {U.shape[1]}, captured {captured:.2f}")
    return captured


def test_paired_routing_finds_the_framing_difference():
    """On token-matched streams the paired routing recovers the persona directions; the mask routing, which
    has to model them as leftover reconstruction error, does much worse."""
    paired, mask = _train_paired("paired", l1_persona=0.1, orth=0.0), _train_paired("mask", orth=0.0)
    assert paired > 0.6 and paired - mask > 0.2, (paired, mask)


def test_orthogonality_penalty_blocks_persona_directions_that_overlap_the_domain_span():
    """When the persona directions are not orthogonal to the content (here they are random), Eq. 6 pushes
    the persona half away from them: the stronger the penalty, the less of them it finds."""
    free, strong = (_train_paired("paired", l1_persona=0.1, orth=o) for o in (0.0, 50.0))
    assert free - strong > 0.3, (free, strong)


def test_data_orth_loss_generalises_eq6():
    """With S = D_d D_d^T the data-weighted term is exactly Eq. 6; with S = I it is just the number of atoms."""
    sae = OBSAE(32, 10, 5)
    assert torch.allclose(sae.orth_loss(sae.D_d @ sae.D_d.T), sae.orth_loss(), rtol=1e-5)
    assert torch.allclose(sae.orth_loss(torch.eye(32)), torch.tensor(5.0), rtol=1e-5)


def _train_busy_quiet(orth, d=64, n=20000):
    """Paired streams whose harmful reading adds one of 4 persona directions: 2 along directions ordinary content
    uses heavily ("busy"), 2 along directions it never uses ("quiet"). Returns the share of each pair found."""
    g = torch.Generator().manual_seed(2)
    Qd = orthonormal_basis(torch.randn(d, 24, generator=g))
    R = torch.randn(d, 2, generator=g)
    quiet, busy = orthonormal_basis(R - Qd @ (Qd.T @ R)), Qd[:, :2]
    Qp = torch.cat([busy, quiet], 1)
    idx = torch.stack([torch.randperm(24, generator=g)[:3] for _ in range(n)])
    content = torch.einsum("nk,nkd->nd", 1 + torch.rand(n, 3, generator=g), Qd.T[idx])
    care = content + 0.02 * torch.randn(n, d, generator=g)
    harm = care + Qp.T[torch.randint(4, (n,), generator=g)] * 0.6 * (1 + torch.rand(n, 1, generator=g))
    s = d ** 0.5 / care.norm(dim=1).mean()
    hp = HP(n_domain=32, n_persona=6, l1=1.0, orth=orth, orth_mode="data", lr=1e-3, steps=3000, batch=512,
            l1_warmup=0.2, routing="paired", l1_persona=0.1)
    torch.manual_seed(0)
    sae = OBSAE(d, 32, 6)
    fit(sae, care * s, harm * s, hp, log=lambda s: None)
    U = sae.persona_basis(sae.firing_rate(harm * s, True, hp) > 0.01, tol=0.05)
    found = lambda Q: ((U.T @ Q).pow(2).sum() / Q.shape[1]).item()
    print(f"    data orth {orth}: rank {U.shape[1]}, busy {found(busy):.2f}, quiet {found(quiet):.2f}")
    return found(busy), found(quiet)


def test_data_orth_drops_busy_persona_directions_and_keeps_quiet_ones():
    busy0, quiet0 = _train_busy_quiet(0.0)
    busy1, quiet1 = _train_busy_quiet(0.3)
    assert busy0 > 0.9 and quiet0 > 0.9, (busy0, quiet0)
    assert busy1 < 0.2 and quiet1 > 0.9, (busy1, quiet1)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
