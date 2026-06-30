"""J3: CPU parity of the JAX core GRPO loss vs the torch ClippedPGLossFn.

Both run on CPU float32 over shared fixtures. Checks scalar-loss parity and
gradient parity (jax.value_and_grad vs torch autograd) across a config matrix
(token/seq level x IS on/off x dual-clip x kl_type x tis x force_on_policy),
plus the logprobs_from_logits helper against the torch reference.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402
torch = pytest.importorskip("torch")

from dockyard_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
from dockyard_rl.algorithms.loss.utils import calculate_kl
from dockyard_rl.models.jax import loss as jax_loss


def _fixtures(seed: int = 0, B: int = 3, S: int = 7):
    rng = np.random.default_rng(seed)
    curr = rng.standard_normal((B, S - 1)).astype(np.float32) * 0.5 - 1.0
    prev = rng.standard_normal((B, S - 1)).astype(np.float32) * 0.5 - 1.0
    gen = rng.standard_normal((B, S - 1)).astype(np.float32) * 0.5 - 1.0
    ref = rng.standard_normal((B, S - 1)).astype(np.float32) * 0.5 - 1.0
    adv = rng.standard_normal((B, S - 1)).astype(np.float32)
    token_mask = (rng.uniform(size=(B, S - 1)) > 0.2).astype(np.float32)
    sample_mask = np.array([1.0, 1.0, 0.0], dtype=np.float32)[:B]
    return curr, prev, gen, ref, adv, token_mask, sample_mask


def _pad_front(x: np.ndarray) -> np.ndarray:
    # data columns are [B, S] and sliced [:, 1:] inside the loss; prepend a dummy col.
    return np.concatenate([np.zeros((x.shape[0], 1), x.dtype), x], axis=1)


def _torch_data(prev, gen, ref, adv, token_mask, sample_mask):
    return {
        "prev_logprobs": torch.from_numpy(_pad_front(prev)),
        "generation_logprobs": torch.from_numpy(_pad_front(gen)),
        "reference_policy_logprobs": torch.from_numpy(_pad_front(ref)),
        "advantages": torch.from_numpy(_pad_front(adv)),
        "token_mask": torch.from_numpy(_pad_front(token_mask)),
        "sample_mask": torch.from_numpy(sample_mask),
    }


def _jax_data(prev, gen, ref, adv, token_mask, sample_mask):
    return {
        "prev_logprobs": jnp.asarray(_pad_front(prev)),
        "generation_logprobs": jnp.asarray(_pad_front(gen)),
        "reference_policy_logprobs": jnp.asarray(_pad_front(ref)),
        "advantages": jnp.asarray(_pad_front(adv)),
        "token_mask": jnp.asarray(_pad_front(token_mask)),
        "sample_mask": jnp.asarray(sample_mask),
    }


_MATRIX = list(
    itertools.product(
        [True, False],          # token_level_loss
        [True, False],          # use_importance_sampling_correction
        [None, 3.0],            # ratio_clip_c
        [0.0, 0.05],            # reference_policy_kl_penalty
        ["k1", "k2", "k3"],     # kl_type
        [None, "tis"],          # truncated IS
        [False, True],          # force_on_policy_ratio
    )
)


@pytest.mark.parametrize("token_level,use_is,clip_c,kl_pen,kl_type,tis,force_op", _MATRIX)
def test_loss_and_grad_parity(token_level, use_is, clip_c, kl_pen, kl_type, tis, force_op):
    if tis is not None and not use_is:
        pytest.skip("truncated IS requires use_importance_sampling_correction=True")
    curr, prev, gen, ref, adv, token_mask, sample_mask = _fixtures()

    cfg = ClippedPGLossConfig(
        token_level_loss=token_level,
        ratio_clip_min=0.2,
        ratio_clip_max=0.2,
        ratio_clip_c=clip_c,
        reference_policy_kl_penalty=kl_pen,
        reference_policy_kl_type=kl_type,
        use_importance_sampling_correction=use_is,
        truncated_importance_sampling_type=tis,
        truncated_importance_sampling_ratio=2.0 if tis else None,
        force_on_policy_ratio=force_op,
    )

    mask = token_mask * sample_mask[:, None]
    gvt = torch.tensor(float(mask.sum()))
    gvs = torch.tensor(float(sample_mask.sum()))

    # torch loss + grad
    curr_t = torch.from_numpy(curr).clone().requires_grad_(True)
    # Plain dict is a deliberate duck-typed fixture; the loss only indexes / .get()s it.
    torch_data = _torch_data(prev, gen, ref, adv, token_mask, sample_mask)
    loss_t, _ = ClippedPGLossFn(cfg)(curr_t, torch_data, gvs, gvt)  # pyright: ignore[reportArgumentType]
    loss_t.backward()
    grad_t = curr_t.grad.detach().numpy()

    # jax loss + grad
    def f(curr_j):
        return jax_loss.clipped_pg_loss(
            curr_j, _jax_data(prev, gen, ref, adv, token_mask, sample_mask),
            jnp.asarray(float(sample_mask.sum())), jnp.asarray(float(mask.sum())), cfg,
        )

    (loss_j, _), grad_j = jax.value_and_grad(f, has_aux=True)(jnp.asarray(curr))

    np.testing.assert_allclose(float(loss_j), float(loss_t.detach()), atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(np.asarray(grad_j), grad_t, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("token_level,use_is", [(True, False), (False, False), (True, True)])
def test_disable_ppo_ratio_parity(token_level, use_is):
    # REINFORCE-style path (ratios = curr_logprobs); not covered by the main matrix.
    curr, prev, gen, ref, adv, token_mask, sample_mask = _fixtures(seed=2)
    cfg = ClippedPGLossConfig(
        token_level_loss=token_level,
        disable_ppo_ratio=True,
        reference_policy_kl_penalty=0.05,
        use_importance_sampling_correction=use_is,
    )
    mask = token_mask * sample_mask[:, None]
    gvt = torch.tensor(float(mask.sum()))
    gvs = torch.tensor(float(sample_mask.sum()))

    curr_t = torch.from_numpy(curr).clone().requires_grad_(True)
    td = _torch_data(prev, gen, ref, adv, token_mask, sample_mask)
    loss_t, _ = ClippedPGLossFn(cfg)(curr_t, td, gvs, gvt)  # pyright: ignore[reportArgumentType]
    loss_t.backward()

    def f(curr_j):
        return jax_loss.clipped_pg_loss(
            curr_j, _jax_data(prev, gen, ref, adv, token_mask, sample_mask),
            jnp.asarray(float(sample_mask.sum())), jnp.asarray(float(mask.sum())), cfg,
        )

    (loss_j, _), grad_j = jax.value_and_grad(f, has_aux=True)(jnp.asarray(curr))
    np.testing.assert_allclose(float(loss_j), float(loss_t.detach()), atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(np.asarray(grad_j), curr_t.grad.detach().numpy(), atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize(
    "tis_type,seq_level_is,on_policy_kl",
    [
        ("icepop", False, False),
        ("seq-mask-tis", False, False),
        (None, True, False),     # sequence-level importance ratios (requires seq-level loss)
        (None, False, True),     # on-policy KL approximation
        ("icepop", False, True),
    ],
)
def test_j3b_exotic_branch_parity(tis_type, seq_level_is, on_policy_kl):
    curr, prev, gen, ref, adv, token_mask, sample_mask = _fixtures(seed=5)
    cfg = ClippedPGLossConfig(
        token_level_loss=not seq_level_is,
        sequence_level_importance_ratios=seq_level_is,
        reference_policy_kl_penalty=0.05,
        use_on_policy_kl_approximation=on_policy_kl,
        use_importance_sampling_correction=tis_type is not None,
        truncated_importance_sampling_type=tis_type,
        truncated_importance_sampling_ratio=2.0 if tis_type else None,
        truncated_importance_sampling_ratio_min=0.5 if tis_type in ("icepop", "seq-mask-tis") else None,
    )
    mask = token_mask * sample_mask[:, None]
    gvt = torch.tensor(float(mask.sum()))
    gvs = torch.tensor(float(sample_mask.sum()))

    curr_t = torch.from_numpy(curr).clone().requires_grad_(True)
    td = _torch_data(prev, gen, ref, adv, token_mask, sample_mask)
    loss_t, _ = ClippedPGLossFn(cfg)(curr_t, td, gvs, gvt)  # pyright: ignore[reportArgumentType]
    loss_t.backward()

    def f(curr_j):
        return jax_loss.clipped_pg_loss(
            curr_j, _jax_data(prev, gen, ref, adv, token_mask, sample_mask),
            jnp.asarray(float(sample_mask.sum())), jnp.asarray(float(mask.sum())), cfg,
        )

    (loss_j, _), grad_j = jax.value_and_grad(f, has_aux=True)(jnp.asarray(curr))
    np.testing.assert_allclose(float(loss_j), float(loss_t.detach()), atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(np.asarray(grad_j), curr_t.grad.detach().numpy(), atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize(
    "kl_pen,kl_type,use_is",
    [
        (0.0, "k3", False),
        (0.05, "k3", False),
        (0.05, "k1", True),
    ],
)
def test_cispo_parity(kl_pen, kl_type, use_is):
    # CISPO: clip_loss = -A * sg(clip(ratio)) * log pi_theta. Token-level only;
    # incompatible with disable_ppo_ratio / force_on_policy / seq-level / dual-clip.
    curr, prev, gen, ref, adv, token_mask, sample_mask = _fixtures(seed=7)
    cfg = ClippedPGLossConfig(
        token_level_loss=True,
        use_cispo=True,
        ratio_clip_min=0.2,
        ratio_clip_max=0.2,
        reference_policy_kl_penalty=kl_pen,
        reference_policy_kl_type=kl_type,
        use_importance_sampling_correction=use_is,
    )
    mask = token_mask * sample_mask[:, None]
    gvt = torch.tensor(float(mask.sum()))
    gvs = torch.tensor(float(sample_mask.sum()))

    curr_t = torch.from_numpy(curr).clone().requires_grad_(True)
    td = _torch_data(prev, gen, ref, adv, token_mask, sample_mask)
    loss_t, _ = ClippedPGLossFn(cfg)(curr_t, td, gvs, gvt)  # pyright: ignore[reportArgumentType]
    loss_t.backward()

    def f(curr_j):
        return jax_loss.clipped_pg_loss(
            curr_j, _jax_data(prev, gen, ref, adv, token_mask, sample_mask),
            jnp.asarray(float(sample_mask.sum())), jnp.asarray(float(mask.sum())), cfg,
        )

    (loss_j, _), grad_j = jax.value_and_grad(f, has_aux=True)(jnp.asarray(curr))
    np.testing.assert_allclose(float(loss_j), float(loss_t.detach()), atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(np.asarray(grad_j), curr_t.grad.detach().numpy(), atol=2e-4, rtol=2e-4)


def test_cispo_rejects_incompatible_config():
    # The five mutual-exclusion guards must fire (torch asserts; JAX ValueError).
    curr, prev, gen, ref, adv, token_mask, sample_mask = _fixtures(seed=8)
    incompatible = [
        {"disable_ppo_ratio": True},
        {"force_on_policy_ratio": True},
        {"token_level_loss": False, "sequence_level_importance_ratios": True},
        {"ratio_clip_c": 3.0},
        {"token_level_loss": False},
    ]
    for extra in incompatible:
        with pytest.raises((AssertionError, ValueError)):
            ClippedPGLossFn(ClippedPGLossConfig(use_cispo=True, **extra))
        mask = token_mask * sample_mask[:, None]
        with pytest.raises(ValueError):
            jax_loss.clipped_pg_loss(
                jnp.asarray(curr), _jax_data(prev, gen, ref, adv, token_mask, sample_mask),
                jnp.asarray(float(sample_mask.sum())), jnp.asarray(float(mask.sum())),
                ClippedPGLossConfig(use_cispo=True, **extra),
            )


def test_logprobs_from_logits_parity():
    rng = np.random.default_rng(1)
    B, S, V = 2, 6, 17
    logits = rng.standard_normal((B, S, V)).astype(np.float32)
    ids = rng.integers(0, V, size=(B, S)).astype(np.int64)

    lp_t = torch.nn.functional.log_softmax(torch.from_numpy(logits)[:, :-1], dim=-1)
    lp_t = lp_t.gather(-1, torch.from_numpy(ids)[:, 1:].unsqueeze(-1)).squeeze(-1).numpy()
    lp_j = np.asarray(jax_loss.logprobs_from_logits(jnp.asarray(logits), jnp.asarray(ids)))

    assert lp_j.shape == (B, S - 1)
    np.testing.assert_allclose(lp_j, lp_t, atol=1e-5, rtol=1e-5)


def test_kl_backward_score_function_term():
    """#2506: the straight-through weight exp(x - sg(x)) injects the score-function
    gradient into the KL term. With w = exp(curr - sg(curr)) (forward value 1, dw/dcurr
    = 1), d/dcurr [w * kl] = kl + dkl/dcurr, so the weighted gradient exceeds the pure
    pathwise gradient by exactly the per-token KL value. The old ones_like weight dropped
    that `+ kl` term; this test fails on that regression."""
    rng = np.random.default_rng(11)
    curr_np = rng.standard_normal((3, 5)).astype(np.float32)
    ref_np = rng.standard_normal((3, 5)).astype(np.float32)
    # Clamps off so the identity is exact (clamp saturation is exercised separately).
    kw = dict(input_clamp_value=None, output_clamp_value=None)

    c1 = torch.from_numpy(curr_np).clone().requires_grad_(True)
    calculate_kl(c1, torch.from_numpy(ref_np), **kw).sum().backward()
    grad_pathwise = c1.grad.detach().numpy()

    kl_val = calculate_kl(
        torch.from_numpy(curr_np), torch.from_numpy(ref_np), **kw
    ).detach().numpy()

    c2 = torch.from_numpy(curr_np).clone().requires_grad_(True)
    w = torch.exp(c2 - c2.detach())
    calculate_kl(
        c2, torch.from_numpy(ref_np), importance_sampling_weights=w, **kw
    ).sum().backward()
    grad_sf = c2.grad.detach().numpy()

    # Score-function term adds exactly the per-token KL value.
    np.testing.assert_allclose(grad_sf, grad_pathwise + kl_val, atol=1e-5, rtol=1e-5)
    # And it genuinely differs from pathwise-only (guards against a ones_like regression).
    assert np.abs(grad_sf - grad_pathwise).max() > 1e-3

    # JAX mirror: same identity.
    def kl_sf(c):
        wj = jnp.exp(c - jax.lax.stop_gradient(c))
        return jax_loss.calculate_kl(
            c, jnp.asarray(ref_np), input_clamp_value=None, output_clamp_value=None,
            importance_sampling_weights=wj,
        ).sum()

    grad_sf_j = np.asarray(jax.grad(kl_sf)(jnp.asarray(curr_np)))
    np.testing.assert_allclose(grad_sf_j, grad_sf, atol=2e-4, rtol=2e-4)


def test_kl_backward_clamp_saturation_detaches_weight():
    """#2958: when input clamping saturates a token's log-ratio, the IS weight is
    detached there, so the clamp suppresses BOTH the KL and the sampling-weight gradient
    on that token (grad -> 0). Unsaturated tokens keep the score-function term. Without
    the detach the weight gradient (kl) would still leak on saturated tokens."""
    clamp = 5.0
    # Token 0 saturates (|curr - ref| >> clamp); token 1 does not.
    curr_np = np.array([[10.0, 0.3]], dtype=np.float32)
    ref_np = np.array([[0.0, -0.2]], dtype=np.float32)

    c = torch.from_numpy(curr_np).clone().requires_grad_(True)
    w = torch.exp(c - c.detach())
    calculate_kl(
        c, torch.from_numpy(ref_np), input_clamp_value=clamp, output_clamp_value=None,
        importance_sampling_weights=w,
    ).sum().backward()
    grad = c.grad.detach().numpy()[0]

    assert abs(grad[0]) < 1e-6   # saturated: clamp kills pathwise, detach kills score-function
    assert abs(grad[1]) > 1e-3   # unsaturated: score-function term present

    def kl_sf(cj):
        wj = jnp.exp(cj - jax.lax.stop_gradient(cj))
        return jax_loss.calculate_kl(
            cj, jnp.asarray(ref_np), input_clamp_value=clamp, output_clamp_value=None,
            importance_sampling_weights=wj,
        ).sum()

    grad_j = np.asarray(jax.grad(kl_sf)(jnp.asarray(curr_np)))[0]
    assert abs(grad_j[0]) < 1e-6
    assert abs(grad_j[1]) > 1e-3
