# Validation — `MCRLoss` (SimDINO coding rate loss)

This file records what was validated for `lightly/loss/mcr_loss.py` in this
change and what is deferred.

## Validated here (CPU, deterministic)

- **Numerical parity with the reference implementation.**
  `tests/loss/test_mcr_loss.py::TestMCRLossParity` embeds a port of the
  reference `MCRLoss` from
  [RobinWu218/SimDINO](https://github.com/RobinWu218/SimDINO/blob/main/simdino/main_dino.py)
  and checks, on a fixed seed, that the ported `calc_compression` /
  `calc_expansion` and the combined forward loss agree for both `expa_type`
  variants (0 and 1) and for non-default `eps` / `coeff`. The observed maximum
  absolute difference is `<= 1.2e-07` (single-process; the distributed
  all-reduce branch is a no-op when no process group is initialized).
- **Forward / output shape / gradient flow.** The loss returns a finite scalar
  and backward produces finite gradients on all (student *and* teacher) view
  features; SimDINO, unlike DINO, does not detach the teacher.
- **Distributed guard.** The `torch.distributed` all-reduce is guarded with
  `is_available() and is_initialized()` and is a no-op single-process (and on
  torch builds without distributed support); covered by a monkeypatched test.
- **Integration with the existing DINO substrate.** A test feeds features from
  the pre-existing `ProjectionHead` (bottleneck without DINO's prototype
  layer) followed by L2 normalization — the SimDINO head — through the loss.
  A short smoke training run of this setup on synthetic data decreases the
  loss monotonically (not part of the committed test suite).

Run with:

```bash
python -m pytest tests/loss/test_mcr_loss.py
```

## Deferred (human, GPU — not this run)

Per the SimDINO paper (arXiv:2502.10385), the headline results are ImageNet
pretraining runs plus linear-probe / k-NN evaluation. Reproducing them needs
multi-GPU resources and is intentionally out of scope for this change:

1. **ImageNet pretraining parity.** Train the SimDINO v1 configuration
   (ViT-S/16, the reference hyperparameters: `eps=0.5`, `coeff=1.0`,
   `expa_type=1` as used by the reference training script, 2 global + 8 local
   crops) with the lightly `MCRLoss` on the `DINOTransform` multi-crop
   pipeline and compare training curves against the reference repository run
   from the same seed.
2. **Downstream evaluation.** Linear-probe and k-NN accuracy on ImageNet
   val, compared against the numbers reported in the paper (e.g. SimDINO
   ViT-S/16 linear probe).
3. **Distributed runs.** Confirm the `reduce_cov=True` covariance all-reduce
   path under a real multi-process group (gloo/NCCL).

The loss implementation, export, autodoc entry, and CPU test suite (including
the parity core) are complete; nothing in this change is stubbed.
