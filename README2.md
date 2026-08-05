# Contribution 2: Port the "Debug a VAE" tutorial into the Flax NNX docs

**Contribution Number:** 2

**Student:** Binoy George

**Issue:** https://github.com/google/flax/issues/5433 — sub-item: *"Part 2: Debug a variational autoencoder (VAE)"*

**Pull Request:** _TBD — draft PR to be opened early in Phase IV._

**Status:** Phase I [Complete], Phase II [In progress], Phase III [Not started], Phase IV [Not started] — sub-item of umbrella issue #5433 claimed via comment to maintainer @vfdev-5; source tutorial analyzed and scope confirmed; local docs dev-env setup and source-notebook reproduction pending.

**Progress Update:** Selected the unchecked "Debug a VAE" sub-item of the NNX-docs example-porting umbrella issue #5433 and claimed it in a comment to @vfdev-5. Read the source tutorial end to end. Scope confirmed as a **port-and-refresh, not a rewrite**: the tutorial already uses `flax.nnx` (`nnx.Module`, `nnx.Linear`), so the work is (a) moving it into the Flax NNX docs tree, (b) refreshing the code against the current NNX API, (c) preserving its deliberate broken-then-fixed debugging narrative, and (d) wiring it into the toctree following the shape of the already-merged sibling examples (#5455 ViT, #5431 machine translation, #5405 miniGPT). Next: stand up the docs dev environment and run the source notebook under the currently pinned Flax/JAX/Optax to capture the exact API drift.

---

## Why I Chose This Issue

_(Draft — adjust to taste.)_ My research trains sequence models (S4, GRUs) in JAX/Flax using the NNX API, so the NNX docs are infrastructure I use directly. Contribution 1 took me *below* the framework, into the MLIR/XLA compiler; this contribution moves in the opposite direction, into the framework's teaching surface. Porting the VAE debugging tutorial means working through the current NNX training-loop, optimizer, and RNG APIs exactly as a new NNX user first encounters them, and making sure the project's own documentation reflects the current API. It is also a cleanly scoped *documentation* contribution — a deliberate contrast in contribution **type** to the compiler bug-fix of Cycle 1, which broadens the portfolio.

**Applying Cycle 1's lessons up front:** Cycle 1's top "what I'd do differently" was *comment to claim an issue before investing*. I did exactly that here — I commented on #5433 to claim the sub-item and to confirm scope with the maintainer before any environment setup. _(Remaining pre-flight to log: search `google/flax` `is:pr VAE` / `digits_vae` to confirm no in-flight port — Cycle 1 lesson #1.)_

---

## Understanding the Issue

### What the issue asks

#5433 is a **tracking / umbrella issue**: the maintainers are porting the Jax AI Stack example notebooks into the Flax NNX docs and checking them off as PRs land. Most items are already done. *"Part 2: Debug a variational autoencoder (VAE)"* is one of the few still unchecked (marked *low priority*), with no linked PR at selection time.

### Source material

- Rendered tutorial: https://docs.jaxstack.ai/en/latest/digits_vae.html
- Source notebook: the `.ipynb` linked from that page (jax-ai-stack repo, `docs/source/digits_vae.ipynb`).
- Model: a small **linear** VAE (encoder → latent `mean`/`std` → decoder) on the scikit-learn `digits` dataset (8×8 images, ~1800 samples). The tiny dataset keeps the notebook cheap to execute in a docs build.

### What "done" looks like

A notebook living in the Flax NNX docs, wired into the toctree, that (a) runs under the current pinned Flax/JAX/Optax, (b) preserves the tutorial's pedagogical arc, and (c) matches the file layout and conventions of the already-merged sibling example PRs.

### The tutorial's actual subject is *debugging* — do not sanitize it

This is **not** a "here's a VAE" tutorial; it is a JAX debugging walkthrough. The VAE is intentionally written with a defect: the raw digit images (integer values 0–16) are passed to `optax.sigmoid_binary_cross_entropy`, whose target argument must be binary. Training diverges to `NaN`. The tutorial then teaches `jax.debug_nans` and `jax.debug.print` to trace the divergence back to the input data, and fixes it by binarizing the images (`(digits.images / 16) > 0.5`).

Preserving this **broken-then-fixed** narrative is a hard requirement of the port. "Refreshing" the code into a VAE that trains cleanly from the top would delete the entire lesson. (Subtlety in the same spot: the prose says "normalize to (0,1)" but the fix actually *binarizes* — binarization is what BCE requires, so this is intended, not a wording bug to be corrected.)

### Affected components

Flax NNX documentation tree (notebook source + toctree entry). **No library source-code changes expected** — this is a docs-only contribution.

---

## Reproduction Process

> For a documentation port there is no bug to reproduce in the project itself. The analogous Phase II step is to **reproduce the source tutorial under the current pinned dependencies and record exactly what breaks** — that failure list becomes the port worklist, the same role the MLIR reproduction played in Cycle 1.

### Environment Setup

_TBD after setup._ Plan: fork + clone `google/flax`; create the docs dev environment using the dependency/build setup cribbed from a recently merged sibling example PR (e.g. #5455); record the pinned versions from the repo.

- OS / hardware: `[FILL]`
- Flax version (pinned in repo): `[FILL]`
- JAX / jaxlib version: `[FILL]`
- Optax version: `[FILL]`
- Docs build command: `[FILL — from the Flax docs contributing guide / a sibling PR]`

### Steps to Reproduce (source tutorial under current Flax)

1. Download the source `.ipynb` from the jax-ai-stack docs page.
2. Run it top to bottom under the pinned Flax/JAX/Optax.
3. Record every cell that errors or behaves differently from the published output.

### Anticipated API drift (confirm during the run — pinned version is the source of truth)

- **`optimizer.update(grads)`** — the NNX optimizer `update` signature changed in a recent Flax version to also take the model (≈ `optimizer.update(model, grads)`). Most likely first break.
- **`nnx.ModelAndOptimizer(...)` vs `nnx.Optimizer(...)`** — this area moved (`nnx.Optimizer` gained a required `wrt` argument). Confirm which the pinned version wants.
- **Named RNG stream** — `nnx.Rngs(0, noise=1)` + `self.rngs.noise()`. Confirm it still resolves.
- **Narrative integrity** — verify the intentional `NaN` still triggers the same way once the loop runs; an optimizer API change could alter how/whether divergence appears.

### Reproduction Evidence

_TBD — capture the pre-refresh failures (tracebacks / diffs vs. the published output). Keep any scratch capture files gitignored from the start (Cycle 1 process carry-over)._

---

## Solution Approach

### Plan

1. Port the notebook into the Flax NNX docs source location used by the sibling examples; match their format (likely a jupytext-paired markdown/notebook — confirm from a merged PR).
2. Refresh each NNX API call flagged above to the current version, changing as little else as possible.
3. Preserve the broken-then-fixed debugging arc verbatim in structure: bug → `jax.debug_nans` → `jax.debug.print` → binarize fix → results exploration (reconstruction / generation / latent interpolation).
4. Wire the new page into the toctree in the same series position the source has it (Part 2, between "JAX neural net basics" and "Train a diffusion model").
5. Handle the intentional-exception cell for CI (see below).

### Key risk to design for up front: CI execution of the intentional-`NaN` cell

The sibling example notebooks all execute cleanly in the docs build. This one contains a cell that **intentionally raises** a `FloatingPointError` (the `jax.debug_nans(True)` block). If the Flax docs build executes notebooks, that cell will fail the build unless it is marked expected-to-raise. Check how the source `.ipynb` tags it — almost certainly a `raises-exception` cell tag (honored by nbsphinx / myst-nb) — and replicate that tagging. Confirm the build system and the tag during Phase II. Getting this right up front is the difference between a clean PR and a red CI check.

### Implementation Plan (UMPIRE)

- **Understand / Match / Plan:** as above.
- **Implement:** _TBD Phase III._
- **Review:** match sibling-PR file conventions; keep the diff to the new notebook + toctree entry only.
- **Evaluate:** docs build passes; notebook executes (or the intended-exception cell is correctly tagged); narrative outputs match the published tutorial.

---

## Testing Strategy

- [ ] Notebook runs top-to-bottom under the pinned Flax/JAX/Optax after the API refresh.
- [ ] The intentional-`NaN` cell behaves as intended and is correctly tagged so the docs build passes.
- [ ] Full docs build succeeds locally (no broken toctree reference; no build error from this page).
- [ ] Narrative outputs preserved: loss → `NaN` pre-fix (by ~epoch 500); post-fix loss decreases to ~0.26 over 2000 epochs; reconstruction, generation, and interpolation figures render.
- [ ] Page appears in the rendered docs navigation in the correct series position.

### Integration / Manual Testing

_TBD — for a docs contribution the "integration surface" is the docs build itself plus a visual read of the rendered page._

---

## Implementation Notes

### Week [N] Progress

- Selected and claimed the "Debug a VAE" sub-item on #5433 (comment to @vfdev-5).
- Read the source tutorial; confirmed port-and-refresh scope and the keep-the-bug constraint.
- Identified the three likely API-drift points and the CI-execution risk before writing any code.
- _[ongoing...]_

### Code Changes

_TBD — files added/modified (notebook source + toctree entry), key commits._

---

## Pull Request

_TBD._ Plan: open as a **draft PR early** (as offered in the claim comment) so the maintainer has something concrete to react to, then iterate. Record PR link, description, review timeline, and responses here as they happen.

---

## Learnings & Reflections

_TBD._ Early note: this cycle directly exercised Cycle 1's top lesson — claiming the issue before investing, and searching the PR queue for existing work first. Further reflections to follow after Phases III–IV.

---

## Next Steps

- [ ] Pre-flight: search `google/flax` `is:pr VAE` / `digits_vae` to confirm no in-flight port (Cycle 1 lesson #1).
- [ ] Stand up the docs dev environment; record pinned versions.
- [ ] Run the source notebook; capture API drift and the intentional-`NaN` behavior.
- [ ] Confirm the docs build system and how it handles the `raises-exception` cell.
- [ ] Open a draft PR early; iterate on maintainer feedback.

---

## Resources Used

- Flax issue #5433 (umbrella) and merged sibling example PRs (#5455 ViT, #5431 machine translation, #5405 miniGPT) — template for file layout, toctree wiring, and review expectations.
- Source tutorial: jax-ai-stack *"Part 2: Debug a variational autoencoder (VAE)"*.
- JAX debugging docs: `jax.debug_nans`, `jax.debug.print`.
- Flax NNX API reference — current optimizer / RNG / training-loop APIs.
