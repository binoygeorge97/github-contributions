# Contribution 1: Extend DotGeneralSimplify to handle Broadcasted Scalars

**Contribution Number:** 1  
**Student:** Binoy George 
**Issue:** https://github.com/EnzymeAD/Enzyme-JAX/issues/1475
**Status:** Phase I [Complete], Phase II [Complete], Phase III [In Progress]
**Progress Update**: commented + forked + approved; reproduction test on branch; home-PC build working (GCC 13.3); git auth + fork sync resolved. Fix not yet implemented — next step is editing DotGeneralSimplify.

---

## Why I Chose This Issue

My research involves training sequence models (like S4 and GRUs) using JAX, Flax, and the NNX API. Because I run these models on high-performance computing clusters, I am invested in how these high-level framework operations are compiled and optimized for hardware execution. By tackling this MLIR optimization pass, I will bridge the gap between theoretical model architecture and backend autodiff compilation, deepening my understanding of the XLA/MLIR ecosystem that powers my research.

---

## Understanding the Issue

### Problem Description

In the JAX/MLIR ecosystem, tensor contractions and matrix multiplications are represented by the `stablehlo.dot_general` operation. Enzyme-JAX includes an optimization pass, `DotGeneralSimplify`, that mathematically simplifies these operations before execution. Currently, this pass recognizes a *splat constant* operand but fails to recognize the equivalent case where an operand is a **broadcast of a scalar** (`broadcast_in_dim` of a rank-0 tensor), particularly when that scalar is a runtime value rather than a compile-time constant.

### Expected Behavior

When one `dot_general` operand is a splat-like tensor (every element equal to the same scalar `s`), the contraction over the contracting dimensions equals `s * reduce(other_operand)`, broadcast back to the result shape. The compiler should perform this rewrite for both the splat-constant form and the broadcast-of-scalar form. When `s == 1.0`, an existing canonicalizer collapses the multiply away, leaving a bare `reduce` + `broadcast_in_dim` (this is exactly what the constant case already produces).

### Current Behavior

`DotGeneralSimplify` detects the splat operand only via `matchPattern(operand, m_Constant(...))`. A `broadcast_in_dim` of a runtime scalar is never folded into a constant by any upstream pass, so the detector walks past it and the full, expensive `dot_general` contraction survives.

### Affected Components

The MLIR C++ optimization passes within Enzyme-JAX, specifically `DotGeneralSimplify` in `src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp`.

---

## Reproduction Process

### Environment Setup

Reproduced and built on **two environments**:

- **Lab PC:** Ubuntu 22.04.5, Intel Xeon E3-1240 v5 (4c/8t), 32 GB RAM. First build with Clang 14 + lld 14: ~2h 20m, 7,252 actions.
- **Home PC:** Ubuntu 24.04.4, Intel Core i7-9700K (8c), 32 GB RAM. First build with **GCC 13.3** (no Clang installed): **~54 min (3,244s), 7,324 actions.** GCC is a first-class supported path in this repo (the BUILD files contain GCC-specific `select()` branches), and the build completed cleanly.

Common to both: the README's stated Bazel version (6.5) is stale — `.bazelversion` and CI both pin **Bazel 7.7.0**. Used **Bazelisk** (auto-downloads the version named in `.bazelversion`), matching CI. The build target lives at the workspace root (`//:enzymexlamlir-opt`), not under `src/enzyme_ad/jax`. Confirmed against `.github/workflows/build.yml`.

### Steps to Reproduce

1. `git clone https://github.com/binoygeorge97/Enzyme-JAX.git && cd Enzyme-JAX`
2. `git checkout fix-issue-1475`
3. Install Bazelisk as `/usr/local/bin/bazel`. It reads `.bazelversion` and fetches Bazel 7.7.0 automatically.
4. `bazel build //:enzymexlamlir-opt` (long; ~1–2 hours first time, seconds-to-minutes on rebuilds).
5. `./bazel-bin/enzymexlamlir-opt --enzyme-hlo-opt test/lit_tests/dot_general_bcast_scalar.mlir`
6. **Expected (correct behavior):** both functions should have their stablehlo.dot_general eliminated — each is mathematically scalar * reduce(arg1), broadcast to the result shape. This is what the fix will produce.
7. **Actual (the gap):** only @bcast_rank0_constant is simplified. An upstream canonicalizer folds broadcast_in_dim(constant_scalar) into a splat constant before DotGeneralSimplify runs, so the existing logic catches it (output: reduce(add) over dim 0, then broadcast_in_dim to the result shape). @bcast_rank0_runtime is NOT simplified — its operand is broadcast_in_dim of a function argument (a non-constant scalar); nothing folds it away, and DotGeneralSimplify only matches via m_Constant. The stablehlo.dot_general survives.

### Reproduction Evidence

Branch in fork: https://github.com/binoygeorge97/Enzyme-JAX/tree/fix-issue-1475

- Reproduction test (committed): `test/lit_tests/dot_general_bcast_scalar.mlir` — commit `559b9f7e`, on branch head `aea96bcf`.
- Captured optimizer output demonstrating the gap (the runtime variant retaining its `dot_general` while the constant variant is reduced to `reduce` + `broadcast_in_dim`).

---

## Solution Approach

### Analysis

`DotGeneralSimplify` simplifies `dot_general` when one operand is a splat tensor — `dot_general(zero, X)` collapses to a constant zero, and `dot_general(ones, X)` rewrites to a reduce over the contracting dimensions of X. The general form (PR #1473) rewrites dot_general(splat, X) to reduce(X over contracting dims) broadcast to the result shape, multiplied by the splat value. This is confirmed by dot_general_ones.mlir: the splat-of-5.0 case (@main2) produces reduce → multiply by 5.0, while the splat-of-1.0 case (@main) produces a bare reduce with no multiply.

The detection relies on a single match: `matchPattern(op.getLhs(), m_Constant(&lhsAttr))`. `m_Constant` only succeeds when the operand is directly defined by a `stablehlo.constant`. It misses the equivalent form: a tensor produced by `stablehlo.broadcast_in_dim` of a rank-0 scalar.

When the scalar is a compile-time constant, an upstream canonicalizer folds the broadcast into a splat constant before DotGeneralSimplify runs, so the existing pattern catches it. When the scalar is a runtime value, no canonicalizer can rewrite it, and `DotGeneralSimplify` walks past it — even though the op is equivalent to `scalar * reduce(X)` for any value of scalar.

The root cause is a too-narrow operand detector: `DotGeneralSimplify` treats "splat constant" and "splat-like tensor" as the same thing, when the latter is the more general concept it should match.

### Proposed Solution

Generalize the operand-classification step in `DotGeneralSimplify::matchAndRewriteImpl` to recognize both forms:

(a) a splat constant via `matchPattern(v, m_Constant(&splat))` + `splat.isSplat()` (current path), and  
(b) a `stablehlo.broadcast_in_dim` of a rank-0 tensor, where the underlying scalar may be a runtime `Value`.

Planned as a small helper, extractSplatLikeScalar, that returns the underlying scalar Value for either form and nullptr otherwise. It mirrors the empty-broadcast_dimensions look-through used by the existing extractSplatInt helper (line 13754) — but returns the scalar operand Value rather than an integer, since the runtime case has no compile-time attribute.

One careful distinction in the rewrite:

- The "splat zero → constant zero" short-circuit can only fire when the scalar is statically known to be zero; for a runtime scalar it is skipped.
- The general "splat-like → scalar × reduce(other_operand)" rewrite applies to both cases, but the multiply differs by source. The existing constant path multiplies by a constant built from the splat attribute, and skips the multiply entirely when the value is 1.0 — which is why @main in dot_general_ones.mlir shows a bare reduce. The runtime path has no attribute, so it multiplies the reduce result by the broadcast's input Value and always emits the multiply, relying on the same downstream canonicalizer to collapse it if the value happens to be 1.0.

The fix must handle the broadcast-of-scalar on either operand position, mirroring the existing code's separate LHS-constant and RHS-constant branches. The reproduction test only exercises the LHS case, but a robust fix (and a reviewer) will expect both.

### Implementation Plan (UMPIRE)

**Understand / Match / Plan:** as above; the file already contains the building blocks (`extractSplatInt`, `BroadcastInDimSimplify`, `BroadcastInDimOpCanon`).

**Implement:** Phase III work — *not yet started in code.* The reproduction test is committed (`952b8104`); `EnzymeHLOOpt.cpp` has not yet been modified.

**Review:** Mirror style from `git log --oneline -20`; check for `CONTRIBUTING.md`. Preserve results of every existing test, especially `dot_general_ones.mlir`.

**Evaluate:** The test already asserts `CHECK-NOT: stablehlo.dot_general` for both variants (the target state). Variant B fails today by design and turns green once the fix lands. All existing `test/lit_tests/` must still pass.

---

## Testing Strategy

- **Reproduction baseline (done):** `test/lit_tests/dot_general_bcast_scalar.mlir` demonstrates the gap. Constant variant simplifies pre-fix; runtime variant does not.
- **Regression check (planned per-edit):** rerun `dot_general_ones.mlir` through the opt tool and diff against captured baseline.
- **Final checks (planned):** confirm the runtime variant collapses to the same `reduce` (+ multiply, when scalar ≠ 1) structure as the constant path; directory sweep over `test/lit_tests/*.mlir`; `bazel test //test/lit_tests/...` for the proper lit run.

### Unit Tests

- [x] Reproduction baseline committed (Phase II): two functions, constant-broadcast (works pre-fix) and runtime-scalar (broken pre-fix).
- [ ] Post-fix: runtime variant no longer contains `stablehlo.dot_general`.
- [ ] Regression: `dot_general_ones.mlir` output unchanged after the fix.
- [ ] Directory sweep + `bazel test //test/lit_tests/...` green.

### Integration Tests

- [ ] Integration scenario 1
- [ ] Integration scenario 2

### Manual Testing

[What you tested manually and results]

---

## Implementation Notes

### Week 5 Progress

- Set up and verified the build on the home PC (Ubuntu 24.04.4, i7-9700K, GCC 13.3): ~54 min first build, 7,324 actions, completed clean.
- Resolved GitHub auth (classic PAT with `repo` scope) and reconciled a diverged `fix-issue-1475` (a duplicate reproduction commit made here vs. the lab commit already on the fork). Reset local to `origin/fix-issue-1475` (`aea96bcf`); local and remote now identical.
- Re-confirmed the gap: variant A (constant scalar) simplifies to reduce + broadcast; variant B (runtime scalar) retains its dot_general.
- Located the code to edit: `DotGeneralSimplify` at line ~9598, and the helper to mirror (`extractSplatInt`) at ~13754, both in `EnzymeHLOOpt.cpp`. Fix editing is the next step.

### Week [Y] Progress

[Continue documenting as you work]

### Code Changes

- **Files to modify (planned):** src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp — add extractSplatLikeScalar helper and refactor DotGeneralSimplify::matchAndRewriteImpl to use it. test/lit_tests/dot_general_bcast_scalar.mlir — reproduction test, already committed.
- **Key commits:** 559b9f7e (reproduction test). Fix commit: not yet.
- **Approach decisions:** [Why you chose certain approaches]

---

## Pull Request

**PR Link:** [GitHub PR URL when submitted]

**PR Description:** [Draft or final PR description - much of the content above can be adapted]

**Maintainer Feedback:**
- [Date]: [Summary of feedback received]
- [Date]: [How you addressed it]

**Status:** [Awaiting review / Iterating / Approved / Merged]

---

## Learnings & Reflections

### Technical Skills Gained

[What you learned technically]

### Challenges Overcome

[What was hard and how you solved it]

### What I'd Do Differently Next Time

[Reflection on your process]

---

## Resources Used

- [Link to helpful documentation]
- [Tutorial or Stack Overflow post that helped]
- [GitHub issues or discussions that helped]
