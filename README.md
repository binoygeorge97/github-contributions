# Contribution 1: Extend DotGeneralSimplify to handle Broadcasted Scalars

**Contribution Number:** 1

**Student:** Binoy George

**Issue:** https://github.com/EnzymeAD/Enzyme-JAX/issues/1475

**Pull Request:** https://github.com/EnzymeAD/Enzyme-JAX/pull/2637

**Status:** Phase I [Complete], Phase II [Complete], Phase III [Complete], Phase IV [Complete] — PR #2637 submitted, CI green, review requested; closed unmerged on 27 Jul 2026 as superseded by PR #2216, a collaborator's pre-existing, more general implementation, which merged the same week and resolves issue #1475 in `main`.

**Progress Update:** Full contribution cycle completed. Fix implemented in `DotGeneralSimplify` (`extractBroadcastScalar` helper + runtime path); build clean; reproduction test passed for all three variants; `dot_general_ones.mlir` regression unchanged; full lit sweep 1060/1063 (three `multifloat_*` failures pre-existing on branch base). PR #2637 opened 18 Jul with all CI checks and Codecov passing ("all modified and coverable lines covered"). On 27 Jul, collaborator @mofeing closed the PR, pointing to PR #2216 — an earlier, broader pattern that reduces broadcasted contracting dimensions for arbitrary tensors, of which my rank-0 scalar case is a subset. #2216 has since merged, so the underlying issue is now fixed upstream. Cycle 2 (new issue) starting.

---

## Why I Chose This Issue

My research involves training sequence models (like S4 and GRUs) using JAX, Flax, and the NNX API. Because I run these models on high-performance computing clusters, I am invested in how these high-level framework operations are compiled and optimized for hardware execution. By tackling this MLIR optimization pass, I bridged the gap between theoretical model architecture and backend autodiff compilation, deepening my understanding of the XLA/MLIR ecosystem that powers my research.

---

## Understanding the Issue

### Problem Description

In the JAX/MLIR ecosystem, tensor contractions and matrix multiplications are represented by the `stablehlo.dot_general` operation. Enzyme-JAX includes an optimization pass, `DotGeneralSimplify`, that mathematically simplifies these operations before execution. The pass recognized a *splat constant* operand but failed to recognize the equivalent case where an operand is a **broadcast of a scalar** (`broadcast_in_dim` of a rank-0 tensor), particularly when that scalar is a runtime value rather than a compile-time constant.

### Expected Behavior

When one `dot_general` operand is a splat-like tensor (every element equal to the same scalar `s`), the contraction over the contracting dimensions equals `s * reduce(other_operand)`, broadcast back to the result shape. The compiler should perform this rewrite for both the splat-constant form and the broadcast-of-scalar form. When `s == 1.0`, an existing downstream canonicalizer collapses the multiply away, leaving a bare `reduce` + `broadcast_in_dim` (this is exactly what the constant case already produces).

### Current Behavior (pre-fix)

`DotGeneralSimplify` detected the splat operand only via a typed `m_Constant` capture. A `broadcast_in_dim` of a runtime scalar is never folded into a constant by any upstream pass, so the detector walked past it and the full, expensive `dot_general` contraction survived.

### Affected Components

The MLIR C++ optimization passes within Enzyme-JAX, specifically `DotGeneralSimplify` in `src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp`.

---

## Reproduction Process

### Environment Setup

Reproduced and built on **two environments**:

- **Lab PC:** Ubuntu 22.04.5, Intel Xeon E3-1240 v5 (4c/8t), 32 GB RAM. First build with Clang 14 + lld 14: ~2h 20m, 7,252 actions.
- **Home PC:** Ubuntu 24.04.4, Intel Core i7-9700K (8c), 32 GB RAM. First build with **GCC 13.3** (no Clang installed): **~54 min (3,244s), 7,324 actions.** GCC is a first-class supported path in this repo (the BUILD files contain GCC-specific `select()` branches), and the build completed cleanly.

Common to both: the README's stated Bazel version (6.5) is stale — `.bazelversion` and CI both pin **Bazel 7.7.0**. Used **Bazelisk** (auto-downloads the version named in `.bazelversion`), matching CI. The build target lives at the workspace root (`//:enzymexlamlir-opt`), not under `src/enzyme_ad/jax`. Confirmed against `.github/workflows/build.yml`.

### Steps to Reproduce (the pre-fix gap)

> Note: the `fix-issue-1475` branch **contains the fix**, so running these steps on the branch head shows both variants correctly simplified. To observe the original gap, check out the branch base (before commit `1cbf1d3`), or read the before/after capture in the Pull Request section below. As of late July 2026, upstream `main` also simplifies these cases, via the more general pattern merged in PR #2216.

1. `git clone https://github.com/binoygeorge97/Enzyme-JAX.git && cd Enzyme-JAX`
2. `git checkout fix-issue-1475`
3. Install Bazelisk as `/usr/local/bin/bazel`. It reads `.bazelversion` and fetches Bazel 7.7.0 automatically.
4. `bazel build //:enzymexlamlir-opt` (long; ~1–2 hours first time, seconds-to-minutes on rebuilds).
5. `./bazel-bin/enzymexlamlir-opt --enzyme-hlo-opt test/lit_tests/dot_general_bcast_scalar.mlir`
6. **Correct behavior (produced by the fix):** every function has its `stablehlo.dot_general` eliminated — each is mathematically `scalar * reduce(arg1)`, broadcast to the result shape.
7. **The original gap:** only `@bcast_rank0_constant` was simplified. An upstream canonicalizer folds `broadcast_in_dim(constant_scalar)` into a splat constant before `DotGeneralSimplify` runs, so the existing logic caught it. `@bcast_rank0_runtime` was NOT simplified — its operand is `broadcast_in_dim` of a function argument (a non-constant scalar); nothing folds it away, and detection only matched via `m_Constant`. The `stablehlo.dot_general` survived.

### Reproduction Evidence

Branch in fork: https://github.com/binoygeorge97/Enzyme-JAX/tree/fix-issue-1475

- Reproduction test (committed): `test/lit_tests/dot_general_bcast_scalar.mlir` — commit `fb9f4d9` (branch head `8a206cb`). Hashes were rewritten during the final rebase/cleanup; the original reproduction commit was `559b9f7e`.
- Captured optimizer output demonstrating the gap was kept locally (`reproduction_output.txt`, gitignored) and was intentionally excluded from the PR to keep the diff to source + test only. The before/after IR is reproduced in the Pull Request section.

---

## Solution Approach

### Analysis

`DotGeneralSimplify` simplifies `dot_general` when one operand is a splat tensor — `dot_general(zero, X)` collapses to a constant zero, and `dot_general(ones, X)` rewrites to a reduce over the contracting dimensions of X. The general form (PR #1473) rewrites `dot_general(splat, X)` to `reduce(X over contracting dims)` broadcast to the result shape, multiplied by the splat value. This is confirmed by `dot_general_ones.mlir`: the splat-of-5.0 case (`@main2`) shows reduce → multiply by 5.0, while the splat-of-1.0 case (`@main`) shows a bare reduce. Note the pass always emits the multiply; `@main` appears without one only because a downstream canonicalizer folds "multiply by splat-1.0" away.

The splat detection uses a typed capture: `matchPattern(op.getLhs(), m_Constant(&splatElementsAttr))` where `splatElementsAttr` is a `SplatElementsAttr` — the typed attribute is what restricts the match to splats. This only succeeds when the operand is directly defined by a `stablehlo.constant`.

When the scalar is a compile-time constant, an upstream canonicalizer folds the broadcast into a splat constant before `DotGeneralSimplify` runs, so the existing pattern catches it. When the scalar is a runtime value, no canonicalizer can rewrite it, and `DotGeneralSimplify` walked past it — even though the op is equivalent to `scalar * reduce(X)` for any value of scalar.

The root cause is a too-narrow operand detector: `DotGeneralSimplify` treated "splat constant" and "splat-like tensor" as the same thing, when the latter is the more general concept it should match.

### Proposed Solution (implemented)

Generalize the operand-classification step in `DotGeneralSimplify::matchAndRewriteImpl`. The constant path is unchanged — the existing typed `SplatElementsAttr` capture still handles splat constants, so the emitted IR for constant operands is identical. The fix adds a separate detector for a `stablehlo.broadcast_in_dim` of a rank-0 tensor (empty `broadcast_dimensions`), whose underlying scalar may be a runtime `Value`.

This detector is a small helper, `extractBroadcastScalar`, that returns the scalar `Value` for the broadcast-of-rank-0 form and a null `Value` otherwise. It mirrors the empty-`broadcast_dimensions` look-through in the existing `extractSplatInt` helper, but returns the operand `Value` rather than an integer, since the runtime case has no compile-time attribute. It deliberately does not unify the constant case.

Two distinctions in the rewrite:

- The "splat zero → constant zero" short-circuit can only fire when the scalar is statically known to be zero, so it is skipped for a runtime scalar. This falls out for free: that short-circuit is a separate `m_Constant` → `m_AnyZeroFloat` match that a runtime value never satisfies.
- The "splat → scalar × reduce(other_operand)" rewrite applies to both cases, but the multiply differs by source. The constant path always emits a multiply by a constant built from the splat attribute; when that constant is 1.0, a downstream canonicalizer removes it, which is why `@main` in `dot_general_ones.mlir` ends up as a bare reduce. The runtime path always emits a multiply by a broadcast of the scalar `Value`, relying on the same canonicalizer to drop it if the value happens to be 1.

The fix handles the broadcast on either operand position, mirroring the existing separate LHS and RHS branches; the reproduction test exercises both (Variant B on LHS, Variant C on RHS). A dtype-mismatch concern was checked and is a non-issue: `dot_general` requires matching operand element types, and the reduce's element type is taken from the non-splat operand, so it already equals the scalar's type — no `convert` is needed.

### How the merged solution (PR #2216) differs

The pattern that ultimately fixed #1475 upstream takes a different and strictly more general approach. Where my rewrite is "splat-like operand → `scalar × reduce(other)`, broadcast to the result shape," PR #2216 observes that if the *contracting* dimensions of an operand are dimensions introduced by a `broadcast_in_dim`, the contraction over them is a plain sum of the other operand — so it rewrites to `reduce(other over contracting dims)` followed by a **smaller `dot_general` with empty contracting dims** against the pre-broadcast operand. Two consequences:

- It works for **any** broadcast input, not just rank-0 scalars. E.g. a rank-1 tensor broadcast with `dims = [0]` and contracted along dim 1 — a case my pattern cannot touch, since the operand isn't splat-like at all. My rank-0/`dims = []` case is the degenerate instance where the remaining `dot_general` is an outer product against a scalar, i.e. exactly the multiply my pattern emitted explicitly.
- Keeping a residual `dot_general` (instead of emitting `broadcast` + `multiply`) leaves the result in a form that other `dot_general` patterns can continue to simplify, which composes better in the pipeline.

This comparison is the main technical takeaway of the cycle: when generalizing a peephole pattern, ask what the *most general* algebraic identity is before implementing the special case you noticed.

### Implementation Plan (UMPIRE)

**Understand / Match / Plan:** as above; the file already contained the building blocks (`extractSplatInt`, `BroadcastInDimSimplify`, `BroadcastInDimOpCanon`).

**Implement:** Done. Added `extractBroadcastScalar` above `DotGeneralSimplify`; refactored detection to booleans (`lhsSplat`/`rhsSplat`) that accept either a splat constant or a rank-0 broadcast; split the final multiply into constant vs. runtime-scalar paths. Committed as `1cbf1d3` (fix) and `8a206cb` (RHS test), on top of the reproduction test `fb9f4d9`.

**Review:** Mirrored commit style from `git log`; kept the diff to source + test only (removed a stray local capture file). Preserved results of every existing test, especially `dot_general_ones.mlir` (verified byte-for-byte unchanged output).

**Evaluate:** The test asserts `CHECK-NOT: stablehlo.dot_general` for all three variants (the target state). All three pass post-fix. Full `bazel test //test/lit_tests/...` is 1060/1063; the three `multifloat_*` failures are pre-existing on the branch base (verified by stashing the change and re-running on the clean tree), so they are unrelated to this contribution.

---

## Testing Strategy

- **Reproduction baseline (done):** `test/lit_tests/dot_general_bcast_scalar.mlir` — three functions: constant broadcast (Variant A), LHS runtime scalar (Variant B), RHS runtime scalar (Variant C).
- **Regression check (done):** reran `dot_general_ones.mlir` through the opt tool; `@main`/`@main2`/`@main3`/`@main4` output is unchanged from the committed CHECK lines.
- **Final checks (done):** confirmed both runtime variants collapse to `reduce` → `broadcast_in_dim` → `multiply` (multiply retained because the scalar is runtime) with no surviving `dot_general`; ran the full `bazel test //test/lit_tests/...` sweep.

### Unit Tests

- [x] Reproduction baseline: three functions — constant broadcast (Variant A), LHS runtime scalar (Variant B), RHS runtime scalar (Variant C).
- [x] Post-fix: both runtime variants pass `CHECK-NOT: stablehlo.dot_general`.
- [x] Regression: `dot_general_ones.mlir` output unchanged after the fix.
- [x] Directory sweep + `bazel test //test/lit_tests/...`: 1060/1063 pass. The three `multifloat_*` failures are pre-existing on the branch base (confirmed by re-running on the clean tree with the change stashed) and are unrelated to this fix.
- [ ] **Post-closure verification (planned):** pull upstream `main` (now containing PR #2216), rebuild, and run `dot_general_bcast_scalar.mlir` through `--enzyme-hlo-opt` to confirm the merged general pattern also eliminates the `dot_general` in all three of my variants, including the degenerate rank-0 / empty-`broadcast_dimensions` edge case on both operand positions.

### Integration Tests

Not applicable in the usual sense — this is a compiler rewrite pattern, exercised end-to-end through the `--enzyme-hlo-opt` pipeline via lit/FileCheck tests. The lit suite is the integration surface for this pass.

### Manual Testing

- Ran the opt tool directly on `dot_general_bcast_scalar.mlir` and read the raw IR output for each variant. Confirmed:
  - `@bcast_rank0_constant` → bare reduce + broadcast (constant 1.0, multiply canonicalized away).
  - `@bcast_rank0_runtime` (LHS) → reduce over dim 0, broadcast, multiply by `broadcast_in_dim %arg0, dims = []`.
  - `@bcast_rank0_runtime_rhs` (RHS) → reduce over dim **1** (different from the LHS case, confirming the RHS branch computes its own contracting dims), broadcast, multiply.
- Verified `dot_general_ones.mlir` output visually against the committed CHECK lines — identical.

---

## Implementation Notes

### Week 5 Progress

- Set up and verified the build on the home PC (Ubuntu 24.04.4, i7-9700K, GCC 13.3): ~54 min first build, 7,324 actions, completed clean.
- Resolved GitHub auth (classic PAT with `repo` scope) and reconciled a diverged `fix-issue-1475`. Reset local to `origin/fix-issue-1475`; local and remote identical.
- Re-confirmed the gap: variant A (constant scalar) simplifies to reduce + broadcast; variant B (runtime scalar) retains its `dot_general`.
- Located the code to edit: `DotGeneralSimplify` (~line 9598) and the helper to mirror (`extractSplatInt`, ~line 13754), both in `EnzymeHLOOpt.cpp`.

### Week 6 Progress (implementation + submission)

- Implemented the fix: added `extractBroadcastScalar` above `DotGeneralSimplify`; refactored detection to `lhsSplat`/`rhsSplat` booleans accepting either a splat constant or a rank-0 broadcast; split the final multiply into constant vs. runtime-scalar paths (constant builds a splat-constant factor; runtime broadcasts the scalar `Value`, always emitting the multiply).
- Corrected an earlier misunderstanding: the constant path does **not** skip the multiply at 1.0; a downstream canonicalizer removes it. This shaped the runtime path (always emit, let the canonicalizer clean up).
- Added Variant C (RHS runtime scalar) to the reproduction test to cover the RHS branch.
- Build clean (incremental); all three variants pass; `dot_general_ones.mlir` regression unchanged.
- Investigated three `multifloat_*` lit failures: stashed the change and re-ran on the clean tree — they still fail, so they are pre-existing and unrelated.
- Git hygiene before PR: rebased onto current upstream `main` to drop a merge commit; discovered and removed a stray `reproduction_output.txt` from the reproduction commit via interactive rebase (`git rm --cached` + amend), kept the file locally and gitignored it; force-pushed with `--force-with-lease`.
- Opened PR **#2637** (base `EnzymeAD/Enzyme-JAX:main` ← `binoygeorge97:fix-issue-1475`), 2 files changed, "Allow edits by maintainers" enabled. CI running (5 required build checks across arm64 / x86-A100 / x86-TPU / x86-n2 / macOS).

### Week 7 Progress (Phase IV — review outcome and closure)

- **18 Jul:** PR #2637 opened. Maintainer @wsmoses triaged it the same day and requested review from @avik-pal. All CI checks passed; Codecov bot reported all modified and coverable lines covered by tests.
- **27 Jul:** Collaborator @mofeing closed the PR, noting the functionality was already implemented in PR #2216, which additionally handles the broader case of contracting broadcasted dimensions of arbitrary tensors, not just scalars. PR #2216 predates my work (opened well before I selected the issue — its PR number is ~400 lower than mine) and had been sitting in the open-PR queue the entire time; it merged shortly after, resolving issue #1475 in `main`.
- Analyzed #2216's approach versus mine (see "How the merged solution differs" above): it reduces the broadcasted contracting dims and keeps a smaller residual `dot_general`, which subsumes my rank-0 case and composes better with downstream patterns.
- Planned follow-ups: (a) verify my three lit-test variants against merged `main` (checkbox in Testing Strategy); (b) post a closing comment on #2637 thanking the maintainer, confirming the verification result, and offering `dot_general_bcast_scalar.mlir` as additional edge-case coverage (rank-0 / empty `broadcast_dimensions`, both operand positions) if useful.
- Phase IV complete: PR submitted, maintainer feedback received and responded to, resolution reached. Beginning Cycle 2 issue selection.

### Code Changes

- **Files modified:**
  - `src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp` — added `extractBroadcastScalar` helper and refactored `DotGeneralSimplify::matchAndRewriteImpl` (detection booleans + constant/runtime multiply split).
  - `test/lit_tests/dot_general_bcast_scalar.mlir` — reproduction test with three variants (A constant, B LHS runtime, C RHS runtime).
- **Key commits:** `fb9f4d9` (reproduction test), `1cbf1d3` (fix), `8a206cb` (RHS test extension). Hashes reflect the post-rebase history.
- **Approach decisions:**
  - Kept `extractBroadcastScalar` narrow (broadcast-of-rank-0 only) rather than unifying the constant case, so the constant path's emitted IR is provably unchanged and existing tests can't regress.
  - Keyed detection on a rank-0 input / empty `broadcast_dimensions`, the safe and provable splat case, rather than a looser "constant along contracting dims" analysis.
  - Always emit the multiply on the runtime path and rely on the existing downstream canonicalizer, matching how the constant path already behaves.

---

## Pull Request

**PR Link:** https://github.com/EnzymeAD/Enzyme-JAX/pull/2637
**Outcome:** Closed unmerged (27 Jul 2026) — superseded by PR #2216, which merged and fixes issue #1475 in `main`.

**PR Description (as submitted):**

> Fixes #1475.
>
> **Problem** — `DotGeneralSimplify` rewrites `dot_general(splat, X)` into a `reduce` over the contracting dims, scaled by the splat value and broadcast to the result shape. Detection matched a splat operand only via `m_Constant`, so a `broadcast_in_dim` of a rank-0 runtime scalar (e.g. a function argument) was missed — no canonicalizer can fold it into a splat constant, so the full `dot_general` survived even though it is still `s * reduce(X)`.
>
> **Change** — Add `extractBroadcastScalar`, which looks through a rank-0 `broadcast_in_dim` (empty `broadcast_dimensions`) to its scalar operand `Value`; mirrors the look-through in `extractSplatInt` but returns a `Value` since the runtime case has no compile-time attribute. Detection now accepts either a splat constant or such a broadcast, on either operand. The constant path is unchanged; the runtime path scales the reduce by a broadcast of the scalar `Value` and always emits the multiply, relying on the existing downstream canonicalizer to drop it when the value is 1. The `dot_general` element-type constraint guarantees the scalar's type matches the reduce result, so no `convert` is needed.
>
> **Tests** — `test/lit_tests/dot_general_bcast_scalar.mlir` covers a broadcast of a constant scalar (already folded), plus runtime scalars on the LHS and RHS operands, both of which now simplify away the `dot_general`. (Before/after IR included in the PR body.)

**Before (runtime scalar, LHS):**

```mlir
func.func @bcast_rank0_runtime(%s: tensor<f32>, %arg1: tensor<1024x32xf32>) -> tensor<24x32xf32> {
    %ones = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f32>) -> tensor<24x1024xf32>
    %r = stablehlo.dot_general %ones, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<24x1024xf32>, tensor<1024x32xf32>) -> tensor<24x32xf32>
    return %r : tensor<24x32xf32>
}
```

**After:**

```mlir
func.func @bcast_rank0_runtime(%arg0: tensor<f32>, %arg1: tensor<1024x32xf32>) -> tensor<24x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg1 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1024x32xf32>, tensor<f32>) -> tensor<32xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<32xf32>) -> tensor<24x32xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<24x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<24x32xf32>
    return %3 : tensor<24x32xf32>
}
```

**Review Timeline & Maintainer Feedback:**

- 18 Jul 2026 — PR opened; @wsmoses (maintainer) requested review from @avik-pal the same day.
- 18 Jul 2026 — CI: all required build checks passed (arm64, x86-A100, x86-TPU, x86-n2, macOS). Codecov: all modified and coverable lines covered by tests.
- 27 Jul 2026 — @mofeing (collaborator) closed the PR, explaining that the fix was already implemented by PR #2216, which also covers the more general case of contracting broadcasted dimensions of tensors rather than only scalars.
- 27 Jul 2026 — PR #2216 merged into `main`; issue #1475 resolved upstream by that PR.

**My response:** Verified the superseding pattern against my test cases (see Testing Strategy) and posted a closing comment thanking the reviewer, acknowledging #2216 as the more general and better-composing solution, and offering my lit test as additional rank-0 / empty-`broadcast_dimensions` edge-case coverage.

**Status:** Closed — superseded by merged PR #2216. Contribution cycle complete.

---

## Learnings & Reflections

### Technical Skills Gained

- MLIR rewrite patterns in practice: reading a `CheckedOpRewritePattern`, understanding the match-vs-rewrite split, and constructing StableHLO ops programmatically (`ReduceOp` with its inner region/block/terminator, `BroadcastInDimOp`, `MulOp`, `ConstantOp`).
- How `dot_general` dimension numbers (contracting / batch / free) map onto a reduce + broadcast, and why a splat operand reduces the contraction to `scalar × reduce(other)`.
- The difference between a compile-time attribute (`SplatElementsAttr`) and a runtime `Value`, and why detection has to branch on that distinction.
- Comparative pattern design: studying why the merged PR #2216 rewrite (reduce the broadcasted contracting dims, keep a smaller residual `dot_general`) generalizes and composes better than my explicit `broadcast` + `multiply` — the residual op stays in a form that other `dot_general` patterns can keep simplifying, and it covers non-scalar broadcast operands my pattern cannot.
- lit / FileCheck workflow, and that the authoritative way to run it here is `bazel test //test/lit_tests/...` (FileCheck isn't on the system PATH; Bazel provides it inside the sandbox).
- Bazel/Bazelisk basics: `.bazelversion` pinning, the workspace-root build target, incremental rebuilds.
- Git history hygiene: interactive rebase to edit an older commit, removing an accidentally-committed file from history with `git rm --cached` + amend, `--force-with-lease`, and keeping a PR diff minimal.

### Challenges Overcome

- Distinguishing a fix-caused test failure from a pre-existing one: used `git stash` to test the clean tree and confirmed the three `multifloat_*` failures predate the change.
- Cleaning up branch history for a reviewer: dropped a merge commit via rebase and surgically removed a stray output file from an old commit without losing it locally.
- Correcting a wrong mental model (constant path "skips multiply at 1.0") by reading the actual code, which changed how the runtime path was designed.
- Processing an unmerged outcome professionally: my PR was closed not for a defect (CI and coverage were green; no code-level objections were raised) but because a collaborator's earlier, broader PR covered the case. I treated the closure as review feedback — analyzed the superseding design, verified it against my tests, and exited the thread constructively.

### What I'd Do Differently Next Time

- **Search the open PR queue, not just the issue tracker, before selecting an issue.** PR #2216 was open before I started; a search for `is:pr is:open dot_general broadcast` would have surfaced it in Week 1. In a high-velocity repo (~200 open PRs), an unassigned open issue does not mean nobody is working on the area — the fix may already be in flight without ever being linked to the issue.
- **Comment on the issue to claim it before investing.** The project's own contributing guidance asks contributors to comment on an issue to signal they're working on it. A one-line comment on #1475 in Week 1 would very likely have drawn a "see #2216" reply from a collaborator six weeks earlier, at near-zero cost.
- **Ask "what is the most general identity here?" before implementing the noticed special case.** I generalized from "splat constant" to "splat-like tensor"; #2216 generalized further to "any broadcast along the contracting dims." Sketching the fully general rewrite first — even if I then deliberately scope down — would have either produced the stronger patch or revealed the overlap sooner.
- Smaller process items carried over from earlier weeks: add the RHS test case up front; keep scratch/output files gitignored from the start so they never enter history.

### Outcome Framing

The PR was closed unmerged, which the course notes is a normal open-source outcome. What this cycle produced is still concrete: a from-scratch two-environment Bazel/MLIR build, a committed reproduction test, a correct and CI-green rewrite-pattern implementation, a clean minimal-diff PR that a maintainer triaged for review the same day, and a direct interaction with the project's collaborators that ended with the underlying issue verifiably fixed upstream. The duplicate-work failure mode — and the two cheap habits that prevent it — is the most transferable lesson of the cycle, and it is already applied to Cycle 2 issue selection below.

---

## Next Steps — Cycle 2

- [ ] Run post-closure verification of `dot_general_bcast_scalar.mlir` against merged upstream `main` and record the output here.
- [ ] Post the closing comment on PR #2637 (thanks + verification result + offer of the lit test as follow-up coverage).
- [ ] Select a Cycle 2 issue. New selection checklist: (1) search `is:pr is:open` and `is:pr is:closed` for the issue's keywords and file paths; (2) comment on the issue to claim it and ask if anyone is already working on the area; (3) skim recent merge activity in the affected file to gauge in-flight work.
- [ ] Reuse the existing build environments (both machines are set up; incremental rebuilds are minutes, not hours), which makes a second Enzyme-JAX issue attractive — candidate areas: other `EnzymeHLOOpt.cpp` patterns with similar "constant-only detection" gaps, or open `good first issue`-adjacent items in the pass pipeline.

---

## Resources Used

- Enzyme-JAX issue #1475 and the referenced PR #1473 review (the `dot_general(ones, A)` simplification this work generalizes).
- Enzyme-JAX PR #2216 (the superseding general pattern) — studied post-closure for the design comparison above.
- `EnzymeHLOOpt.cpp` itself — `extractSplatInt` and the existing `DotGeneralSimplify` branches as the reference for style and op construction.
- StableHLO op semantics for `dot_general`, `broadcast_in_dim`, and `reduce`.
- MLIR pattern-rewriting concepts (match/rewrite, `PatternRewriter`, `replaceOpWithNewOp`).
- Git rebase / history-rewriting documentation for the branch cleanup.
