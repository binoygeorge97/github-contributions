# Contribution 1: Extend DotGeneralSimplify to handle Broadcasted Scalars

**Contribution Number:** 1  
**Student:** Binoy George 
**Issue:** https://github.com/EnzymeAD/Enzyme-JAX/issues/1475
**Status:** Phase I [Complete], Phase II [Complete]
**Progress Update**: commented + forked + approved

---

## Why I Chose This Issue

My research involves training sequence models (like S4 and GRUs) using JAX, Flax, and the NNX API. Because I run these models on high-performance computing clusters, I am invested in how these high-level framework operations are compiled and optimized for hardware execution. By tackling this MLIR optimization pass, I will bridge the gap between theoretical model architecture and backend autodiff compilation, deepening my understanding of the XLA/MLIR ecosystem that powers my research.

---

## Understanding the Issue

### Problem Description

In the JAX/MLIR ecosystem, tensor contractions and matrix multiplications are represented by the stablehlo.dot_general operation. The Enzyme autodiff compiler includes an optimization pass called DotGeneralSimplify designed to mathematically simplify these operations before execution. Currently, this pass fails to recognize when one of the operands is simply a broadcasted scalar.

### Expected Behavior

If a scalar is broadcasted into a tensor and fed into a dot_general contraction, the compiler should recognize this and replace the heavy contraction with a cheaper, standard element-wise multiplication (mul) using the original scalar.

### Current Behavior

The DotGeneralSimplify pass ignores the broadcasted scalar, forcing the compiler to execute a full, computationally expensive tensor contraction.

### Affected Components

The MLIR C++ optimization passes within Enzyme-JAX, specifically the file containing the DotGeneralSimplify logic.

---

## Reproduction Process

### Environment Setup

Built on Ubuntu 22.04 (desktop, Xeon E3-1240 v5, 32 GB RAM). The README's stated Bazel version (6.5) is stale — .bazelversion and CI both pin Bazel 7.7.0. Switched from apt-installed Bazel 6.5 to Bazelisk (which auto-downloads the version named in .bazelversion), matching what CI uses. The build target lives at the workspace root (//:enzymexlamlir-opt), not under src/enzyme_ad/jax. Confirmed against .github/workflows/build.yml. First build: ~2h 20m, 7252 actions.

### Steps to Reproduce

1. git clone https://github.com/binoygeorge97/Enzyme-JAX.git && cd Enzyme-JAX
2. git checkout fix-issue-1475
3. Install Bazelisk as /usr/local/bin/bazel. It reads .bazelversion and fetches Bazel 7.7.0 automatically.
4. bazel build //:enzymexlamlir-opt (long; ~2 hours first time, seconds on rebuilds)
5. ./bazel-bin/enzymexlamlir-opt --enzyme-hlo-opt test/lit_tests/dot_general_bcast_scalar.mlir
6. Expected: Both functions have their stablehlo.dot_general replaced by a reduce (the same pattern PR #1473 added for splat-constant operands).
7. Actual: @bcast_rank0_constant is simplified — a different pass folds broadcast_in_dim(constant_scalar) into a splat constant before DotGeneralSimplify runs, so the existing logic catches it. @bcast_rank0_runtime is not simplified — the operand is broadcast_in_dim of a function argument (a non-constant scalar), nothing folds it away, and DotGeneralSimplify only matches via m_Constant. The stablehlo.dot_general survives. Captured in reproduction_output.txt.

### Reproduction Evidence

Branch in fork: https://github.com/binoygeorge97/Enzyme-JAX/tree/fix-issue-1475
Specifically:
- Reproduction test: test/lit_tests/dot_general_bcast_scalar.mlir
- Captured output: reproduction_output.txt

---

## Solution Approach

### Analysis

[Your analysis of the root cause - what's causing the issue?]

### Proposed Solution

[High-level description of your fix approach]

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** DotGeneralSimplify::matchAndRewriteImpl in src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp simplifies dot_general(splat, X) by:
- replacing with a zero constant when the splat value is zero, or
- rewriting dot_general(splat, X) as splat_value * reduce(X over contracting dims) for non-zero splats.

It uses matchPattern(operand, m_Constant(&attr)) and checks attr.isSplat(). This misses the equivalent case where the splat-like tensor is produced by stablehlo.broadcast_in_dim of a rank-0 (or rank-1 size-1) tensor that isn't a constant — the most common version of which is a broadcast of a function argument.

**Match:** The same file already contains the building blocks:
- extractSplatInt walks through BroadcastInDimOp with empty broadcast_dimensions to find an underlying scalar.
- BroadcastInDimSimplify (further down) handles the rank-0 input case for a different op.
- BroadcastInDimOpCanon shows the canonical "splat detection" path.
The fix reuses this shape of logic.

**Plan:** [Step-by-step implementation plan]
1. In DotGeneralSimplify::matchAndRewriteImpl, generalize the operand-classification step: each operand counts as "splat-like" if either
(a) it matches m_Constant with attr.isSplat() (today's path), or
(b) it's a stablehlo::BroadcastInDimOp whose input has rank 0 (or rank 1 with a single element).
2. When (b) holds, the per-element "scalar value" is the broadcast's input value itself (a Value, not a constant attribute).
3. Rewrite the existing logic so the rewrite is expressed as "scalar * reduce(other_operand)" where scalar may come from either source. Broadcast scalar back to the result shape and multiply with the reduced tensor.
4. The existing zero-splat short-circuit only needs to fire when the scalar source is a known-zero constant; for case (b) the value isn't statically known so the general path applies.

**Implement:** Phase III work. Branch: https://github.com/binoygeorge97/Enzyme-JAX/tree/fix-issue-1475

**Review:** Read CONTRIBUTING.md if present; mirror style from git log --oneline -20. Ensure the fix preserves results of every existing test in test/lit_tests/, especially dot_general_ones.mlir.

**Evaluate:** Add CHECK-NOT: stablehlo.dot_general assertions to test/lit_tests/dot_general_bcast_scalar.mlir for both functions. After the fix, @bcast_rank0_runtime should no longer contain stablehlo.dot_general in its optimized output. All existing tests under test/lit_tests/ must still pass.

---

## Testing Strategy

### Unit Tests

- [ ] Test case 1: [Description]
- [ ] Test case 2: [Description]
- [ ] Test case 3: [Description]

### Integration Tests

- [ ] Integration scenario 1
- [ ] Integration scenario 2

### Manual Testing

[What you tested manually and results]

---

## Implementation Notes

### Week [X] Progress

[What you built this week, challenges faced, decisions made]

### Week [Y] Progress

[Continue documenting as you work]

### Code Changes

- **Files modified:** [List]
- **Key commits:** [Links to important commits]
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
