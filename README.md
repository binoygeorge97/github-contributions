# Contribution 1: Extend DotGeneralSimplify to handle Broadcasted Scalars

**Contribution Number:** 1  
**Student:** Binoy George 
**Issue:** https://github.com/EnzymeAD/Enzyme-JAX/issues/1475
**Status:** Phase I [Complete]
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

[Notes on setting up your local development environment - challenges you faced, how you solved them]

### Steps to Reproduce

1. Write a small MLIR unit test (or JAX script lowered to MLIR) that broadcasts a scalar and passes it into stablehlo.dot_general.
2. Run the Enzyme-JAX MLIR optimizer over this specific file.
3. Observe the output Intermediate Representation (IR).

### Reproduction Evidence

- **Commit showing reproduction:** [Link to commit in your fork]
- **Screenshots/logs:** [If applicable]
- **My findings:** [What you discovered during reproduction]

---

## Solution Approach

### Analysis

[Your analysis of the root cause - what's causing the issue?]

### Proposed Solution

[High-level description of your fix approach]

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** [Restate the problem]

**Match:** [What similar patterns/solutions exist in the codebase?]

**Plan:** [Step-by-step implementation plan]
1. [Modify file X to do Y]
2. [Add function Z]
3. [Update tests]

**Implement:** [Link to your branch/commits as you work]

**Review:** [Self-review checklist - does it follow the project's contribution guidelines?]

**Evaluate:** [How will you verify it works?]

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
