# Methodology and Approach

This document describes the technical approach and design decisions for adaptive contrastive curriculum learning on MMLU.

## Problem Statement

Multi-task learning on MMLU (57 diverse knowledge domains) faces challenges:
- Tasks vary significantly in difficulty and domain
- Models struggle with negative transfer between conflicting tasks
- Uniform sampling may waste compute on already-learned tasks

## Proposed Solution

### 1. Uncertainty-Weighted InfoNCE Loss

Standard contrastive learning uses InfoNCE to pull similar examples together and push dissimilar ones apart. We extend this with uncertainty weighting:

```
For each sample i with task t_i:
  - Positive pairs: samples from same task
  - Negative pairs: samples from different tasks
  
L_InfoNCE = -Σ w_i * log(exp(sim_pos/τ) / Σ exp(sim_neg/τ))

where w_i = H(p_i) / log(K)  (normalized prediction entropy)
```

**Rationale**: High-uncertainty samples are harder to classify and benefit more from contrastive alignment. Low-uncertainty samples need less contrastive adjustment.

### 2. Adaptive Curriculum Scheduler

Tracks inter-task confusion via embedding similarity and adjusts sampling weights:

```
Algorithm:
1. Compute embedding similarity between samples from different tasks
2. High similarity across tasks = confusion
3. Update task weights: w_t ∝ softmax(confusion_t / T)
4. Sample tasks according to weights
```

**Rationale**: Tasks that produce similar embeddings to other tasks are harder to disambiguate and should be sampled more frequently during training.

### 3. Task Confusion Matrix

Maintains a 57×57 matrix tracking cross-task similarities:

```
C[i,j] = Σ max(0, cos_sim(embed_i, embed_j)) for samples from task_i, task_j
```

This matrix reveals task relationships and guides curriculum updates.

## Architecture

**Model Components:**
- Base: DeBERTa-v3-base (184M params)
- Task embeddings: 57 × 768
- Contrastive head: 768 → 256
- Classification head: 1536 → 768 → 4

**Total Parameters:** ~185M (all trainable)

## Training Configuration

**Default Settings:**
- Batch size: 16
- Learning rate: 2e-5 with cosine decay
- Warmup: 10% of steps
- Mixed precision: bfloat16
- Curriculum warmup: 2 epochs (uniform sampling)
- Curriculum update frequency: every 100 steps

**Loss Weights:**
- Classification: 0.7
- Contrastive: 0.3

**Hardware:**
- Tested on: CUDA-capable GPU
- Training time: ~6 hours for 10 epochs on full MMLU (estimated)
- Memory: ~12GB VRAM

## Ablation Study Design

Compare three configurations:

1. **Baseline**: No curriculum, no contrastive loss (λ_contrast=0)
2. **Contrastive Only**: Add contrastive loss, uniform sampling
3. **Full Model**: Contrastive loss + adaptive curriculum

**Metrics:**
- Overall accuracy
- Per-task accuracy
- Task confusion matrix
- Training dynamics (loss curves)

## Implementation Details

**Key Design Decisions:**

1. **InfoNCE vs. other losses**: InfoNCE is theoretically grounded and widely validated for contrastive learning.

2. **Curriculum strategy**: Uncertainty gradient focuses on confused tasks rather than just difficult tasks, promoting better task disambiguation.

3. **Temperature parameters**:
   - Contrastive τ=0.07: Standard value from SimCLR
   - Curriculum T=2.0: Smooth task distribution, avoid over-focusing

4. **No data augmentation**: MMLU is text-based QA; standard augmentations (dropout, mixup) less applicable than in vision domains.

## Expected Outcomes

Based on related work in multi-task learning and curriculum learning:

1. **Contrastive loss** should improve cross-task knowledge transfer by ~2-3% over baseline
2. **Adaptive curriculum** should improve sample efficiency, reaching same accuracy in fewer epochs
3. **Combined approach** should show strongest performance, especially on confusing task pairs

## Reproducibility

All hyperparameters are specified in YAML configs. Random seed fixed at 42. Exact model version pinned in requirements.txt.

## Limitations

1. **Computational cost**: Contrastive loss requires pairwise comparisons (O(n²) in batch)
2. **Curriculum overhead**: Tracking confusion matrix and updating weights adds ~5% training time
3. **MMLU-specific**: Approach designed for multi-domain QA; generalization to other multi-task settings requires validation
4. **Requires sufficient batch diversity**: Small batches may not contain enough cross-task pairs for effective contrastive learning

## Future Work

- Test on other multi-task benchmarks (SuperGLUE, GLUE)
- Experiment with hard negative mining
- Investigate task clustering for hierarchical curriculum
- Explore meta-learning for task weight initialization
