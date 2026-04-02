# Lab 1B Written Answers

## Part 2 — Training Choices (Discussion, by Architecture)

- **2-16-3**
  - Learning rate: `0.01` was okay, but this small model underfit spirals.
  - Epochs: needed close to full training to stabilize.
  - Initialization sensitivity: medium; different starts changed spiral quality.
  - Note: good for blobs, weaker on complex curved boundaries.

- **2-32-32-3**
  - Learning rate: `0.01` trained stably.
  - Epochs: converged faster than 2-16-3.
  - Initialization sensitivity: lower than the smallest model.
  - Note: solid balance of speed and accuracy across datasets.

- **2-64-64-3**
  - Learning rate: `0.01` worked well and converged smoothly.
  - Epochs: usually reached a good boundary before the end.
  - Initialization sensitivity: low-to-medium.
  - Note: strong performance on spirals without being too slow.

- **2-64-64-64-3**
  - Learning rate: `0.01` worked, but depth made training a bit slower.
  - Epochs: needed more epochs to fully settle than shallower models.
  - Initialization sensitivity: medium (deeper nets are slightly pickier).
  - Note: can model complex boundaries well, but not always better than 2-64-64-3.

- **2-128-128-3**
  - Learning rate: `0.005` was more stable than `0.01`.
  - Epochs: converged well, but each epoch cost more time.
  - Initialization sensitivity: low after lowering the learning rate.
  - Note: high capacity; good results but higher compute cost.

**Short comparison:**
- Small models train fast but can underfit spirals.
- Moderate depth/width gives the best trade-off.
- Bigger/deeper models are not automatically best; they need better tuning and more compute.

## Part 2 — Reflection Questions

1. **Why does adding depth help on the spiral dataset but not on the blobs dataset?**  
   Blobs are close to linearly separable, so even simple models already capture the boundary well. Spirals require multiple nonlinear transformations to “untangle” class manifolds, which depth provides by composing several learned feature mappings. Therefore, depth adds little on easy linear structure but helps significantly on complex geometry.

2. **Why might very wide networks succeed even when they have only one or two hidden layers?**  
   Width gives many parallel neurons that can approximate rich nonlinear functions even with limited depth. A sufficiently wide hidden layer can carve complex decision regions by combining many local features. This can compensate for less hierarchical feature composition.

3. **Which dataset was the hardest for the MLP, and what properties of the dataset made it difficult?**  
   The spiral dataset was hardest. Its classes are intertwined and non-convex, so good performance requires curved, high-complexity decision boundaries and careful optimization. Small or undertrained models tend to underfit this geometry.

4. **How did depth and width affect training, in terms of sensitivity to initialization and training time?**  
   More depth generally increased expressiveness but also made optimization slightly more sensitive and slower per full convergence. More width usually improved fitting and robustness but increased per-epoch computation. Moderate depth plus moderate-to-high width was the most stable configuration.

5. **Did deeper networks always perform better? Why or why not?**  
   No. On simple datasets, extra depth provides little benefit and may only add optimization overhead. Performance depends on matching model capacity to data complexity and training setup; beyond that point, deeper models can plateau or occasionally underperform if not tuned well.

## Part 3 — Reflection Questions

1. **Why does an MLP without nonlinearities collapse into a linear model?**  
   A composition of linear maps is still a linear map. If every layer is linear, the whole network is equivalent to one matrix multiplication plus bias, so it cannot represent nonlinear class boundaries.

2. **What would happen if ReLU backward incorrectly returned gradients for negative inputs?**  
   Neurons that should be inactive would still receive updates, which changes the effective function and breaks the true derivative. Training could become biased or unstable, and the model may converge to worse solutions because gradients no longer match the forward computation.

3. **Why is it useful to wrap forward/backward logic inside `torch.autograd.Function` instead of writing manual gradient loops?**  
   `torch.autograd.Function` integrates custom ops into PyTorch’s computation graph cleanly and efficiently. It keeps tensor operations vectorized, allows automatic chaining with other layers, and avoids error-prone manual per-parameter loop code. This is the standard way to define custom differentiable operations in PyTorch.
