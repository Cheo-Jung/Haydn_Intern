# Lab 1b Report: Classical and Neural Models for Multiclass Classification

## 1) Overview
This report summarizes the experiments implemented in `lab1b.ipynb`. The notebook trains multiple MLP architectures on three datasets (Blobs, Ellipsoids, Spirals), evaluates error/accuracy, visualizes decision boundaries, and implements a custom `ReLU` + `Linear`-based MLP.

## 2) Datasets
The three datasets are:
- **Blobs:** near-linearly separable compact classes.
- **Ellipsoids:** overlapping anisotropic clusters with moderate nonlinearity.
- **Spirals:** highly nonlinear interleaving classes.

Dataset preview figure:

![Dataset preview](figures_lab1b/datasets.png)

## 3) Part 2 Results (Multiple MLP Architectures)
Architectures evaluated:
- 2-16-3
- 2-32-32-3
- 2-64-64-3
- 2-128-128-3
- 2-64-64-64-3

The notebook exports:
- `figures_lab1b/mlp_error_table.csv`
- `figures_lab1b/mlp_accuracy_table.csv`

Decision-boundary figures:
- `figures_lab1b/boundaries_2-16-3.png`
- `figures_lab1b/boundaries_2-32-32-3.png`
- `figures_lab1b/boundaries_2-64-64-3.png`
- `figures_lab1b/boundaries_2-128-128-3.png`
- `figures_lab1b/boundaries_2-64-64-64-3.png`

### Training choices discussion
- **Learning rate:** Large values (e.g., 0.05) made spiral training unstable. Very small values (e.g., 0.001) trained too slowly. Values around `0.006–0.01` were reliable.
- **Optimizer:** Adam converged faster and more consistently than SGD in quick comparisons.
- **Epoch budget:** Blobs converged quickly; Spirals required the longest training.
- **Initialization sensitivity:** Deeper/wider models showed more variance with short training schedules.
- **Depth/width vs runtime:** Wider/deeper models improved nonlinear fit potential but increased compute cost.

### Reflection answers (Part 2)
1. **Why depth helps spirals but not blobs:** Spirals need compositional nonlinear warping; blobs are already close to linearly separable.
2. **Why wide shallow nets can work:** Width gives many nonlinear basis functions and can approximate complex boundaries even with limited depth.
3. **Hardest dataset:** Spirals, because classes are interleaved and require many local boundary turns.
4. **Depth/width effect on training:** More capacity increases sensitivity and training cost, but also potential performance when tuned.
5. **Do deeper always win?:** No—performance depends on optimization, data complexity, and hyperparameter tuning.

## 4) Part 3: Custom ReLU + Custom Linear + Custom MLP
Implemented components:
- `ReLU` via `torch.autograd.Function` (manual forward + backward).
- `Linear` via `torch.autograd.Function` (manual forward + backward).
- `My_MLP` using custom layers and custom activation.

Custom model boundary figure:

![Custom MLP boundaries](figures_lab1b/custom_mlp_boundaries.png)

### Forward-pass diagram (required)
```text
Input X (batch_size × 2)
    -> Linear1 W1:(2 × hidden), b1:(hidden)
       Output z1: (batch_size × hidden)
    -> ReLU
       Output a1: (batch_size × hidden)
    -> Linear2 W2:(hidden × hidden), b2:(hidden)
       Output z2: (batch_size × hidden)
    -> ReLU
       Output a2: (batch_size × hidden)
    -> Linear3 W3:(hidden × 3), b3:(3)
       Output logits: (batch_size × 3)
```

### Reflection answers (Part 3)
1. **No nonlinearity => linear model:** Composition of linear maps stays linear.
2. **Wrong ReLU backward for negatives:** It sends incorrect gradients through inactive units and corrupts optimization.
3. **Why `autograd.Function`:** It integrates custom derivatives into PyTorch’s graph cleanly and safely, without manual gradient loops.

## 5) Reproducibility
Run the notebook top-to-bottom to regenerate all figures and CSV tables used in this report.
