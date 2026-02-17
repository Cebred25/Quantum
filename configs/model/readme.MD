# Model Policies (Encoder + Heads + Loss + Ensemble)

This folder defines **model architecture and training semantics** for the QSAR predictor:
- shared 3D GNN encoder
- hierarchical task heads
- masked multi-task loss
- learned uncertainty weighting
- ensemble strategy + distillation behavior

Unlike chemistry policies, these configs typically do **not** invalidate chemistry caches
(conformers/graphs), but they **do** define what a “model run” means and should be recorded
for reproducibility and comparison.

---

## Folder layout

---

## What belongs here vs elsewhere

### Belongs here (model semantics)
- encoder architecture / hyperparameters
- head structures and grouping logic
- loss type and uncertainty weighting settings
- ensemble strategy and how uncertainty is combined

### Does NOT belong here (run-time knobs)
These usually live in `configs/train/` or `configs/inference/`:
- batch size
- learning rate schedule specifics
- distributed backend settings
- dataloader workers
- inference throughput settings

Reason: model policies should capture **what the model is**, while train/inference configs capture
**how you run it**.

---

## File descriptions

### `encoder.yaml`
Defines the shared 3D GNN trunk:
- type (EGNN / SchNet / equivariant MPNN)
- number of layers / hidden dim
- geometry handling (distance expansion, cutoffs)
- pooling/readout method
- memory knobs (mixed precision, checkpointing)
- optional pretrained loading

Changing this changes the learned representation and compute cost.

---

### `heads.yaml`
Defines the task heads and hierarchy:
- shared head dimension
- task head dimension
- number of layers and activation
- task groups and types (regression/classification)
- masking behavior for missing labels

Changing this changes how the representation maps to assays and how tasks interact.

---

### `loss.yaml`
Defines:
- regression/classification loss type
- masked loss behavior (missing labels)
- learned uncertainty weighting (heteroscedastic / Kendall-style)
- manual task weights fallback
- regularization defaults (weight decay, label smoothing)

Changing this changes optimization behavior and uncertainty meaning.

---

### `ensemble.yaml`
Defines how you produce epistemic uncertainty:
- deep ensemble (N independent models)
- snapshot ensembles (multiple checkpoints from one run)
- shared trunk multi-head approximation
- distillation (ensemble → student)
- adaptive inference (run fewer members on “easy” cases)
- how to aggregate aleatoric + epistemic uncertainty

Changing this changes both compute cost and uncertainty quality.

---

## Recommended experiment logging

For every training run, log:
- full contents of this folder (encoder/heads/loss/ensemble)
- git commit hash
- library versions (torch, rdkit, cuda)
- dataset ID + chemistry policy hash (from `qsar/configs/policies/`)

This lets you reproduce results and compare runs fairly.

---

## Compute-aware tips

- If training is too slow:
  - reduce `hidden_dim` or `num_layers`
  - reduce geometry kernel counts (RBF kernels)
  - use mixed precision + activation checkpointing
  - consider snapshot ensembles instead of full deep ensembles

- If inference is too slow:
  - enable adaptive inference
  - distill ensemble into a student
  - reduce ensemble members for production serving

---

## Minimal baseline to start with

A practical first baseline is:
- encoder: 256 dim, 6 layers
- heads: 2 layers, moderate dropout
- loss: masked + uncertainty weighting enabled
- ensemble: 5 members for research, distilled student for deployment