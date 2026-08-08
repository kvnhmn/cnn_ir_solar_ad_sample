# IR Solar Module Anomaly Classifier — CNN from Scratch

A convolutional neural network built **from scratch** (no pretrained backbone) to classify
infrared images of photovoltaic modules as *normal* or *anomalous*. Includes a hand-written
training loop, honest evaluation (confusion matrix, precision/recall), and a per-class error
analysis that traces exactly where the model's recall leaks.

Trained and evaluated on [InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules)
(RaptorMaps, MIT-licensed): 20,000 IR images of PV modules, 24×40 px, 12 classes
(11 anomaly types + No-Anomaly), collapsed here to a binary normal/anomaly task.

## What this demonstrates

- A CNN implemented and trained from scratch — architecture, training loop (`forward → loss →
  backward → step`), and metrics all hand-written, no high-level framework doing the work.
- A leakage-safe, reproducible data pipeline with an explicit disjointness test.
- Evaluation beyond accuracy: the confusion matrix and per-class recall reveal a failure profile
  that accuracy alone hides.
- A root-cause error analysis: which *original* anomaly types the binary model misses, and why.

## Dataset

InfraredSolarModules provides 20,000 single-module IR crops (24×40 px, grayscale) with a class
label per image in `module_metadata.json`. Classes are balanced ~10k normal / ~10k anomaly across
11 anomaly types. For this project all anomaly types are merged into a single `anomaly` class
(binary). The dataset is **MIT-licensed**, so it carries no non-commercial restriction.

Not included in this repo — download it from the link above and point `--root` at the folder
containing `images/` and `module_metadata.json`.

## Model — `IRNet`

Three convolutional blocks (`Conv → ReLU → MaxPool`) followed by a small classifier head.
Input is single-channel IR (`in_channels=1`), matching the grayscale data.

The head outputs raw logits (no softmax) — `CrossEntropyLoss` consumes logits directly.

## Training

- Loss: `CrossEntropyLoss` · Optimizer: `Adam (lr=1e-3)` · Batch size: 64 · 20 epochs
- 80/20 train/val split with a fixed seed (disjoint, verified by test)

The loss starts near `ln(2) ≈ 0.69` (random guessing for two balanced classes) and drops steadily.
Train and validation loss fall together for most of training; the gap begins to widen around
epoch 10–12, marking the onset of mild overfitting. Validation accuracy plateaus near **0.89–0.90**.

## Evaluation (validation set, 4,000 images)

**Accuracy: 0.90**

The number that matters for an inspection task is **recall on the anomaly class: 0.84** — the model
misses 316 of 1983 real anomalies. Precision on anomalies is high (0.95, few false alarms), so the
model is *conservative*: it rarely cries wolf, but it lets defects through. For inspection this is
the unfavourable trade-off — a missed defect costs more than a false alarm.

## References & license

- Dataset: [InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules) (RaptorMaps), MIT.
- Code: MIT — see `LICENSE`.
