---
name: fine-tuning-vision-models
description: Use when fine-tuning or doing transfer learning with a pretrained vision backbone (ResNet, EfficientNet, ViT, ConvNeXt via torchvision or timm) on a small image dataset, when a CNN is trained from scratch on ~2k images, when freezing/unfreezing layers, when one learning rate is used for the whole network, or when no feature-extraction baseline exists.
---

# Fine-Tuning Vision Models

## Overview

With ~2k images, a CNN from scratch memorizes. A pretrained backbone already encodes edges, textures and shapes, but only if the input looks like what it was pretrained on and the first updates do not wreck its weights. **Core principle:** earn every unit of complexity. Order: (1) linear probe on frozen features (the baseline), (2) train the head only, (3) unfreeze the top blocks with discriminative LR. Stop at the first rung the validation set cannot distinguish from the next.

Tested with torch 2.x and torchvision 0.29.2 (resnet18, CPU, synthetic images). The `timm` and ViT lines are not run here.

## Protocol

| Step | Rule |
|---|---|
| 1. Split first | Group-aware split, test untouched (REQUIRED SUB-SKILL: splitting-image-datasets). Selecting epochs/LR on test is the same trap as in text (REQUIRED SUB-SKILL: fine-tuning-transformers-hf, "Protocol") |
| 2. Normalization | Use the **backbone's** mean/std and input size, never the dataset's own. Read it from `weights.transforms()` (torchvision) or `timm.data.resolve_data_config({}, model=m)`. Same transform in train (after augmentation), val, test and serving |
| 3. Baseline | `fc -> Identity`, extract features once, fit `LogisticRegression` on train, score on validation. This is the number to beat, and it is cheap (seconds). REQUIRED SUB-SKILL: establishing-baselines |
| 4. Head only | New head, backbone frozen **and its BatchNorm in eval** (see below). A few epochs |
| 5. Partial unfreeze | Unfreeze top blocks only (`layer4`, then `layer3`). LR per block: head `1e-3`, backbone `1e-4`, decayed ×0.5 per block toward the input; AdamW; linear warmup then cosine. One global high LR (e.g. `1e-2` SGD or `1e-3` Adam on everything) destroys the pretrained features in the first steps |
| 6. Small data | Augment (flip, random-resized-crop, mild color), weight decay, early stopping on validation. Full unfreeze only if validation beats the partial one |
| 7. Report | Baseline, head-only and fine-tuned on the same validation split, ≥3 seeds, mean ± CI (REQUIRED SUB-SKILL: evaluating-models-rigorously, ensuring-reproducibility). Log backbone name, weights tag, mean/std, LR per group (REQUIRED SUB-SKILL: tracking-experiments-mlflow) |

**BatchNorm trap (measured):** `requires_grad=False` does not stop `running_mean/var` from updating under `model.train()`: after one forward pass of 32 images, `bn1.running_mean` changed; with `set_train_mode` it did not. With small batches those drifting statistics also degrade features. Call `set_train_mode(model)` at the start of **every** epoch (not once before the loop), since `model.train()` flips them back.

**Wrong normalization:** `Normalize([0.5]*3, [0.5]*3)` or `/255` only will not raise an error. It shifts activations and can cost accuracy silently (not measured here), which is why the baseline must use the exact same transform as the fine-tune.

## Implementation

Reusable code: `fine-tuning-vision-models_template.py` (copy to `src/training/finetune.py`). Functions: `get_backbone_normalization`, `build_eval_transform`, `replace_head`, `extract_features`, `freeze_backbone`, `set_train_mode`, `build_param_groups`, `set_warmup_cosine_lr`.

```python
import logging
from sklearn.linear_model import LogisticRegression
from torch import nn
from torchvision import models

logger = logging.getLogger(__name__)
weights = models.ResNet18_Weights.IMAGENET1K_V1
transform = build_eval_transform(weights)            # media/desvio do backbone

# 1) baseline: sondagem linear nas features congeladas
probe = models.resnet18(weights=weights); probe.fc = nn.Identity()
X_tr, y_tr = extract_features(probe, train_loader)
X_va, y_va = extract_features(probe, val_loader)
acc = LogisticRegression(max_iter=1000).fit(X_tr, y_tr).score(X_va, y_va)
logger.info("Baseline sondagem linear: acc=%.3f", acc)

# 2) fine-tuning parcial com LR por camada
model = replace_head(models.resnet18(weights=weights), num_classes=3)
freeze_backbone(model)
for name, p in model.named_parameters():
    p.requires_grad = p.requires_grad or name.startswith("layer4")
optimizer = torch.optim.AdamW(build_param_groups(model))
for step, (x, y) in enumerate(batches):               # set_train_mode no inicio de cada epoca
    set_warmup_cosine_lr(optimizer, step, total_steps, warmup_steps=50)
```

Verified on 600 synthetic 64×64 images: `build_param_groups` produced groups at LR `5e-5`, `1e-4` and `1e-3` covering exactly the trainable parameters, the warmup/cosine schedule ran end to end, and the loop trained. The synthetic task is trivially separable (the linear probe scored 1.0), so it proves the mechanics, not the accuracy gain. Judge gains on your own validation split.

Pretrained weights download on first use (resnet18: ~45 MB). For `timm`, `timm.create_model(name, pretrained=True, num_classes=k)` plus `resolve_data_config`/`create_transform`; the same protocol applies. ViT/ConvNeXt: LayerNorm has no running stats, so only the LR decay and baseline rules matter, with `layer_decay` over transformer blocks.

## Common Mistakes

| Mistake | Fix |
|---|---|
| CNN from scratch on 2k images | Pretrained backbone; from-scratch only as a reported comparison |
| Dataset mean/std or `[0.5]` with a pretrained backbone | Backbone's stats from `weights.transforms()` |
| One LR for all layers | `build_param_groups`: head > top block > lower blocks, with warmup |
| `requires_grad=False` and call it frozen | Also keep BN in eval (`set_train_mode`, every epoch) |
| Fine-tune first, baseline never | Linear probe first; report it beside the result |
| Optimizer built before unfreezing | Rebuild it (or add param groups) after changing `requires_grad` |
| Augmentation applied to val/test | Eval transform has no random ops |
| One seed, best epoch read off test | ≥3 seeds, selection on validation |

## Rationalizations

| Excuse | Reality |
|---|---|
| "Deadline: skip the baseline, just fine-tune" | The probe takes seconds. Without it you cannot say fine-tuning helped |
| "I already spent days on the from-scratch CNN" | Sunk cost. Keep it as a row in the comparison table, not as the deliverable |
| "The lead said use lr=1e-2, normalization is minor" | Do the requested run, and add the discriminative-LR run beside it. The report shows both |
| "The dataset's own mean/std is more accurate" | The backbone's weights were learned under its own statistics |
| "Frozen layers do not change" | Their BatchNorm statistics do, unless in eval |

## Red Flags: Stop

- `Normalize` values that are not the backbone's
- A single `lr=` for `model.parameters()` on a pretrained network
- Training loss spikes or collapses to chance in the first epochs
- `model.train()` called after freezing, with no BN handling
- No feature-extraction number in the report
- Accuracy compared across different splits or one seed

**All of these mean: rebuild the baseline, fix the transform, and re-run with discriminative LR and validation-based selection.**
