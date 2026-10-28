---
name: fine-tuning-transformers-hf
description: Use when fine-tuning BERT, RoBERTa, BERTimbau, or an LLM with Hugging Face transformers, Trainer, peft (LoRA, QLoRA), or trl, when asked to "train more epochs" until a metric target is hit, to report the best checkpoint's score, to use the test set as eval_dataset, or when there is "no data to spare" for validation.
---

# Fine-Tuning Transformers with Hugging Face

## Overview

`load_best_model_at_end`, `EarlyStoppingCallback`, and "train more epochs until F1 > 0.85" are all **model selection**. Whatever set they read becomes a validation set. If that set is the test set, the reported F1 is the maximum of many noisy looks, so it is biased upward, and a bootstrap CI on it is precise about the wrong number. **Core principle:** three splits. Validation picks the epoch and checkpoint. Test is read **once per seed**, at the end. Report mean ± CI across **≥3 seeds**.

**Violating the letter is violating the spirit:** a label on the slide ("evaluation set used for selection") does not make a contaminated number a test result. For a demo, the default must still be the clean protocol.

## Protocol

| Element | Rule |
|---|---|
| Splits | Train / validation (~10%) / test (~20%), stratified. Dedupe texts first. With repeated authors, group the split (REQUIRED SUB-SKILL: detecting-data-leakage). "No data to spare": 10% of 5k is 500 texts, which is plenty for early stopping |
| `eval_dataset` | Validation only. The test set is never passed to `Trainer` before training ends |
| Selection | `eval_strategy` == `save_strategy`, `load_best_model_at_end=True`, `metric_for_best_model` on validation, `EarlyStoppingCallback(patience=2)`, high epoch cap. "Train more epochs" = raise the cap. It never means re-running until the test hits a target |
| Test | `trainer.evaluate(test, metric_key_prefix="test")` once, after training |
| Seeds | `set_seed(seed)` **before** `from_pretrained` (the classification head is random), plus `seed` and `data_seed` in `TrainingArguments`. Run ≥3 seeds and report mean ± std and a t-based 95% CI. One seed hides collapse to the majority class. REQUIRED SUB-SKILL: ensuring-reproducibility |
| Rigor | Compare against a baseline with a paired test (REQUIRED SUB-SKILL: evaluating-models-rigorously). Log params, seeds, and the data hash (REQUIRED SUB-SKILL: tracking-experiments-mlflow, or `report_to="mlflow"`) |
| LoRA / QLoRA | `LoraConfig(task_type=TaskType.SEQ_CLS, ...)`. For sequence classification, peft adds the classifier to `modules_to_save` automatically (verified with peft 0.21.2). Same split and selection protocol. QLoRA needs `bitsandbytes` and a CUDA GPU |
| LLM SFT (`trl`) | `SFTTrainer` with `eval_dataset` = validation. Score the task metric with generation on the held-out test, not `eval_loss` |

**API (verified transformers 5.21.0):** `Trainer(tokenizer=...)` and `evaluation_strategy` were **removed**. Use `processing_class=tokenizer` and `eval_strategy`. A base checkpoint (e.g. `neuralmind/bert-base-portuguese-cased`) gets a new head, and the "MISSING classifier" report is expected. A checkpoint that already has a head with another label count raises unless you pass `ignore_mismatched_sizes=True`.

Working code (three-way split, `train_one_seed`, `summarize_seeds`): `fine-tuning-transformers-hf_template.py`. It was run end to end on synthetic data with transformers 5.21.0, datasets 5.1.0, peft 0.21.2, trl 1.16.0, and accelerate 1.15.0.

**User insists on selecting on test:** keep it as an opt-in flag that is off by default. With the flag on, the result is stored and printed as `selection_f1`, never as `test_f1`.

**Under a deadline:**

| Can be cut | Cannot be cut |
|---|---|
| `max_length` | The validation split |
| The epoch cap | The single read of the test |
| `fp16=True` on GPU (it speeds training up, it does not change the protocol) | At least 3 seeds |
| MLflow and the paired test, **if** the reply names them as pending | The baseline |

1 hour, 5k texts, BERT-base, one T4 with fp16: about 1 min per epoch, so 3 seeds with early stopping fit.

**Clean F1 below the target:** report it as it is, with the baseline and the seed CI. Name the next steps, all chosen on validation: error analysis, more or better labels, and `lr` / `max_length`. A sub-target honest number is a result. A target hit by retrying on test is not.

## Rationalizations

| Excuse | Reality |
|---|---|
| "It's just a demo" | The number travels into decisions. The clean protocol costs about 500 texts, not time |
| "User decides, I just execute" | Then make the clean protocol the default and the user's version the flag. Do not ship the contaminated version as the default |
| "I'll label the number as optimistic" | A labeled biased number is still biased, and the label falls off the slide first |
| "Bootstrap CI on the test makes it rigorous" | The CI measures sampling noise, not selection bias. It is precise about the wrong number |
| "Seed doesn't matter for a demo" | Fine-tuning variance across seeds often exceeds the gap you will claim. One line per seed |
| "Train more epochs until it passes 0.85" | Retrying until the test passes is selection on the test. Raise the cap and let validation early-stop |
| "Run 5 seeds, report the best one" | The max over seeds is selection on the test. Report the mean ± CI of **all** seeds run |
| "LoRA generalizes better, rerun with eval=test" | A new method gets the same protocol. LoRA `r` / `alpha` are hyperparameters chosen on validation |

## Red Flags: Stop

- `eval_dataset=test_ds`, or `eval_df = test_df`
- A flag like `USE_CLEAN_HOLDOUT = False` as the default
- A loop that retrains until a test metric target is reached
- "Best seed" or a max over runs reported as the test score
- One seed, or `set_seed` after `from_pretrained`
- `Trainer(tokenizer=...)` or `evaluation_strategy=` (removed in transformers 5)

**All of these mean: split off validation, read the test once per seed, report across seeds.**
