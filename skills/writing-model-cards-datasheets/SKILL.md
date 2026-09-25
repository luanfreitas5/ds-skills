---
name: writing-model-cards-datasheets
description: Use when documenting a trained model or a dataset for other teams, reviewers, or publication, when writing a README for a models/ or data/ folder, when preparing a model for registry promotion or production, or when asked for "short" model documentation.
---

# Writing Model Cards and Datasheets

## Overview

**Core principle:** documentation tells the reader what the model/dataset is *for*, what it is *not* for, and where it *fails*. A card with only architecture and one metric invites misuse. Use the standard structures: Model Cards (Mitchell et al., 2019) and Datasheets for Datasets (Gebru et al., 2021).

## Rules

- **Fill templates, don't improvise structure:** `model_card_template.md` and `datasheet_template.md` in this skill directory. Copy to `reports/model_cards/<model>_v<N>.md` and `reports/datasheets/<dataset>.md`.
- **Never invent numbers.** Unknown value → `TODO(owner): <what is needed>`. A card with honest TODOs beats a complete-looking card with fabricated CIs.
- **Out-of-scope uses are mandatory.** Name concrete misuses (e.g. "not for denying care", "not validated outside São Paulo").
- **Slice table is mandatory** when the model affects people: metric ± CI per subgroup (REQUIRED SUB-SKILL: evaluating-models-rigorously, auditing-fairness).
- **Status line** at top: `DRAFT` / `APPROVED FOR <specific use>` / `DEPRECATED`. Each unfilled required section lists which use it blocks.
- "Short" means concise sections, not dropped sections. Every heading stays; a section may be one line.
- Version the card with the model (same version tag, same Git commit, same MLflow run id).
- Write in the team's language (pt-BR for this stack); keep section headings from the template.

## Minimum Required Sections

| Model Card | Datasheet |
|---|---|
| Model details (version, owner, date, algorithm, run id) | Motivation (why created, by whom, funding) |
| Intended use + users | Composition (instances, fields, label, missingness, sensitive fields) |
| Out-of-scope uses | Collection process (source, period, sampling, consent) |
| Factors (groups, environments) | Preprocessing (cleaning, anonymization, hash of raw) |
| Metrics + decision threshold + why | Uses (intended, discouraged) |
| Evaluation data | Distribution + license |
| Training data (→ datasheet link) | Maintenance (owner, update cadence, deletion requests) |
| Quantitative analysis per slice | LGPD: legal basis, PII handling, retention |
| Ethical considerations + fairness findings | |
| Caveats, limitations, known failure modes, monitoring | |

## Common Mistakes

| Mistake | Fix |
|---|---|
| Card = architecture + AUC | Use full template |
| Temporal caveats omitted (e.g. COVID years in training data) | List distribution shifts in Caveats |
| "Validated" claimed with only internal test split | Say "internal validation only; no external validation" |
| Card not updated on retrain | Card version = model version |
