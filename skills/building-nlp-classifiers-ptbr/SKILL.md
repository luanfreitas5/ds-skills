---
name: building-nlp-classifiers-ptbr
description: Use when classifying Brazilian Portuguese social media text (tweets, posts, comments, reviews) for sentiment, toxicity, or topic, when labels come from two or more human annotators, when text has slang, emojis, or misspellings, when asked to "clean" text by removing emojis and stopwords, or when choosing between TF-IDF and a Transformer (BERTimbau, XLM-RoBERTa).
---

# Building NLP Classifiers for pt-BR Social Media

## Overview

On social media text, the test score depends on three things before it depends on the model: **label quality** (do annotators agree?), **who wrote the post** (the same author in train and test), and **what "cleaning" deletes** (emojis and "não" carry the label). **Core principle:** gate the labels, split by author, normalize minimally, and climb a baseline ladder before fine-tuning.

**Violating the letter is violating the spirit:** "the board already approved RoBERTa" approves the model family. It does not approve skipping the ladder rung that shows what RoBERTa buys.

## Label Contract (before any model)

| Step | Rule |
|---|---|
| Agreement | Cohen's kappa (2 annotators) or Krippendorff's alpha (3+), **per class too**. Kappa < 0.6 → fix the annotation guide and re-annotate a sample. Do not model yet |
| Disagreements | Never resolve by "annotator A wins". That trains the model on one person's bias. Use adjudication by a third person, or drop them from training and keep a flag |
| Test labels | **Consensus or adjudicated only.** An unresolved disagreement used as ground truth is noise in the metric you report |
| Ceiling | Report kappa next to F1. A model F1 above human agreement is learning an annotator's habits, not sentiment |
| Hard cases | A consensus-only test holds the easier posts, so its F1 is optimistic. Adjudicate a sample of test-side disagreements and report it as a separate slice |
| Gate fails under a deadline | The deliverable becomes the agreement report (`compute_agreement`): kappa per class, the A × B matrix, and the label distribution per annotator. Add a bootstrap CI as in labeling-with-llms. Do not downgrade the gate to a warning |
| Only some classes pass | Narrowing to the classes that pass (e.g. drop "neutro") is a **scope change**. The stakeholder approves it, and it is reported as a different task, not as the 3-class result |

LLM pre-labeling or LLM adjudication: REQUIRED SUB-SKILL: labeling-with-llms.

## Split and Normalization

- **Split by author** with `StratifiedGroupKFold(groups=user_id)`. Put retweets and copypasta (identical normalized text from different authors) in the same group. REQUIRED SUB-SKILL: detecting-data-leakage
- **Normalize minimally.** Strip the `RT @user:` prefix. Replace URLs with `http` and `@mentions` with `@user` (this is also PII removal: REQUIRED SUB-SKILL: protecting-pii-lgpd). Shorten elongations ("gosteiiiii" → "gosteiii"). **Keep** emojis, case, punctuation ("!!!"), and negation
- **Emojis per tokenizer:**

  | Consumer | Emoji handling | Action |
  |---|---|---|
  | XLM-R | Real tokens | Keep the emoji |
  | BERTimbau | Becomes `[UNK]` | `emoji.demojize(text, language="pt", delimiters=(" :", ": "))` → `:rosto_furioso:` |
  | TF-IDF, default word analyzer | Dropped | Use the `char_wb` analyzer |

  "The tokenizer doesn't understand emoji" means map it, never delete it. Verified with transformers 5.21.0 and emoji 2.16.0. Check your own tokenizer with `tok.tokenize("não gostei 😡")`
- **Never remove stopwords for sentiment.** spaCy's Portuguese `STOP_WORDS` contains **"não" and "nem"** (verified spaCy 3.8.18), so "não gostei" becomes "gostei". Transformers were pretrained on raw text, so stripping only hurts them

## Baseline Ladder (same split, same test)

REQUIRED SUB-SKILL: establishing-baselines. Report every rung's F1-macro on the same test set:

1. Majority class (`DummyClassifier`)
2. TF-IDF **word (1-2) + char_wb (2-5)** n-grams + `LogisticRegression(class_weight="balanced")`. Char n-grams absorb slang, misspellings, and emojis with no dictionary. Runs in seconds
3. Frozen `sentence-transformers` embeddings (e.g. `paraphrase-multilingual-MiniLM-L12-v2`) + logistic regression. Minutes on CPU
4. Fine-tuning: **BERTimbau** (`neuralmind/bert-base-portuguese-cased`, pt-BR native) or **XLM-RoBERTa** (multilingual). With no model chosen yet, try BERTimbau first. With one already approved, fine-tune that one and list the other as a next step. REQUIRED SUB-SKILL: fine-tuning-transformers-hf

**Model still open:** if fine-tuning beats rung 2 or 3 by less than the bootstrap CI of the difference, ship the cheaper model. **Model already chosen by a stakeholder:** run the ladder as measurement, not as a veto. Report the gain with its CI, and say plainly when the CI includes zero.

**Report:** F1-macro on test (consensus labels, unseen authors) with a 95% CI from a **bootstrap over authors**, not over posts. Add per-class F1, the kappa, and every ladder rung. REQUIRED SUB-SKILL: evaluating-models-rigorously

Working code (normalization, kappa gate with per-class kappa, author+text grouping, ladder). Filter to consensus posts before `split_by_author`; it raises on NaN labels. `split_by_author` logs the test share and the top author's share of the test. Code: `building-nlp-classifiers-ptbr_template.py`. Verified on synthetic data with scikit-learn 1.9.1, spaCy 3.8.18, and sentence-transformers 6.2.0.

## Rationalizations

| Excuse | Reality |
|---|---|
| "I paid for two annotators, use A when they disagree" | That money bought agreement information. Kappa takes one line. A-wins throws it away and puts A's bias into the test labels |
| "Kappa is computed, done" | Computing it is not gating on it. Below 0.6 the label scheme is the problem, and no model fixes it |
| "TF-IDF is 2015" | It is the reference that tells the board what RoBERTa adds. It costs 10 seconds |
| "Cleaning makes the text cleaner" | It deletes the signal: emojis, "!!!", "não". Normalize, do not clean |
| "Random 80/20 is standard" | With 200+ posts per author, the model memorizes authors. The random-split F1 will not repeat in production |
| "XLM-R was approved, so the ladder would reopen the decision" | The ladder measures, it does not veto. The board decides better with the gain and its CI in front of it |
| "Gate failed, but the deadline is tomorrow, so I'll make it a warning" | A model on labels with kappa < 0.6 measures noise. The agreement report is tomorrow's honest deliverable |

## Red Flags: Stop

- `label = annotator_a` with no kappa gate
- Disagreed posts in the test set
- `train_test_split` on posts with repeated authors
- Removing emojis, punctuation, or a stopword list containing "não"
- Fine-tuning with no TF-IDF or embeddings rung on the same split
- Giving the user an F1 number before the code has run

**All of these mean: apply the label contract and climb the ladder before reporting a number.**
