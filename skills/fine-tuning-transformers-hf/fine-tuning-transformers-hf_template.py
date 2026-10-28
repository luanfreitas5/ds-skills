"""Fine-tuning com Trainer do Hugging Face: validação para seleção, teste lacrado, várias seeds.

Verificado com transformers 5.21.0, datasets 5.1.0, peft 0.21.2 e accelerate 1.15.0 (dados sintéticos).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


def split_train_val_test(
    df: pd.DataFrame, seed: int = 42, val_size: float = 0.1, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split estratificado em treino, validação (seleção) e teste (lacrado).

    Com autores ou entidades repetidas, troque por StratifiedGroupKFold
    (REQUIRED SUB-SKILL: detecting-data-leakage).

    Parameters
    ----------
    df : pd.DataFrame
        Colunas ``text`` e ``label`` (inteiro 0..n-1; mapeie strings antes).
    seed : int, optional
        Semente do split, fixa entre as seeds de treino, by default 42.
    val_size, test_size : float, optional
        Frações do total, by default 0.1 e 0.2.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Treino, validação e teste.
    """
    df = df.drop_duplicates(subset="text")  # texto repetido dos dois lados infla o teste
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val["label"], random_state=seed)
    logger.info(f"Split: treino={len(train)}, validação={len(val)}, teste={len(test)} (lacrado)")
    return train, val, test


def compute_f1_macro(eval_pred) -> dict[str, float]:
    """Métrica de seleção calculada na validação pelo Trainer."""
    logits, labels = eval_pred
    return {"f1_macro": f1_score(labels, np.argmax(logits, axis=-1), average="macro")}


def train_one_seed(
    model_name: str,
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    seed: int,
    output_dir: Path,
    max_epochs: int = 10,
    num_labels: int = 3,
) -> float:
    """Treina com early stopping na validação e avalia o teste uma única vez.

    Parameters
    ----------
    model_name : str
        Checkpoint do Hub, ex.: "neuralmind/bert-base-portuguese-cased".
    splits : tuple
        Treino, validação e teste de ``split_train_val_test``.
    seed : int
        Semente desta execução (inicialização da cabeça, ordem dos lotes, dropout).
    output_dir : Path
        Pasta dos checkpoints desta seed.
    max_epochs : int, optional
        Teto de épocas; quem para é o early stopping, by default 10.
    num_labels : int, optional
        Número de classes, by default 3.

    Returns
    -------
    float
        F1-macro no teste do checkpoint escolhido pela validação.
    """
    set_seed(seed)  # antes de criar o modelo: a cabeça de classificação é aleatória
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def to_dataset(frame: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(frame[["text", "label"]], preserve_index=False)
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=128), batched=True)

    train, val, test = (to_dataset(s) for s in splits)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=max_epochs,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",  # tem de casar com eval_strategy para load_best_model_at_end
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=2,
        seed=seed,
        data_seed=seed,
        fp16=torch.cuda.is_available(),
        report_to="none",  # ou "mlflow" (REQUIRED SUB-SKILL: tracking-experiments-mlflow)
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,  # nunca o teste: ele escolheria checkpoint e época
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_f1_macro,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    test_f1 = trainer.evaluate(test, metric_key_prefix="test")["test_f1_macro"]
    logger.info(f"Seed {seed}: melhor checkpoint={trainer.state.best_model_checkpoint}, F1 teste={test_f1:.4f}")
    return float(test_f1)


def summarize_seeds(scores: list[float]) -> dict[str, float]:
    """Média, desvio e IC 95% (t de Student) do F1 de teste entre seeds.

    Examples
    --------
    >>> summarize_seeds([0.80, 0.82, 0.78])["mean"]
    0.8
    """
    arr = np.asarray(scores)
    half = stats.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr))
    summary = {"mean": round(float(arr.mean()), 4), "std": float(arr.std(ddof=1)),
               "ci_low": float(arr.mean() - half), "ci_high": float(arr.mean() + half), "n_seeds": len(arr)}
    logger.info(f"F1 teste: {summary['mean']:.4f} ± {summary['std']:.4f} "
                f"(IC 95% {summary['ci_low']:.4f}–{summary['ci_high']:.4f}, {len(arr)} seeds)")
    return summary
