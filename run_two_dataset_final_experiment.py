from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import random
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = PROJECT_DIR / "two_dataset_final_artifacts"

KVASIR_DATA_PATH = PROJECT_DIR / "clip_caption_artifacts" / "closed_answer_captioned_stable.csv"
CLIP_EMBEDDINGS_PATH = PROJECT_DIR / "clip_caption_artifacts" / "clip_vit_b32_image_embeddings_fp16.npy"
CLIP_IMAGE_IDS_PATH = PROJECT_DIR / "clip_caption_artifacts" / "clip_vit_b32_image_embedding_img_ids.json"

X1_DATASET_NAME = "SimulaMet/Kvasir-VQA-x1"
SEED = 42


@dataclass(frozen=True)
class TrainConfig:
    seed: int = SEED
    hidden_dim: int = 256
    question_dim: int = 48
    text_svd_dim: int = 96
    text_max_features: int = 3000
    dropout: float = 0.30
    batch_size: int = 256
    epochs: int = 15
    patience: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    class_weight_power: float = 0.5
    max_class_weight: float = 6.0
    use_question_mask: bool = True
    use_class_weights: bool = True
    optimizer_name: str = "adamw"


DEFAULT_CONFIG = TrainConfig()


class FeatureDataset(Dataset):
    def __init__(self, x: np.ndarray, question_idx: np.ndarray, answer_idx: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.question_idx = torch.tensor(question_idx, dtype=torch.long)
        self.answer_idx = torch.tensor(answer_idx, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.answer_idx)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "question_idx": self.question_idx[index],
            "answer_idx": self.answer_idx[index],
        }


class FusionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_questions: int,
        num_answers: int,
        hidden_dim: int,
        question_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.question_embedding = nn.Embedding(num_questions, question_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim + question_dim),
            nn.Linear(hidden_dim + question_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_answers),
        )

    def forward(self, x: torch.Tensor, question_idx: torch.Tensor) -> torch.Tensor:
        feature_out = self.feature_net(x)
        question_out = self.question_embedding(question_idx)
        return self.classifier(torch.cat([feature_out, question_out], dim=1))


def clean_answer(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.replace(" ;", ";").replace("; ", ";")


def clean_question(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def parse_original_items(value: Any) -> list[dict[str, str]]:
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        try:
            loaded = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return []
    return loaded if isinstance(loaded, list) else []


def configure(seed: int, artifact_dir: Path) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    artifact_dir.mkdir(exist_ok=True)
    (artifact_dir / "plots").mkdir(exist_ok=True)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_clip_embeddings() -> tuple[np.ndarray, dict[str, int]]:
    embeddings = np.load(CLIP_EMBEDDINGS_PATH).astype("float32")
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    image_ids = json.loads(CLIP_IMAGE_IDS_PATH.read_text(encoding="utf-8"))
    return embeddings, {img_id: index for index, img_id in enumerate(image_ids)}


def load_kvasir_rows(smoke: bool = False) -> pd.DataFrame:
    df = pd.read_csv(KVASIR_DATA_PATH, keep_default_na=False)
    df = df[["qa_id", "img_id", "question", "answer", "split", "caption_text"]].copy()
    df["dataset"] = "Kvasir-VQA"
    df["source_split"] = df["split"]
    df["question"] = df["question"].map(clean_question)
    df["answer"] = df["answer"].map(clean_answer)
    df["question_type"] = df["question"]
    df["caption_text"] = df["caption_text"].fillna("")
    df["caption_available"] = df["caption_text"].ne("")
    df["image_ref"] = df["img_id"].map(
        lambda img_id: f"https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-x1/resolve/main/images/{img_id}.jpg"
    )
    if smoke:
        keep_images = df["img_id"].drop_duplicates().head(240)
        df = df[df["img_id"].isin(keep_images)].copy()
    return df.reset_index(drop=True)


def build_x1_rows(artifact_dir: Path, smoke: bool = False, force: bool = False) -> pd.DataFrame:
    cache_path = artifact_dir / "kvasir_vqa_x1_original_qa_rows.csv"
    if cache_path.exists() and not force:
        df = pd.read_csv(cache_path, keep_default_na=False)
        if smoke:
            keep_images = df["img_id"].drop_duplicates().head(240)
            df = df[df["img_id"].isin(keep_images)].copy()
        return df.reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for hf_split in ["train", "test"]:
        dataset = load_dataset(X1_DATASET_NAME, split=hf_split, streaming=True)
        for row_index, row in enumerate(dataset):
            question_classes = row.get("question_class") or []
            if isinstance(question_classes, str):
                question_classes = [question_classes]
            for item_index, item in enumerate(parse_original_items(row.get("original", ""))):
                question = clean_question(item.get("q", ""))
                answer = clean_answer(item.get("a", ""))
                if not question or not answer:
                    continue
                rows.append(
                    {
                        "qa_id": f"x1_{hf_split}_{row_index}_{item_index}",
                        "dataset": "Kvasir-VQA-x1",
                        "source_split": hf_split,
                        "split": hf_split,
                        "img_id": row["img_id"],
                        "image_ref": row["image"],
                        "question": question,
                        "answer": answer,
                        "question_type": str(question_classes[item_index])
                        if item_index < len(question_classes)
                        else question,
                        "complexity": int(row.get("complexity", 0)),
                        "caption_text": "",
                        "caption_available": False,
                    }
                )
            if smoke and row_index >= 400:
                break

    df = pd.DataFrame(rows)
    if not smoke:
        df.to_csv(cache_path, index=False)
    return df.reset_index(drop=True)


def add_x1_train_val_split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    df = df.copy()
    train_images = sorted(df.loc[df["source_split"].eq("train"), "img_id"].unique())
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(train_images)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * 0.15)))
    val_images = set(shuffled[:val_count])
    mask = df["source_split"].eq("train")
    df.loc[mask, "split"] = np.where(df.loc[mask, "img_id"].isin(val_images), "val", "train")
    df.loc[df["source_split"].eq("test"), "split"] = "test"
    return df


def keep_embedding_covered_rows(df: pd.DataFrame, id_to_embedding_row: dict[str, int]) -> pd.DataFrame:
    return df[df["img_id"].isin(id_to_embedding_row)].copy().reset_index(drop=True)


def smoke_limit_frame(df: pd.DataFrame, max_images_per_dataset: int = 220) -> pd.DataFrame:
    parts = []
    for _, group in df.groupby("dataset", sort=False):
        keep = group["img_id"].drop_duplicates().head(max_images_per_dataset)
        parts.append(group[group["img_id"].isin(keep)])
    return pd.concat(parts, ignore_index=True)


def prepare_metadata(artifact_dir: Path, smoke: bool = False, force_x1: bool = False) -> dict[str, pd.DataFrame]:
    _, id_to_embedding_row = load_clip_embeddings()
    kvasir_df = keep_embedding_covered_rows(load_kvasir_rows(smoke=smoke), id_to_embedding_row)
    x1_df = keep_embedding_covered_rows(build_x1_rows(artifact_dir, smoke=smoke, force=force_x1), id_to_embedding_row)
    x1_df = add_x1_train_val_split(x1_df)
    if smoke:
        kvasir_df = smoke_limit_frame(kvasir_df)
        x1_df = smoke_limit_frame(x1_df)

    kvasir_df.to_csv(artifact_dir / "kvasir_vqa_final_rows.csv", index=False)
    x1_df.to_csv(artifact_dir / "kvasir_vqa_x1_final_rows.csv", index=False)
    combined = pd.concat([kvasir_df, x1_df], ignore_index=True)
    combined.to_csv(artifact_dir / "two_dataset_all_rows.csv", index=False)
    split_summary = (
        combined.groupby(["dataset", "split"], dropna=False)
        .agg(
            rows=("qa_id", "count"),
            images=("img_id", "nunique"),
            answers=("answer", "nunique"),
            questions=("question", "nunique"),
            caption_coverage=("caption_available", "mean"),
        )
        .reset_index()
    )
    split_summary.to_csv(artifact_dir / "two_dataset_split_summary.csv", index=False)
    return {"kvasir": kvasir_df, "x1": x1_df, "combined": combined}


def make_answer_vocab(train_df: pd.DataFrame) -> dict[str, int]:
    return {answer: index for index, answer in enumerate(sorted(train_df["answer"].unique()))}


def make_question_vocab(train_df: pd.DataFrame) -> dict[str, int]:
    questions = sorted(train_df["question_type"].fillna("").astype(str).unique().tolist())
    return {"[UNKNOWN_QUESTION]": 0, **{question: index + 1 for index, question in enumerate(questions)}}


def apply_vocabs(df: pd.DataFrame, answer_vocab: dict[str, int], question_vocab: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    out["answer_idx"] = out["answer"].map(answer_vocab)
    out["question_idx"] = out["question_type"].map(question_vocab).fillna(0).astype(int)
    return out[out["answer_idx"].notna()].copy().assign(answer_idx=lambda s: s["answer_idx"].astype(int))


def build_question_mask(train_df: pd.DataFrame, question_vocab: dict[str, int], answer_vocab: dict[str, int]) -> torch.Tensor:
    mask = torch.zeros((len(question_vocab), len(answer_vocab)), dtype=torch.bool)
    mask[0, :] = True
    for question_idx, answer_idx in train_df[["question_idx", "answer_idx"]].drop_duplicates().to_numpy():
        mask[int(question_idx), int(answer_idx)] = True
    for idx in range(mask.shape[0]):
        if not mask[idx].any():
            mask[idx, :] = True
    return mask


def build_features(
    train_df: pd.DataFrame,
    all_df: pd.DataFrame,
    embeddings: np.ndarray,
    id_to_embedding_row: dict[str, int],
    config: TrainConfig,
    use_captions: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    image_rows = [id_to_embedding_row[img_id] for img_id in all_df["img_id"]]
    image_features = embeddings[image_rows].astype("float32")

    def text_for(frame: pd.DataFrame) -> pd.Series:
        text = frame["question"].fillna("")
        if use_captions:
            text = text + " " + frame["caption_text"].fillna("")
        return text

    vectorizer = TfidfVectorizer(
        max_features=config.text_max_features,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    train_tfidf = vectorizer.fit_transform(text_for(train_df))
    all_tfidf = vectorizer.transform(text_for(all_df))
    svd_dim = min(config.text_svd_dim, max(1, train_tfidf.shape[1] - 1), max(1, train_tfidf.shape[0] - 1))
    svd = TruncatedSVD(n_components=svd_dim, random_state=config.seed)
    svd.fit(train_tfidf)
    text_features = svd.transform(all_tfidf).astype("float32")
    x = np.concatenate([image_features, text_features], axis=1).astype("float32")
    train_positions = all_df.index[all_df["row_role"].eq("train")].to_numpy()
    mean = x[train_positions].mean(axis=0, keepdims=True)
    std = x[train_positions].std(axis=0, keepdims=True) + 1e-6
    x = (x - mean) / std
    return x.astype("float32"), {
        "input_dim": int(x.shape[1]),
        "text_svd_dim": int(svd_dim),
        "text_features": int(len(vectorizer.get_feature_names_out())),
    }


def make_loader(frame: pd.DataFrame, x: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        FeatureDataset(
            x[frame.index.to_numpy()],
            frame["question_idx"].to_numpy(dtype=np.int64),
            frame["answer_idx"].to_numpy(dtype=np.int64),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def class_weight_tensor(train_df: pd.DataFrame, num_answers: int, config: TrainConfig, device: torch.device) -> torch.Tensor | None:
    if not config.use_class_weights:
        return None
    counts = np.bincount(train_df["answer_idx"].to_numpy(), minlength=num_answers).astype("float32")
    counts = np.maximum(counts, 1.0)
    weights = (counts.sum() / (num_answers * counts)) ** config.class_weight_power
    weights = np.clip(weights, 0.25, config.max_class_weight)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def masked_logits(logits: torch.Tensor, question_idx: torch.Tensor, answer_mask: torch.Tensor | None) -> torch.Tensor:
    if answer_mask is None:
        return logits
    row_mask = answer_mask.to(logits.device)[question_idx]
    return logits.masked_fill(~row_mask, -1e4)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    answer_mask: torch.Tensor | None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    labels: list[int] = []
    preds: list[int] = []
    confidences: list[float] = []
    for batch in loader:
        x = batch["x"].to(device)
        question_idx = batch["question_idx"].to(device)
        answer_idx = batch["answer_idx"].to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            raw_logits = model(x, question_idx)
            # Use unmasked logits for the loss so validation labels that were
            # unseen for a question type are not given an artificial -1e4 loss.
            # The question mask is still used for answer selection.
            loss = criterion(raw_logits, answer_idx)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        logits = masked_logits(raw_logits.detach(), question_idx, answer_mask)
        probabilities = torch.softmax(logits, dim=1)
        confidence, pred = probabilities.max(dim=1)
        total_loss += float(loss.item()) * len(answer_idx)
        labels.extend(answer_idx.detach().cpu().numpy().tolist())
        preds.extend(pred.cpu().numpy().tolist())
        confidences.extend(confidence.cpu().numpy().tolist())
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "accuracy": float(accuracy_score(labels, preds)) if labels else 0.0,
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)) if labels else 0.0,
        "labels": np.asarray(labels, dtype=np.int64),
        "preds": np.asarray(preds, dtype=np.int64),
        "confidence": np.asarray(confidences, dtype=np.float32),
    }


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer_name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    x: np.ndarray,
    question_vocab: dict[str, int],
    answer_vocab: dict[str, int],
    config: TrainConfig,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, pd.DataFrame, torch.Tensor | None]:
    model = FusionMLP(
        input_dim=x.shape[1],
        num_questions=len(question_vocab),
        num_answers=len(answer_vocab),
        hidden_dim=config.hidden_dim,
        question_dim=config.question_dim,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(train_df, len(answer_vocab), config, device))
    optimizer = build_optimizer(model, config)
    answer_mask = build_question_mask(train_df, question_vocab, answer_vocab) if config.use_question_mask else None
    train_loader = make_loader(train_df, x, config.batch_size, shuffle=True)
    val_loader = make_loader(val_df, x, config.batch_size, shuffle=False)
    best_state = copy.deepcopy(model.state_dict())
    best_val_macro_f1 = -1.0
    stale = 0
    rows: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, answer_mask, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, answer_mask)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": config.learning_rate,
        }
        rows.append(row)
        print(row)
        if row["val_macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = row["val_macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    return model, criterion, pd.DataFrame(rows), answer_mask


def corpus_bleu_score(references: list[str], predictions: list[str], max_n: int = 4) -> float:
    clipped_counts = np.zeros(max_n, dtype="float64")
    total_counts = np.zeros(max_n, dtype="float64")
    ref_len = 0
    pred_len = 0
    for reference, prediction in zip(references, predictions):
        ref_tokens = tokenize(reference)
        pred_tokens = tokenize(prediction)
        ref_len += len(ref_tokens)
        pred_len += len(pred_tokens)
        for n in range(1, max_n + 1):
            ref_ngrams: dict[tuple[str, ...], int] = {}
            pred_ngrams: dict[tuple[str, ...], int] = {}
            for i in range(0, max(0, len(ref_tokens) - n + 1)):
                key = tuple(ref_tokens[i : i + n])
                ref_ngrams[key] = ref_ngrams.get(key, 0) + 1
            for i in range(0, max(0, len(pred_tokens) - n + 1)):
                key = tuple(pred_tokens[i : i + n])
                pred_ngrams[key] = pred_ngrams.get(key, 0) + 1
            clipped_counts[n - 1] += sum(min(count, ref_ngrams.get(key, 0)) for key, count in pred_ngrams.items())
            total_counts[n - 1] += max(1, sum(pred_ngrams.values()))
    precisions = (clipped_counts + 1.0) / (total_counts + 1.0)
    brevity = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / max(1, pred_len))
    return float(brevity * math.exp(np.log(precisions).mean()))


def simple_meteor_score(references: list[str], predictions: list[str]) -> float:
    scores = []
    for reference, prediction in zip(references, predictions):
        ref_tokens = tokenize(reference)
        pred_tokens = tokenize(prediction)
        if not ref_tokens or not pred_tokens:
            scores.append(0.0)
            continue
        ref_counts: dict[str, int] = {}
        for token in ref_tokens:
            ref_counts[token] = ref_counts.get(token, 0) + 1
        overlap = 0
        for token in pred_tokens:
            if ref_counts.get(token, 0) > 0:
                overlap += 1
                ref_counts[token] -= 1
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        scores.append(0.0 if precision + recall == 0 else (10 * precision * recall) / (recall + 9 * precision))
    return float(np.mean(scores)) if scores else 0.0


def rouge_l_score(references: list[str], predictions: list[str]) -> float:
    def lcs(a: list[str], b: list[str]) -> int:
        dp = [0] * (len(b) + 1)
        for token_a in a:
            prev = 0
            for j, token_b in enumerate(b, start=1):
                temp = dp[j]
                if token_a == token_b:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = temp
        return dp[-1]

    scores = []
    for reference, prediction in zip(references, predictions):
        ref_tokens = tokenize(reference)
        pred_tokens = tokenize(prediction)
        if not ref_tokens or not pred_tokens:
            scores.append(0.0)
            continue
        overlap = lcs(ref_tokens, pred_tokens)
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_model(
    experiment_name: str,
    model: nn.Module,
    criterion: nn.Module,
    eval_df: pd.DataFrame,
    x: np.ndarray,
    answer_mask: torch.Tensor | None,
    answer_vocab: dict[str, int],
    device: torch.device,
    artifact_dir: Path,
    use_captions: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    loader = make_loader(eval_df, x, 512, shuffle=False)
    metrics = run_epoch(model, loader, criterion, device, answer_mask)
    idx_to_answer = {idx: answer for answer, idx in answer_vocab.items()}
    pred_df = eval_df.copy()
    pred_df["pred_idx"] = metrics["preds"]
    pred_df["pred_answer"] = pred_df["pred_idx"].map(idx_to_answer)
    pred_df["gold_answer"] = pred_df["answer"]
    pred_df["correct"] = pred_df["answer_idx"].eq(pred_df["pred_idx"])
    pred_df["confidence"] = metrics["confidence"]
    pred_df["experiment"] = experiment_name
    pred_df["caption_used"] = bool(use_captions)
    pred_df["caption_available"] = pred_df["caption_text"].fillna("").ne("")
    pred_df.to_csv(artifact_dir / f"{experiment_name}_predictions.csv", index=False)
    labels = metrics["labels"]
    preds = metrics["preds"]
    references = pred_df["gold_answer"].astype(str).tolist()
    predictions = pred_df["pred_answer"].astype(str).tolist()
    result = {
        "experiment": experiment_name,
        "eval_rows": int(len(pred_df)),
        "test_loss": float(metrics["loss"]),
        "test_accuracy": float(accuracy_score(labels, preds)) if len(labels) else 0.0,
        "test_macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)) if len(labels) else 0.0,
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)) if len(labels) else 0.0,
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)) if len(labels) else 0.0,
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)) if len(labels) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)) if len(set(labels.tolist())) > 1 else 0.0,
        "bleu": corpus_bleu_score(references, predictions),
        "meteor": simple_meteor_score(references, predictions),
        "rouge_l": rouge_l_score(references, predictions),
    }
    with (artifact_dir / f"{experiment_name}_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(classification_report(labels, preds, output_dict=True, zero_division=0), f, indent=2)
    return result, pred_df


def question_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for question_type, group in pred_df.groupby("question_type", dropna=False):
        rows.append(
            {
                "question_type": question_type,
                "support": int(len(group)),
                "accuracy": float(accuracy_score(group["answer_idx"], group["pred_idx"])),
                "macro_f1": float(f1_score(group["answer_idx"], group["pred_idx"], average="macro", zero_division=0)),
                "precision": float(precision_score(group["answer_idx"], group["pred_idx"], average="macro", zero_division=0)),
                "recall": float(recall_score(group["answer_idx"], group["pred_idx"], average="macro", zero_division=0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["accuracy", "support"], ascending=[True, False])


def build_experiment_frames(
    train_source: pd.DataFrame,
    val_source: pd.DataFrame,
    eval_source: pd.DataFrame,
    answer_vocab: dict[str, int],
    question_vocab: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    train_df = apply_vocabs(train_source, answer_vocab, question_vocab)
    val_df = apply_vocabs(val_source, answer_vocab, question_vocab)
    eval_before = len(eval_source)
    eval_df = apply_vocabs(eval_source, answer_vocab, question_vocab)
    all_df = pd.concat(
        [train_df.assign(row_role="train"), val_df.assign(row_role="val"), eval_df.assign(row_role="eval")],
        ignore_index=True,
    )
    stats = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "eval_rows_before_filter": int(eval_before),
        "eval_rows_after_filter": int(len(eval_df)),
        "eval_rows_dropped_unseen_answer": int(eval_before - len(eval_df)),
        "answer_classes": int(len(answer_vocab)),
        "question_classes": int(len(question_vocab)),
    }
    return (
        all_df[all_df["row_role"].eq("train")].copy(),
        all_df[all_df["row_role"].eq("val")].copy(),
        all_df[all_df["row_role"].eq("eval")].copy(),
        all_df,
        stats,
    )


def run_single_experiment(
    experiment_name: str,
    train_source: pd.DataFrame,
    val_source: pd.DataFrame,
    eval_source: pd.DataFrame,
    use_captions: bool,
    artifact_dir: Path,
    config: TrainConfig,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    embeddings, id_to_embedding_row = load_clip_embeddings()
    answer_vocab = make_answer_vocab(train_source)
    question_vocab = make_question_vocab(train_source)
    train_df, val_df, eval_df, all_df, frame_stats = build_experiment_frames(
        train_source, val_source, eval_source, answer_vocab, question_vocab
    )
    if len(train_df) == 0 or len(val_df) == 0 or len(eval_df) == 0:
        raise ValueError(f"{experiment_name} has empty split after vocabulary filtering: {frame_stats}")
    x, feature_info = build_features(train_df, all_df, embeddings, id_to_embedding_row, config, use_captions)
    model, criterion, history_df, answer_mask = train_model(train_df, val_df, x, question_vocab, answer_vocab, config, device)
    history_df["experiment"] = experiment_name
    history_df.to_csv(artifact_dir / f"{experiment_name}_history.csv", index=False)
    metrics, pred_df = evaluate_model(
        experiment_name, model, criterion, eval_df, x, answer_mask, answer_vocab, device, artifact_dir, use_captions
    )
    question_metrics(pred_df).to_csv(artifact_dir / f"{experiment_name}_question_metrics.csv", index=False)
    metrics.update(
        {
            "dataset_train": sorted(train_source["dataset"].unique().tolist()),
            "dataset_eval": sorted(eval_source["dataset"].unique().tolist()),
            "use_captions": bool(use_captions),
            "config": asdict(config),
            "frame_stats": frame_stats,
            "feature_info": feature_info,
        }
    )
    with (artifact_dir / f"{experiment_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics, pred_df, history_df


def support_bin_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    answer_support = pred_df["gold_answer"].value_counts()
    frame = pred_df.copy()
    frame["support"] = frame["gold_answer"].map(answer_support)
    bins = [0, 4, 19, 99, 10**9]
    labels = ["1-4", "5-19", "20-99", "100+"]
    frame["support_bin"] = pd.cut(frame["support"], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for support_bin, group in frame.groupby("support_bin", observed=True):
        rows.append(
            {
                "support_bin": str(support_bin),
                "rows": int(len(group)),
                "answers": int(group["gold_answer"].nunique()),
                "accuracy": float(accuracy_score(group["answer_idx"], group["pred_idx"])),
                "macro_f1": float(f1_score(group["answer_idx"], group["pred_idx"], average="macro", zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def save_plots(metrics_df: pd.DataFrame, histories: list[pd.DataFrame], predictions: dict[str, pd.DataFrame], artifact_dir: Path) -> None:
    plot_dir = artifact_dir / "plots"
    sns.set_theme(style="whitegrid")
    history_df = pd.concat(histories, ignore_index=True)
    history_df.to_csv(artifact_dir / "two_dataset_all_histories.csv", index=False)
    selected = history_df[history_df["experiment"].isin(["kvasir_no_caption", "kvasir_caption", "x1_no_caption"])]
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=selected, x="epoch", y="val_macro_f1", hue="experiment", marker="o")
    plt.title("Validation macro-F1 for main experiments")
    plt.tight_layout()
    plt.savefig(plot_dir / "validation_macro_f1_main_experiments.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=selected, x="epoch", y="train_loss", hue="experiment", marker="o", linestyle="--")
    sns.lineplot(data=selected, x="epoch", y="val_loss", hue="experiment", marker="o")
    plt.title("Training and validation loss for main experiments")
    plt.tight_layout()
    plt.savefig(plot_dir / "loss_curves_main_experiments.png", dpi=160, bbox_inches="tight")
    plt.close()

    plot_df = metrics_df.melt(
        id_vars=["experiment"],
        value_vars=["test_accuracy", "test_macro_f1", "weighted_f1", "bleu", "meteor"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(13, 7))
    sns.barplot(data=plot_df, x="score", y="experiment", hue="metric")
    plt.xlim(0, 1)
    plt.title("Two-dataset final metric comparison")
    plt.tight_layout()
    plt.savefig(plot_dir / "two_dataset_metric_comparison.png", dpi=160, bbox_inches="tight")
    plt.close()

    pred_df = predictions.get("kvasir_caption", next(iter(predictions.values())))
    top_answers = pred_df["gold_answer"].value_counts().head(25).index.tolist()
    cm_df = pred_df[pred_df["gold_answer"].isin(top_answers) & pred_df["pred_answer"].isin(top_answers)]
    cm = confusion_matrix(cm_df["gold_answer"], cm_df["pred_answer"], labels=top_answers)
    plt.figure(figsize=(13, 11))
    sns.heatmap(cm, cmap="Blues", xticklabels=top_answers, yticklabels=top_answers)
    plt.title("Top 25 answer confusion matrix")
    plt.xlabel("Predicted answer")
    plt.ylabel("Gold answer")
    plt.tight_layout()
    plt.savefig(plot_dir / "confusion_matrix_top25.png", dpi=160, bbox_inches="tight")
    plt.close()


def pick_examples(frame: pd.DataFrame, condition: pd.Series, analysis_type: str, count: int = 5, ascending_conf: bool = False) -> pd.DataFrame:
    subset = frame[condition].copy()
    if subset.empty:
        return subset
    subset = subset.sort_values("confidence", ascending=ascending_conf).head(count)
    subset["selection_note"] = "unique examples"
    if 0 < len(subset) < count:
        subset = subset.sample(n=count, replace=True, random_state=SEED).reset_index(drop=True)
        subset["selection_note"] = "repeated because fewer than five unique examples were available"
    subset["analysis_type"] = analysis_type
    return subset


def build_caption_delta_examples(no_caption: pd.DataFrame, caption: pd.DataFrame, helped: bool) -> pd.DataFrame:
    merged = no_caption.merge(caption, on="qa_id", suffixes=("_no_caption", "_caption"))
    if helped:
        subset = merged[(~merged["correct_no_caption"]) & (merged["correct_caption"])].copy()
        analysis_type = "caption_helped"
    else:
        subset = merged[(merged["correct_no_caption"]) & (~merged["correct_caption"])].copy()
        analysis_type = "caption_hurt"
    if subset.empty:
        return pd.DataFrame()
    rows = []
    subset = subset.sort_values("confidence_caption", ascending=False).head(5)
    selection_note = "unique examples"
    if 0 < len(subset) < 5:
        subset = subset.sample(n=5, replace=True, random_state=SEED).reset_index(drop=True)
        selection_note = "repeated because fewer than five unique examples were available"
    for _, row in subset.iterrows():
        rows.append(
            {
                "analysis_type": analysis_type,
                "experiment": "caption_delta",
                "dataset": row["dataset_caption"],
                "qa_id": row["qa_id"],
                "img_id": row["img_id_caption"],
                "image_ref": row["image_ref_caption"],
                "question": row["question_caption"],
                "gold_answer": row["gold_answer_caption"],
                "pred_answer": row["pred_answer_caption"],
                "pred_answer_no_caption": row["pred_answer_no_caption"],
                "correct": bool(row["correct_caption"]),
                "confidence": float(row["confidence_caption"]),
                "question_type": row["question_type_caption"],
                "caption_text": row["caption_text_caption"],
                "selection_note": selection_note,
            }
        )
    return pd.DataFrame(rows)


def build_qualitative_examples(predictions: dict[str, pd.DataFrame], artifact_dir: Path) -> pd.DataFrame:
    rows = []
    main = predictions.get("kvasir_caption", predictions.get("kvasir_no_caption", next(iter(predictions.values()))))
    answer_support = main["gold_answer"].value_counts()
    weak_questions = main["question_type"].astype(str).str.contains(
        "color|colour|location|abnormal|landmark|instrument|polyp", case=False, regex=True, na=False
    )
    qualitative_sets = [
        pick_examples(main, main["correct"], "correct_predictions"),
        pick_examples(main, ~main["correct"], "incorrect_predictions"),
        pick_examples(main, main["correct"], "high_confidence_correct"),
        pick_examples(main, ~main["correct"], "high_confidence_wrong"),
        pick_examples(main, pd.Series(True, index=main.index), "low_confidence_ambiguous", ascending_conf=True),
        pick_examples(main, main["gold_answer"].map(answer_support).le(5), "rare_answer_examples", ascending_conf=True),
        pick_examples(main, main["gold_answer"].map(answer_support).ge(100), "frequent_answer_examples"),
        pick_examples(main, weak_questions, "weak_question_type_examples", ascending_conf=True),
    ]
    for key in ["kvasir_to_x1", "x1_to_kvasir"]:
        if key in predictions:
            frame = predictions[key]
            qualitative_sets.append(pick_examples(frame, frame["correct"], f"{key}_correct"))
            qualitative_sets.append(pick_examples(frame, ~frame["correct"], f"{key}_failed"))
    if "kvasir_no_caption" in predictions and "kvasir_caption" in predictions:
        qualitative_sets.append(build_caption_delta_examples(predictions["kvasir_no_caption"], predictions["kvasir_caption"], True))
        qualitative_sets.append(build_caption_delta_examples(predictions["kvasir_no_caption"], predictions["kvasir_caption"], False))

    for subset in qualitative_sets:
        if subset is None or subset.empty:
            continue
        keep_cols = [
            "analysis_type",
            "experiment",
            "dataset",
            "qa_id",
            "img_id",
            "image_ref",
            "question",
            "gold_answer",
            "pred_answer",
            "pred_answer_no_caption",
            "correct",
            "confidence",
            "question_type",
            "caption_text",
            "selection_note",
        ]
        for col in keep_cols:
            if col not in subset.columns:
                subset[col] = ""
        rows.append(subset[keep_cols].copy())
    qualitative = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    qualitative = cache_qualitative_images(qualitative, artifact_dir)
    qualitative.to_csv(artifact_dir / "two_dataset_qualitative_examples.csv", index=False)
    return qualitative


def cache_qualitative_images(qualitative: pd.DataFrame, artifact_dir: Path) -> pd.DataFrame:
    if qualitative.empty:
        return qualitative
    image_dir = artifact_dir / "qualitative_images"
    image_dir.mkdir(exist_ok=True)
    qualitative = qualitative.copy()

    # Cache Kvasir-VQA qualitative images from the original dataset stream.
    kvasir_targets = set(qualitative.loc[qualitative["dataset"].eq("Kvasir-VQA"), "img_id"].dropna().astype(str))
    cached_paths: dict[str, str] = {}
    if kvasir_targets:
        remaining = set(kvasir_targets)
        for row in load_dataset("SimulaMet-HOST/Kvasir-VQA", split="raw", streaming=True):
            img_id = str(row["img_id"])
            if img_id not in remaining:
                continue
            path = image_dir / f"{img_id}.jpg"
            row["image"].save(path)
            cached_paths[img_id] = str(path)
            remaining.remove(img_id)
            if not remaining:
                break

    # Cache Kvasir-VQA-x1 qualitative images from their public image URLs.
    x1_rows = qualitative[qualitative["dataset"].eq("Kvasir-VQA-x1")][["img_id", "image_ref"]].drop_duplicates()
    for _, row in x1_rows.iterrows():
        img_id = str(row["img_id"])
        if img_id in cached_paths:
            continue
        path = image_dir / f"{img_id}.jpg"
        if not path.exists():
            try:
                urllib.request.urlretrieve(str(row["image_ref"]), path)
            except Exception:
                continue
        if path.exists():
            cached_paths[img_id] = str(path)

    qualitative["image_ref"] = qualitative.apply(
        lambda row: cached_paths.get(str(row["img_id"]), row["image_ref"]),
        axis=1,
    )
    return qualitative


def run_all(smoke: bool = False, force_x1: bool = False, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict[str, Any]:
    config = TrainConfig(epochs=2, patience=1, batch_size=128, text_svd_dim=32, text_max_features=500) if smoke else DEFAULT_CONFIG
    device = configure(config.seed, artifact_dir)
    frames = prepare_metadata(artifact_dir, smoke=smoke, force_x1=force_x1)
    kvasir = frames["kvasir"]
    x1 = frames["x1"]
    experiments = {
        "kvasir_no_caption": {
            "train": kvasir[kvasir["split"].eq("train")],
            "val": kvasir[kvasir["split"].eq("val")],
            "eval": kvasir[kvasir["split"].eq("test")],
            "captions": False,
        },
        "kvasir_caption": {
            "train": kvasir[kvasir["split"].eq("train")],
            "val": kvasir[kvasir["split"].eq("val")],
            "eval": kvasir[kvasir["split"].eq("test")],
            "captions": True,
        },
        "x1_no_caption": {
            "train": x1[x1["split"].eq("train")],
            "val": x1[x1["split"].eq("val")],
            "eval": x1[x1["split"].eq("test")],
            "captions": False,
        },
        "kvasir_to_x1": {
            "train": kvasir[kvasir["split"].eq("train")],
            "val": kvasir[kvasir["split"].eq("val")],
            "eval": x1[x1["split"].eq("test")],
            "captions": False,
        },
        "x1_to_kvasir": {
            "train": x1[x1["split"].eq("train")],
            "val": x1[x1["split"].eq("val")],
            "eval": kvasir[kvasir["split"].eq("test")],
            "captions": False,
        },
        "combined_no_caption": {
            "train": pd.concat([kvasir[kvasir["split"].eq("train")], x1[x1["split"].eq("train")]], ignore_index=True),
            "val": pd.concat([kvasir[kvasir["split"].eq("val")], x1[x1["split"].eq("val")]], ignore_index=True),
            "eval": pd.concat([kvasir[kvasir["split"].eq("test")], x1[x1["split"].eq("test")]], ignore_index=True),
            "captions": False,
        },
    }
    metrics_rows = []
    histories = []
    predictions: dict[str, pd.DataFrame] = {}
    for name, spec in experiments.items():
        print(f"\nRunning {name}")
        metrics, pred_df, history_df = run_single_experiment(
            name,
            spec["train"],
            spec["val"],
            spec["eval"],
            bool(spec["captions"]),
            artifact_dir,
            config,
            device,
        )
        metrics_rows.append(metrics)
        predictions[name] = pred_df
        histories.append(history_df)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(artifact_dir / "two_dataset_experiment_comparison.csv", index=False)
    with (artifact_dir / "two_dataset_experiment_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, indent=2)
    for name, pred_df in predictions.items():
        support_bin_table(pred_df).to_csv(artifact_dir / f"{name}_support_bin_metrics.csv", index=False)
    pd.concat(predictions.values(), ignore_index=True).to_csv(artifact_dir / "two_dataset_all_predictions.csv", index=False)
    save_plots(metrics_df, histories, predictions, artifact_dir)
    qualitative = build_qualitative_examples(predictions, artifact_dir)
    summary = {
        "artifact_dir": str(artifact_dir),
        "smoke": bool(smoke),
        "device": str(device),
        "experiments": [row["experiment"] for row in metrics_rows],
        "qualitative_rows": int(len(qualitative)),
        "metrics_path": str(artifact_dir / "two_dataset_experiment_comparison.csv"),
    }
    with (artifact_dir / "two_dataset_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final two-dataset Kvasir VQA experiments.")
    parser.add_argument("--smoke", action="store_true", help="Run a small fast validation run.")
    parser.add_argument("--force-x1", action="store_true", help="Rebuild cached Kvasir-VQA-x1 parsed metadata.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    return run_all(smoke=args.smoke, force_x1=args.force_x1, artifact_dir=args.artifact_dir)


if __name__ == "__main__":
    main()
