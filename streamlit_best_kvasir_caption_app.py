from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from datasets import load_dataset


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_DIR / "question_masked_neural_artifacts" / "question_masked_neural_best.pt"
DATA_PATH = PROJECT_DIR / "clip_caption_artifacts" / "closed_answer_captioned_stable.csv"
CLIP_EMBEDDINGS_PATH = PROJECT_DIR / "clip_caption_artifacts" / "clip_vit_b32_image_embeddings_fp16.npy"
CLIP_IDS_PATH = PROJECT_DIR / "clip_caption_artifacts" / "clip_vit_b32_image_embedding_img_ids.json"
VIT_EMBEDDINGS_PATH = PROJECT_DIR / "vit_caption_artifacts" / "vit_b16_image_embeddings_fp16.npy"
VIT_IDS_PATH = PROJECT_DIR / "vit_caption_artifacts" / "vit_b16_image_embedding_img_ids.json"
RESNET_EMBEDDINGS_PATH = PROJECT_DIR / "caption_enhanced_artifacts" / "resnet18_image_embeddings_fp16.npy"
RESNET_IDS_PATH = PROJECT_DIR / "caption_enhanced_artifacts" / "resnet18_image_embedding_img_ids.json"
SAMPLE_CASES_PATH = PROJECT_DIR / "sample_test_cases.csv"


class QuestionMaskedFusionMLP(nn.Module):
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
        features = self.feature_net(x)
        question_features = self.question_embedding(question_idx)
        return self.classifier(torch.cat([features, question_features], dim=1))


@st.cache_data
def load_metadata() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, keep_default_na=False)


@st.cache_data
def load_sample_cases() -> pd.DataFrame:
    if not SAMPLE_CASES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SAMPLE_CASES_PATH, keep_default_na=False)


@st.cache_resource
def load_checkpoint() -> dict:
    return torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)


def load_embedding_lookup(embedding_path: Path, ids_path: Path) -> dict[str, np.ndarray]:
    embeddings = np.load(embedding_path).astype("float32")
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    image_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    return {image_id: embeddings[index] for index, image_id in enumerate(image_ids)}


@st.cache_resource
def load_embedding_lookups() -> dict[str, dict[str, np.ndarray]]:
    return {
        "clip": load_embedding_lookup(CLIP_EMBEDDINGS_PATH, CLIP_IDS_PATH),
        "vit": load_embedding_lookup(VIT_EMBEDDINGS_PATH, VIT_IDS_PATH),
        "resnet": load_embedding_lookup(RESNET_EMBEDDINGS_PATH, RESNET_IDS_PATH),
    }


@st.cache_resource
def load_kvasir_images():
    return load_dataset("SimulaMet-HOST/Kvasir-VQA", split="raw")


def load_kvasir_images_safely():
    try:
        return load_kvasir_images()
    except Exception as error:
        st.warning(f"Image display needs access to the Kvasir-VQA image dataset. Prediction can still run. Details: {error}")
        return None


def build_answer_mask(frame: pd.DataFrame, question_to_idx: dict[str, int], answer_vocab: dict[str, int]) -> torch.Tensor:
    if "split" in frame.columns:
        frame = frame[frame["split"].eq("train")]

    mask = torch.zeros((len(question_to_idx), len(answer_vocab)), dtype=torch.bool)
    for _, row in frame.iterrows():
        question_idx = question_to_idx.get(row["question"])
        answer_idx = answer_vocab.get(row["answer"])
        if question_idx is not None and answer_idx is not None:
            mask[int(question_idx), int(answer_idx)] = True
    for index in range(mask.shape[0]):
        if not mask[index].any():
            mask[index, :] = True
    return mask


@st.cache_resource
def load_model_and_assets():
    checkpoint = load_checkpoint()
    config = checkpoint["selected_config"]
    answer_vocab = checkpoint["answer_to_idx"]
    question_to_idx = checkpoint["question_to_idx"]
    feature_payload = {
        "input_dim": int(checkpoint["model_state_dict"]["feature_net.1.weight"].shape[1]),
        "vectorizer": checkpoint["caption_vectorizer"],
        "svd": checkpoint["caption_svd"],
        "mean": checkpoint["feature_mean"],
        "std": checkpoint["feature_std"],
    }

    model = QuestionMaskedFusionMLP(
        input_dim=int(feature_payload["input_dim"]),
        num_questions=len(question_to_idx),
        num_answers=len(answer_vocab),
        hidden_dim=int(config["hidden_dim"]),
        question_dim=int(config["question_dim"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    idx_to_answer = {int(index): answer for index, answer in checkpoint["idx_to_answer"].items()}
    metadata = load_metadata()
    answer_mask = build_answer_mask(metadata, question_to_idx, answer_vocab)

    return model, config, answer_vocab, idx_to_answer, question_to_idx, feature_payload, answer_mask


def find_image(dataset, img_id: str):
    for row in dataset:
        if row["img_id"] == img_id:
            return row["image"]
    return None


def predict(img_id: str, question: str, caption: str) -> tuple[str, float, pd.DataFrame]:
    model, _, _, idx_to_answer, question_to_idx, feature_payload, answer_mask = load_model_and_assets()
    lookups = load_embedding_lookups()

    image_features = np.concatenate(
        [
            lookups["clip"][img_id],
            lookups["vit"][img_id],
            lookups["resnet"][img_id],
        ]
    ).reshape(1, -1)

    caption_tfidf = feature_payload["vectorizer"].transform([caption])
    caption_features = feature_payload["svd"].transform(caption_tfidf).astype("float32")
    features = np.concatenate([image_features, caption_features], axis=1).astype("float32")
    features = (features - feature_payload["mean"]) / feature_payload["std"]

    question_idx_value = question_to_idx.get(question)
    if question_idx_value is None:
        raise ValueError("The selected question is not in the trained question vocabulary.")

    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        question_idx = torch.tensor([question_idx_value], dtype=torch.long)
        logits = model(x, question_idx)
        row_mask = answer_mask[question_idx]
        logits = logits.masked_fill(~row_mask, -1e4)
        probabilities = torch.softmax(logits, dim=1)[0]
        top_probabilities, top_indices = torch.topk(probabilities, k=5)

    top_rows = []
    for probability, index in zip(top_probabilities.tolist(), top_indices.tolist()):
        top_rows.append({"answer": idx_to_answer[index], "confidence": probability})

    best = top_rows[0]
    return best["answer"], float(best["confidence"]), pd.DataFrame(top_rows)


st.set_page_config(page_title="Kvasir-VQA Qwen Caption Model", layout="wide")
st.title("Kvasir-VQA Qwen Caption Model")
st.caption("Best trainable model only: cached image embeddings + Qwen caption features + question-masked MLP.")

required_paths = [
    CHECKPOINT_PATH,
    DATA_PATH,
    CLIP_EMBEDDINGS_PATH,
    CLIP_IDS_PATH,
    VIT_EMBEDDINGS_PATH,
    VIT_IDS_PATH,
    RESNET_EMBEDDINGS_PATH,
    RESNET_IDS_PATH,
]
missing_paths = [path for path in required_paths if not path.exists()]

if missing_paths:
    st.error("The app needs the trained best-model artifacts before it can run.")
    st.write([str(path) for path in missing_paths])
    st.stop()

metadata = load_metadata()
model, config, _, _, question_to_idx, _, _ = load_model_and_assets()

st.sidebar.header("Model")
st.sidebar.write("Model:", "Question-masked neural caption MLP")
st.sidebar.write("Learning rate:", config["learning_rate"])
st.sidebar.write("Dropout:", config["dropout"])
st.sidebar.write("Final refit epochs:", load_checkpoint()["final_refit_epochs"])

sample_cases = load_sample_cases()
test_mode_options = ["Curated sample test cases", "Cached Kvasir image dropdown"] if not sample_cases.empty else ["Cached Kvasir image dropdown"]
test_mode = st.radio("Choose test mode", test_mode_options)

sample_image_path = None
case_note = None

if test_mode == "Curated sample test cases":
    case_labels = sample_cases["case_label"].tolist()
    selected_case_label = st.selectbox("Select curated test case", case_labels)
    sample_row = sample_cases[sample_cases["case_label"].eq(selected_case_label)].iloc[0]
    selected_img_id = sample_row["img_id"]
    selected_question = sample_row["question"]
    sample_image_path = PROJECT_DIR / sample_row["image_path"]
    case_note = sample_row["case_note"]
    st.info(case_note)
else:
    image_ids = metadata["img_id"].drop_duplicates().tolist()
    selected_img_id = st.selectbox("Select cached Kvasir image", image_ids)
    image_rows = metadata[metadata["img_id"].eq(selected_img_id)].copy()
    available_questions = image_rows["question"].drop_duplicates().tolist()
    selected_question = st.selectbox("Select question", available_questions)

image_rows = metadata[metadata["img_id"].eq(selected_img_id)].copy()
selected_row = image_rows[image_rows["question"].eq(selected_question)].iloc[0]
caption = selected_row["caption_text"]
ground_truth = selected_row["answer"]

left, right = st.columns([1, 1])

with left:
    st.subheader("Input")
    if sample_image_path is not None and sample_image_path.exists():
        st.image(str(sample_image_path), caption=selected_img_id, width="stretch")
    else:
        kvasir_images = load_kvasir_images_safely()
        image = find_image(kvasir_images, selected_img_id) if kvasir_images is not None else None
        if image is not None:
            st.image(image, caption=selected_img_id, width="stretch")
        else:
            st.info("Image preview is unavailable, but the cached model features for this image are available.")
    st.write("Question:", selected_question)
    st.write("Qwen caption:", caption)
    st.write("Ground truth:", ground_truth)

with right:
    st.subheader("Prediction")
    try:
        prediction, confidence, top5 = predict(selected_img_id, selected_question, caption)
        colour = "green" if prediction == ground_truth else "red"
        st.markdown(f"**Prediction:** :{colour}[{prediction}]")
        st.write("Confidence:", round(confidence, 4))
        st.dataframe(top5, width="stretch")
    except Exception as error:
        st.error(str(error))
