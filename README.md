# Kvasir-VQA Caption-Augmented Medical VQA Project

This repository contains the code and cached artifacts for a deep learning project on gastrointestinal visual question answering using Kvasir-VQA and Kvasir-VQA-x1.

The main notebook is:

```text
Deep_Learning_Project_Kvasir_VQA_Two_Dataset_Final.ipynb
```

It contains the final project workflow, including data loading, caption loading, model training, validation evidence, quantitative analysis, qualitative examples, and the Streamlit demo section.

## Project Summary

The project tests whether image-only Qwen captions improve a closed-answer gastrointestinal VQA model on Kvasir-VQA. Qwen received only the endoscopy image, not the question or ground-truth answer, so the captions were used as image-derived descriptions rather than direct answer leakage.

Kvasir-VQA-x1 was used as a larger no-caption comparison dataset and for cross-dataset generalisation testing.

## Main Results

Kvasir-VQA caption model:

- Accuracy: `0.8517`
- Macro-F1: `0.2045`
- Weighted-F1: `0.8510`
- Balanced accuracy: `0.2213`

Kvasir-VQA no-caption ablation:

- Accuracy: `0.8413`
- Macro-F1: `0.1699`

Kvasir-VQA-x1 no-caption model:

- Accuracy: `0.9542`
- Macro-F1: `0.8730`

Kvasir-to-X1 transfer:

- Accuracy: `0.5021`
- Macro-F1: `0.0602`

## Repository Contents

Core files:

- `Deep_Learning_Project_Kvasir_VQA_Two_Dataset_Final.ipynb`: final executed notebook.
- `run_two_dataset_final_experiment.py`: script version of the final experiment.
- `streamlit_best_kvasir_caption_app.py`: Streamlit demo for the selected Kvasir caption model.
- `requirements-local.txt`: Python dependencies.
- `sample_test_cases.csv`: curated cases used by the Streamlit demo.
- `sample_test_images/`: images linked to the curated Streamlit cases.

Cached inputs and model artifacts:

- `clip_caption_artifacts/`: captioned Kvasir table, CLIP embeddings, and image IDs.
- `vit_caption_artifacts/`: ViT embeddings and image IDs.
- `caption_enhanced_artifacts/`: ResNet embeddings and image IDs.
- `question_masked_neural_artifacts/`: selected Kvasir caption model checkpoint, metrics, predictions, validation grid, and plots.
- `two_dataset_final_artifacts/`: final Kvasir, X1, and cross-dataset metrics, plots, predictions, and qualitative images.
- `Kvasir_Qwen_Caption_Experiment_Package.zip`: cached Qwen image-caption records referenced by the notebook.

## Setup

Install dependencies:

```bash
pip install -r requirements-local.txt
```

The notebook was executed with a local Python kernel named `Python (kvasir-vqa-local)`.

## Running The Notebook

Open:

```text
Deep_Learning_Project_Kvasir_VQA_Two_Dataset_Final.ipynb
```

The notebook includes saved outputs and cached artifacts. A full rerun may take time and may need access to the public Hugging Face datasets referenced in the notebook.

GitHub may fail to preview the notebook because it is an executed notebook with many outputs. If that happens, download it and open it locally in Jupyter, VS Code, or Colab.

## Running The Script

Full experiment:

```bash
python run_two_dataset_final_experiment.py
```

Smoke test:

```bash
python run_two_dataset_final_experiment.py --smoke --artifact-dir two_dataset_final_artifacts_smoke
```

## Running The Streamlit Demo

```bash
streamlit run streamlit_best_kvasir_caption_app.py
```

The app loads the saved checkpoint, cached image embeddings, cached captions, curated sample test cases, and the Kvasir image dropdown from the packaged artifacts.

## Notes

The raw datasets are not included in this repository. The project uses public dataset sources and cached compact features/results.

The cached Qwen image-caption records are included because the final notebook references them when loading image-only descriptions.
