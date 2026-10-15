# Speech Command Classification with Wav2Vec2

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-F97316?style=flat-square&logo=gradio&logoColor=white)
![Test accuracy](https://img.shields.io/badge/test%20accuracy-95.37%25-success?style=flat-square)

Classifier for spoken commands ("yes", "no", "up", "down", digits, …) built by fine-tuning a pretrained
**Wav2Vec2** model on the [Speech Commands](https://arxiv.org/abs/1804.03209) dataset (35 classes).
The model reaches **95.37% accuracy on the test set**.

## Approach

- **Data:** `torchaudio.datasets.SPEECHCOMMANDS` with the official train/validation/test split. To fit
  the Kaggle GPU budget, every 10th sample of each split was used (a 10% subset; 1,101 test samples).
- **Exploration:** class distribution, waveforms and Mel spectrograms per class.
- **Model:** `facebook/wav2vec2-base` as the backbone, with the encoder frozen except for its **last 2
  transformer layers**, plus a classification head (BatchNorm → Dropout → Linear 768→384 → … → 35 classes).
- **Training:** AdamW (lr 1e-4, weight decay 1e-4), `ReduceLROnPlateau` on validation accuracy,
  cross-entropy loss, 20 epochs, keeping the best checkpoint by validation accuracy.
- **Evaluation:** accuracy, per-class precision/recall/F1 and confusion matrix on the held-out test set.

## Results

| Split | Accuracy |
|---|---|
| Validation (best epoch) | ~95% |
| **Test** | **95.37%** |

Macro F1 on the test set: 0.94.

## Repository

| File | Contents |
|---|---|
| [`speech-classification.ipynb`](speech-classification.ipynb) | Data loading, EDA, model, training and evaluation |
| [`deploy-audio-commands.ipynb`](deploy-audio-commands.ipynb) | Inference function + **Gradio** interface: upload a `.wav` and get the predicted command |

## Running

The notebooks were developed on Kaggle with a GPU. To run locally:

```bash
pip install torch torchaudio transformers gradio scikit-learn matplotlib seaborn
jupyter notebook speech-classification.ipynb
```

The training notebook downloads the dataset (~2.3 GB) automatically. The deploy notebook expects the
trained checkpoint `best_model.pth`.
