# Seagull: A Large Language Transformer Model that does Humorous Caption Generation  

CS 4740 Assignment

A decoder-only transformer (~110M parameters) fine-tuned to generate humorous captions from scene descriptions, inspired by the [New Yorker Caption Contest](https://www.newyorker.com/cartoons/contest). The architecture is heavily inspired by [Meta's LLaMA-2](https://arxiv.org/pdf/2307.09288.pdf) and implements multi-head attention, RoPE positional embeddings, RMSNorm, and a causal attention mask.

Given a scene and an "uncanny" description, the model generates a humorous caption:
> **Scene**: Two explorers are hiking through nature. They come across two subway cars in the brush.  
> **Uncanny**: Subway cars wouldn't be way out in the middle of nowhere.  
> **Caption**: *Of all the things we packed, I didn't think to bring my MetroCard.*

---

## Repository Structure

```
Seagull/
├── hw3.ipynb                  # Main notebook — run this
├── requirements.txt
├── seagull/                   # Model source code
│   ├── data_processing/       # Tokenization, dataset processing
│   ├── model/                 # Transformer architecture (MHA, FFN, embeddings)
│   └── utils/                 # Metrics, logging, torch utilities
├── scripts/                   # Training and evaluation scripts
├── dataset/                   # Arrow-format dataset (scene, uncanny, caption)
├── pretrained_artefacts/      # Base pretrained model weights (auto-downloaded)
└── artefacts/                 # Fine-tuned experiment checkpoints (auto-downloaded)
```

---

## Running Locally

### 1. Clone the repo

```bash
git clone https://github.com/JimmyChen02/Seagull.git
cd Seagull
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

```bash
jupyter notebook hw3.ipynb
```

### 4. Download model weights

The first cell of `hw3.ipynb` will **automatically download** the pretrained and fine-tuned model weights from Google Drive — no manual steps needed. Just run it and the weights will be placed in the correct directories.

---

## Running on Google Colab

1. Upload the repo to your Google Drive under `MyDrive/cs4740-hw3`
2. Open `hw3.ipynb` in Colab
3. Run all cells top to bottom — weights download automatically
