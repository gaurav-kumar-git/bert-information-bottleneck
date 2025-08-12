# Investigating the Information Bottleneck in BERT for Intent Classification

This project presents an **empirical study** of **Tishby's Information Bottleneck (IB)** principle applied to a `bert-base-uncased` model. We examine how different **information compression strategies** affect the performance of an **intent classifier** on the **challenging CLINC-150 dataset**.

The main objective is to explore the trade-off between **compactness** of a sentence representation and the model’s **classification accuracy**, using three distinct bottleneck architectures.

---

## 📜 Table of Contents
- [Project Overview](#project-overview)
- [Methodology](#methodology)
  - [Part 1: Simple Bottleneck](#part-1-simple-bottleneck)
  - [Part 2: Autoencoder Bottleneck](#part-2-autoencoder-bottleneck)
  - [Part 3: Stochastic (Denoising) Autoencoder Bottleneck](#part-3-stochastic-denoising-autoencoder-bottleneck)
- [Dataset: CLINC-150](#dataset-clinc-150)
- [Setup & Installation](#setup--installation)
- [Running the Experiments](#running-the-experiments)
- [Results & Artifacts](#results--artifacts)
- [File Structure](#file-structure)

---

## 📌 Project Overview
Large Language Models like **BERT** produce high-dimensional embeddings that are powerful but potentially redundant for specific tasks.  
The **Information Bottleneck theory** suggests that an optimal model should compress information, keeping only what’s necessary for predicting the target label.

We investigate this by:
- **Freezing** a pre-trained BERT as a feature extractor.
- Adding **different bottleneck layers** on top of BERT’s `[CLS]` output.
- **Varying** bottleneck sizes: `[8, 32, 64, 128]`.
- Observing how this impacts classification accuracy.

---

## 🛠 Methodology

### Part 1: Simple Bottleneck
- Single linear layer projects **768 → bottleneck dimension**.
- Direct lossy compression — classifier must adapt to reduced features.

### Part 2: Autoencoder Bottleneck
- **Encoder**: 2-layer MLP compresses 768 → bottleneck.
- **Decoder**: 2-layer MLP reconstructs 768 from bottleneck.
- Loss = *Classification Loss* + α × *Reconstruction Loss*.

### Part 3: Stochastic (Denoising) Autoencoder Bottleneck
- Builds on Part 2.
- Adds **Gaussian noise** to bottleneck during training.
- Encourages **robust, generalizable** features.

---

## 📂 Dataset: CLINC-150
- ~23,000 utterances across **150 intents**.
- **+1 Out-of-Scope** class → total **151 classes**.
- **Data Cleaning**: Removes corrupted rows with `None` labels in `dataset.py`.
- Dataset used: **DeepPavlov/clinc150** ("plus" configuration).

---

## ⚙️ Setup & Installation

### 1️⃣ Clone Repository
```bash
git clone git@github.com:your-username/bert-information-bottleneck.git
cd bert-information-bottleneck
````

### 2️⃣ Create Conda Environment

```bash
conda env create -f environment.yml
conda activate info_bottleneck_env
```

### 3️⃣ Offline Setup (Important)

Manually download:

* **BERT model**: Place in `local_cache/bert-base-uncased/`
* **Dataset**: Place in `local_cache/DeepPavlov___clinc150/` (in `.arrow` format)

Paths are set in `config.py`.

---

## 🚀 Running the Experiments

Run **all experiments**:

```bash
python train.py
```

* Trains **all 3 parts** for all bottleneck sizes.
* Saves **best model** for each configuration.
* Generates comparison plots progressively.

---

## 📊 Results & Artifacts

Saved in `results/`:

* **Model weights**: `best_model_partX_dimY.pth`
* **Logs**: `experiment_results.json`
* **Plots**:

  * `comparison_after_part_1.png`
  * `comparison_after_part_2.png`
  * `comparison_final_all_parts.png`

---

## 📁 File Structure

```
.
├── config.py             # Central config (paths, hyperparams, NUM_LABELS)
├── dataset.py            # Data loading, cleaning, tokenization
├── models.py             # Model definitions for all 3 parts
├── train.py              # Main experiment runner
├── environment.yml       # Conda environment
├── local_cache/          # Offline storage for model + dataset
│   ├── bert-base-uncased/
│   └── DeepPavlov___clinc150/
└── results/              # Outputs: models, plots, logs
```

---
