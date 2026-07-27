# Tackling OOV: Masked Language Models vs. Character Diffusion

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/Task-SemEval--2019%20Task%206-orange.svg)](https://aclanthology.org/N19-1144.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Abstract:** While Large Language Models (LLMs) have achieved remarkable success, generalizing effectively in noisy or out-of-vocabulary (OOV) domains remains a persistent bottleneck. This project presents a controlled experimental comparison between **Masked Language Models (MLM)** and **Discrete Character Diffusion Models (DLM)** on the Offensive Language Identification Dataset (OLID). Utilizing an iterative corrupt-and-reconstruct sequence strategy, DLM demonstrates superior capability in handling OOV tokens, achieving higher precision, recall, and overall F1-scores on OOV-prevalent offensive text classification.

---

## 📌 Key Findings

* **Robustness to Noise:** The Discrete Character Diffusion Model (DLM) outperforms Masked Language Modeling (MLM) on out-of-vocabulary instances.
* **Superior Offensive Text Classification:** DLM achieves the highest F1-scores on OOV-prevalent *Offensive* (`OFF`) classification labels.
* **Better Generalization:** Recall and precision metrics indicate that character-level diffusion builds a more complete sub-word/character understanding across both `OFF` and `NOT` (Non-offensive) classes.

---

## 📊 Experimental Results

Evaluated on the **OLID (SemEval-2019 Task 6: SubTask A)** classification task across two test settings: raw text (`base`) and spelling-corrected (`processed`).

<table>
  <thead>
    <tr>
      <th rowspan="3">Category</th>
      <th colspan="8">OFF</th>
      <th colspan="8">NOT</th>
      <th colspan="8">AVG.</th>
    </tr>
    <tr>
      <th colspan="2">UW</th>
      <th colspan="2">P</th>
      <th colspan="2">R</th>
      <th colspan="2">F1</th>
      <th colspan="2">UW</th>
      <th colspan="2">P</th>
      <th colspan="2">R</th>
      <th colspan="2">F1</th>
      <th colspan="2">UW</th>
      <th colspan="2">P</th>
      <th colspan="2">R</th>
      <th colspan="2">F1</th>
    </tr>
    <tr>
      <!-- OFF -->
      <th>R</th><th>P</th>
      <th>R</th><th>P</th>
      <th>R</th><th>P</th>
      <th>R</th><th>P</th>
      <!-- NOT -->
      <th>R</th><th>P</th>
      <th>R</th><th>P</th>
      <th>R</th><th>P</th>
      <th>R</th><th>P</th>
      <!-- AVG -->
      <th>R</th><th>P</th>
      <th>R</th><th>P</th>
      <th>R</th><th>P</th>
      <th>R</th><th>P</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>MLM</strong></td>
      <td rowspan="2">95.94</td>
      <td rowspan="2">95.47</td>
      <td>23.68</td>
      <td>25.28</td>
      <td>23.88</td>
      <td>25.10</td>
      <td>20.84</td>
      <td>23.80</td>
      <td rowspan="2">96.40</td>
      <td rowspan="2">95.40</td>
      <td>55.25</td>
      <td>52.65</td>
      <td>51.65</td>
      <td>51.13</td>
      <td><strong>48.68</strong></td>
      <td><strong>48.88</strong></td>
      <td rowspan="2">96.28</td>
      <td rowspan="2">95.68</td>
      <td>51.65</td>
      <td>50.83</td>
      <td>50.49</td>
      <td>52.06</td>
      <td><strong>46.65</strong></td>
      <td><strong>48.10</strong></td>
    </tr>
    <tr>
      <td><strong>DLM</strong></td>
      <!-- UW OFF rowspanned -->
      <td><strong>27.24</strong></td>
      <td><strong>34.01</strong></td>
      <td><strong>25.43</strong></td>
      <td><strong>25.99</strong></td>
      <td><strong>23.42</strong></td>
      <td><strong>23.84</strong></td>
      <!-- UW NOT rowspanned -->
      <td><strong>62.17</strong></td>
      <td><strong>64.54</strong></td>
      <td><strong>51.75</strong></td>
      <td><strong>52.31</strong></td>
      <td>46.42</td>
      <td>47.45</td>
      <!-- UW AVG rowspanned -->
      <td><strong>59.62</strong></td>
      <td><strong>52.25</strong></td>
      <td><strong>51.51</strong></td>
      <td><strong>65.40</strong></td>
      <td>46.56</td>
      <td>47.57</td>
    </tr>
  </tbody>
</table>

> **Legend:** **UW**: Unique Word Percentage | **P**: Precision | **R**: Recall | **F1**: F1-Score  
> **Columns:** **R** = Raw test set (`base.tsv`) | **P** = Preprocessed/Spelling-fixed test set (`processed.csv`)

---

# How to Reproduce Experiments

This guide provides step-by-step instructions to reproduce the experimental results from the project **"Tackling OOV: Masked Language Models vs Character Diffusion"**.

---

## 📋 Prerequisites

### Environment & Dependencies
Ensure you have Python 3.8+ and PyTorch installed. Key dependencies include:

```bash
pip install torch torchvision torchaudio numpy pandas tqdm
```

### Directory Structure Overview
Before running the commands, verify that your project structure matches:

```text
.
├── task_pipeline.py           # Main entry point for evaluation
├── data/
│   └── process.py             # Preprocessing script for dataset splits
├── dataset/
│   ├── train.tsv              # Training data
│   ├── val.tsv                # Validation data
│   └── test/
│       ├── base.tsv           # Raw text test split
│       └── processed.csv      # Spelling-corrected test split
├── discrete_diffusion/
│   ├── best_model.pth         # Character diffusion model checkpoint
│   └── discrete_diffusion.ipynb # Notebook for diffusion model training
├── nanoGPT_CoLi/
│   └── out-reddit-clean/
│       └── ckpt.pt            # nanoGPT base model checkpoint
└── mis/                       # Preprocessing and base corpus dump
    ├── corpus_process.py
    └── input.txt
```

---

## 🛠️ Step-by-Step Pipeline

### Step 1: Base Corpus Preparation
Place your raw pre-training corpus inside `mis/` as `input.txt`, then run `corpus_process.py` to prepare binary train and validation splits:

```bash
cd mis/
python corpus_process.py
cd ..
```

*Outputs generated in `mis/`: `train.bin`, `val.bin`, `meta.pkl`*

---

### Step 2: Train the MLM Base Model (nanoGPT)
1. Copy `train.bin`, `val.bin`, and `meta.pkl` into your local [nanoGPT](https://github.com/karpathy/nanoGPT) `data/` folder.
2. Follow the nanoGPT training routine to train the base model on your corpus.
3. Once training is complete, save the resulting checkpoint file into:
   ```text
   nanoGPT_CoLi/out-reddit-clean/ckpt.pt
   ```

---

### Step 3: Train the Character Diffusion Base Model (DLM)
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook discrete_diffusion/discrete_diffusion.ipynb
   ```
2. Execute all cells to train the discrete character diffusion model. (Refer to the notebook comments for parameter configurations).
3. Export the trained model weights to:
   ```text
   discrete_diffusion/best_model.pth
   ```

---

### Step 4: Prepare Test Sets
To generate the spelling-corrected test split alongside the raw text version:

```bash
python data/process.py
```

*This updates/creates the test files in `dataset/test/` (`base.tsv` and `processed.csv`).*

---

### Step 5: Execute Downstream Evaluation

Run `task_pipeline.py` to train and evaluate the downstream classification task across different model architectures and test settings.

#### Command Arguments
- `--model_type`: Model architecture (`dlm` for Character Diffusion, `mlm` for Masked Language Model / nanoGPT).
- `--test_set_type`: Evaluation set split (`base` for raw text, `processed` for spelling-fixed text).
- `--epoch`: Number of fine-tuning epochs (default: `50`).

#### Reproduction Commands

1. **Evaluate Character Diffusion (DLM) on Processed Test Set:**
   ```bash
   python task_pipeline.py --model_type dlm --test_set_type processed --epoch 50
   ```

2. **Evaluate Character Diffusion (DLM) on Base Raw Test Set:**
   ```bash
   python task_pipeline.py --model_type dlm --test_set_type base --epoch 50
   ```

3. **Evaluate Masked Language Model (MLM) on Processed Test Set:**
   ```bash
   python task_pipeline.py --model_type mlm --test_set_type processed --epoch 50
   ```

4. **Evaluate Masked Language Model (MLM) on Base Raw Test Set:**
   ```bash
   python task_pipeline.py --model_type mlm --test_set_type base --epoch 50
   ```

---

## 📈 Output Logs & Results

- **Evaluation Logs:** Check `logs/fine-tuning/` for logs corresponding to `discrete/` and `nanoGPT/`.
- **Predictions & Detailed Results:** TSV output files will be saved in `outputs/dlm/` and `outputs/mlm/`.
