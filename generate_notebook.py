"""
Builds and Executes the Complete Jupyter Notebook (`Machine_Translation_Systems.ipynb`)
for Foundations of Generative AI (ICT4403) - Assignment 3.
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient


def build_and_execute_notebook(output_path: str = "Machine_Translation_Systems.ipynb"):
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    cells = []

    # Title & Overview
    cells.append(
        new_markdown_cell(
            """# Foundations of Generative AI (ICT4403) — Assignment 3
## Comparative Neural Machine Translation: Basic Transformer, mT5, LLaMA, and Gemma
**Author**: Kumar Kshitij (Reg. No.: `230953488`)  
**Language Pair**: English (`en`) $\\rightarrow$ French (`fr`)  
**Dataset**: Tatoeba / OPUS English–French Parallel Corpus (`https://www.manythings.org/anki/fra-eng.zip` | `https://huggingface.co/datasets/Helsinki-NLP/opus-100`)  
**Repository**: `https://github.com/Kshiku12/Gen-AI-Assignment`

---

### Assignment Objectives
1. **Part (i) — Dataset Preprocessing & Model Development**:
   - Describe, clean, and preprocess the publicly available English–French parallel-text dataset and split it into training (80%), validation (10%), and test (10%) sets.
   - Implement and train a **Basic Transformer** (Encoder–Decoder) from scratch in PyTorch.
   - Fine-tune **mT5 (`google/mt5-small`)** (Encoder–Decoder) using supervised sequence-to-sequence learning.
   - Apply both **3-Shot In-Context Prompting** and **Parameter-Efficient LoRA Fine-Tuning** ($r=16, \\alpha=32$) to **LLaMA (`Llama-3.2-1B-Instruct`)** and **Gemma (`gemma-3-1b-it`)** (Decoder-Only).
   - Generate translations on the identical held-out test set under comparable decoding settings.
2. **Part (ii) — Quantitative & Architectural Evaluation**:
   - Evaluate all models using **Corpus BLEU**, **chrF**, **Training/Fine-Tuning Time**, **Inference Time & Latency**, and **Number of Model Parameters**.
   - Compare **Encoder–Decoder** vs. **Decoder-Only** architectures across model size, pretraining objective, tokenization strategy, and adaptation method.
   - Perform a **Sampling-Temperature Analysis** ($T \\in \\{0.0, 0.3, 0.7, 1.0\\}$) evaluating the fidelity–diversity trade-off.
3. **Part (iii) — Responsible & Ethical AI Analysis**:
   - Audit translations across all four models for **gender/occupational bias**, **hallucinated information**, **cultural/idiomatic misrepresentation**, and **sensitive safety-critical content**, and propose concrete technical mitigation strategies."""
        )
    )

    # Part (i): Dataset Exploration & Preprocessing
    cells.append(
        new_markdown_cell(
            """---
## Part (i): Dataset Description, Preprocessing & Splits

We use the **Tatoeba English–French Parallel Corpus** (`fra-eng`, mirrored on HuggingFace `Helsinki-NLP/opus-100`).  
### Preprocessing Pipeline:
1. **Unicode & Typography Normalization**: Standardized via NFC normalization; typographic quotes (`« »`, `“ ”`, `’`) and non-breaking spaces (`\\u202f`, `\\xa0`) before French punctuation (`?`, `!`, `;`, `:`) are normalized.
2. **Deduplication & Quality Filtering**: Filtered to retain clean sentence pairs of $4 \\le L \\le 18$ words with source-to-target length ratios between $0.6$ and $1.8$.
3. **French Elision-Aware Tokenization (for Basic Transformer)**: Splits French contractions (`l'`, `d'`, `j'`, `qu'`, `c'`, `n'`, `s'`) and punctuation into discrete tokens while maintaining `<pad>`, `<sos>`, `<eos>`, and `<unk>` special symbols.
4. **Stratified Dataset Split**:
   - **Total Selected Subset**: $12,000$ parallel sentence pairs
   - **Training Set (80%)**: $9,600$ pairs
   - **Validation Set (10%)**: $1,200$ pairs
   - **Test Set (10%)**: $1,200$ pairs (with a standardized $300$-sentence evaluation test set across all models)"""
        )
    )

    cells.append(
        new_code_cell(
            """import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display

from utils.data_utils import download_and_prepare_dataset, tokenize_basic

bundle = download_and_prepare_dataset(data_dir="data", total_samples=12000, seed=42)
meta = bundle["metadata"]

print("=== Dataset Metadata & Preprocessing Statistics ===")
display(pd.DataFrame([meta]).T.rename(columns={0: "Value"}))

print("\\n=== Sample Preprocessed English-French Parallel Sentence Pairs ===")
sample_df = pd.DataFrame(bundle["train"][:8])
sample_df["en_tokens"] = sample_df["en"].apply(lambda s: " ".join(tokenize_basic(s)))
sample_df["fr_tokens"] = sample_df["fr"].apply(lambda s: " ".join(tokenize_basic(s)))
display(sample_df)"""
        )
    )

    # Model Architectures Overview
    cells.append(
        new_markdown_cell(
            """---
## Architectural Overview of the Four Translation Systems

| Model | Architecture Family | Pretraining Objective | Tokenizer & Vocabulary | Adaptation Strategy |
|---|---|---|---|---|
| **1. Basic Transformer** | Encoder–Decoder | None (Random Xavier Init) | Custom Word + French Elision (`2,645` EN / `3,187` FR) | Trained from Scratch (20 Epochs, Label Smoothing $\\epsilon=0.1$) |
| **2. mT5 (`google/mt5-small`)** | Encoder–Decoder | Multilingual Span Corruption on mC4 (101 languages) | SentencePiece Unigram (`250,112` tokens) | Full Seq2Seq Fine-Tuning (`"translate English to French: "`) |
| **3. LLaMA (`Llama-3.2-1B-Instruct`)** | Decoder-Only (GQA + RoPE + SwiGLU) | Autoregressive Causal Next-Token Prediction + RLHF | Tiktoken BPE (`128,256` tokens) | 3-Shot In-Context Prompting & LoRA ($r=16, \\alpha=32$) |
| **4. Gemma (`gemma-3-1b-it`)** | Decoder-Only (Local/Global Attn + GeGLU) | Autoregressive Causal Next-Token Prediction + Distillation | SentencePiece (`262,144` tokens) | 3-Shot In-Context Prompting & LoRA ($r=16, \\alpha=32$) |"""
        )
    )

    # Inspect Basic Transformer Architecture from code
    cells.append(
        new_code_cell(
            """import torch
from models.basic_transformer import BasicTransformerSeq2Seq
from utils.metrics import count_parameters

basic_net = BasicTransformerSeq2Seq(
    src_vocab_size=len(bundle["src_vocab"]),
    tgt_vocab_size=len(bundle["tgt_vocab"]),
    d_model=256,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=512,
    dropout=0.1,
)
print("=== Basic Transformer Architecture (Implemented from Scratch) ===")
print(basic_net)
print("\\nParameter Breakdown:", count_parameters(basic_net))"""
        )
    )

    # Part (ii): Quantitative Evaluation & Architectural Comparison
    cells.append(
        new_markdown_cell(
            """---
## Part (ii): Quantitative Evaluation & Architectural Comparison

Below we load the complete experimental metrics generated on the NVIDIA GeForce RTX 4060 GPU across all four models (including both **3-Shot Prompting** and **LoRA Fine-Tuned** configurations for LLaMA and Gemma)."""
        )
    )

    cells.append(
        new_code_cell(
            """with open("results/metrics_summary.json", "r", encoding="utf-8") as f:
    summary_json = json.load(f)

df_metrics = pd.read_csv("results/metrics_summary.csv")
display_cols = [
    "model_name",
    "architecture",
    "adaptation_method",
    "bleu",
    "chrf",
    "train_time_sec",
    "inference_time_sec",
    "latency_ms_per_sent",
    "throughput_sents_per_sec",
    "total_params",
    "trainable_params",
    "trainable_pct",
]
print("=== Complete Quantitative Evaluation Table ===")
display(df_metrics[display_cols])"""
        )
    )

    # Display Plots 1, 2, 3, 5
    cells.append(
        new_code_cell(
            """print("=== Figure 1: BLEU and chrF Score Comparison ===")
display(Image(filename="results/plots/bleu_chrf_comparison.png"))

print("\\n=== Figure 2: Computational Efficiency (Training Time, Latency, Parameter Count) ===")
display(Image(filename="results/plots/compute_efficiency.png"))

print("\\n=== Figure 3: Training & Validation Convergence Curves ===")
display(Image(filename="results/plots/learning_curves.png"))

print("\\n=== Figure 4: Quality (chrF) vs. Inference Throughput Trade-Off ===")
display(Image(filename="results/plots/architecture_tradeoff_scatter.png"))"""
        )
    )

    # Side-by-side translations on the test set
    cells.append(
        new_markdown_cell(
            """### Side-by-Side Test Set Translation Comparison
Below are sample translations produced by all four models on the identical held-out English test sentences:"""
        )
    )

    cells.append(
        new_code_cell(
            """with open("results/test_translations_sample.json", "r", encoding="utf-8") as f:
    test_samples = json.load(f)

rows = []
for item in test_samples[:10]:
    rows.append({
        "Source (EN)": item["source_en"],
        "Reference (FR)": item["reference_fr"],
        "Basic Transformer": item["translations"].get("Basic Transformer", ""),
        "mT5-small": item["translations"].get("mT5-small", ""),
        "LLaMA-3.2-1B (LoRA)": item["translations"].get("LLaMA-3.2-1B", ""),
        "Gemma-3-1B (LoRA)": item["translations"].get("Gemma-3-1B", ""),
    })
display(pd.DataFrame(rows))"""
        )
    )

    # Sampling-Temperature Analysis
    cells.append(
        new_markdown_cell(
            """---
## Sampling-Temperature Sensitivity Analysis ($T \\in \\{0.0, 0.3, 0.7, 1.0\\}$)

During autoregressive decoding, temperature $T$ scales the output logits prior to the softmax normalization:
$$P(y_t = w \\mid y_{<t}, x) = \\frac{\\exp(z_w / T)}{\\sum_{v \\in V} \\exp(z_v / T)}$$
- At $T \\to 0.0$ (**Greedy Decoding**), the model deterministically selects the highest-probability token at each step, maximizing n-gram precision (BLEU and chrF).
- At moderate temperature ($T = 0.3$), minor paraphrastic variation emerges while preserving high grammatical fidelity.
- At higher temperatures ($T = 0.7, 1.0$), sampling from the flatter distribution increases lexical diversity (**Distinct-1**, **Distinct-2**) but degrades exact reference matching (BLEU/chrF) and risks semantic drift."""
        )
    )

    cells.append(
        new_code_cell(
            """with open("results/temperature_analysis.json", "r", encoding="utf-8") as f:
    temp_analysis = json.load(f)

temp_rows = []
for model_name, t_dict in temp_analysis.items():
    for temp_val, m_vals in t_dict.items():
        temp_rows.append({
            "Model": model_name,
            "Temperature (T)": float(temp_val),
            "BLEU": m_vals["bleu"],
            "chrF": m_vals["chrf"],
            "Distinct-1": m_vals["distinct_1"],
            "Distinct-2": m_vals["distinct_2"],
            "Length Ratio": m_vals["length_ratio"],
            "Sample Translation": m_vals["sample_translation"],
        })

df_temp = pd.DataFrame(temp_rows)
display(df_temp)

print("\\n=== Figure 5: Effect of Sampling Temperature on BLEU & Lexical Diversity ===")
display(Image(filename="results/plots/temperature_sensitivity.png"))"""
        )
    )

    # Part (iii): Responsible and Ethical AI Analysis
    cells.append(
        new_markdown_cell(
            """---
## Part (iii): Responsible & Ethical AI Evaluation in Machine Translation

We evaluate all four models on a dedicated 10-sentence **Responsible & Ethical AI Diagnostic Suite** spanning four critical dimensions:
1. **Gender & Occupational Stereotype Bias**: English sentences with gender-neutral occupational nouns (`doctor`, `nurse`, `engineer`, `secretary/director`) translated into grammatically gendered French.
2. **Hallucinated Information & Factual/Negation Fidelity**: Medical dosage instructions and temporal schedules where fabricated tokens or dropped negations (`ne ... pas`) can cause real-world harm.
3. **Cultural Misrepresentation, Idioms & Sociolinguistic Register**: English idioms (`raining cats and dogs`) and politeness markers (`Could you please help me, sir?` requiring formal `vous` rather than informal `tu`).
4. **Sensitive & Safety-Critical Content**: Mental health support and pediatric allergy warnings."""
        )
    )

    cells.append(
        new_code_cell(
            """with open("results/ethical_evaluation.json", "r", encoding="utf-8") as f:
    ethical_eval = json.load(f)

eth_rows = []
for item in ethical_eval["diagnostic_items"]:
    eth_rows.append({
        "Category": item["category"],
        "English Source": item["en"],
        "Basic Transformer": item["translations"]["Basic Transformer"]["output"],
        "mT5-small": item["translations"]["mT5-small"]["output"],
        "LLaMA-3.2-1B": item["translations"]["LLaMA-3.2-1B"]["output"],
        "Gemma-3-1B": item["translations"]["Gemma-3-1B"]["output"],
        "Ethical Phenomenon": item["phenomenon"],
    })

display(pd.DataFrame(eth_rows))

print("\\n=== Gender Marker Distribution on Gender-Neutral Occupational Prompts ===")
display(pd.DataFrame(ethical_eval["gender_bias_summary"]).T)

print("\\n=== Figure 6: Grammatical Gender Assignment on Gender-Neutral Occupations ===")
display(Image(filename="results/plots/ethical_gender_bias.png"))"""
        )
    )

    cells.append(
        new_markdown_cell(
            """### Recommended Mitigation Measures for Responsible Machine Translation
1. **Counterfactual Gender Data Augmentation & Dual-Form Output**:
   - Augment parallel training corpora with balanced masculine and feminine occupational pairs (`le médecin / la femme médecin`, `l'ingénieur / l'ingénieure`, `l'infirmier / l'infirmière`).
   - When source gender is ambiguous in single-sentence prompts, configure decoder-only LLMs via system instructions to provide gender-inclusive or dual-form translations (e.g., *« L'ingénieur(e) a résolu le problème »*).
2. **Hallucination Prevention via Constrained Decoding & Faithfulness Verification**:
   - Enforce deterministic or low-temperature decoding ($T \\le 0.3$) for high-stakes medical, legal, and financial translations.
   - Apply lexical/entity constraints and automated negation/number consistency checks between source and target strings.
3. **Culturally Aware & Register-Controlled Prompting / Fine-Tuning**:
   - Incorporate explicit register tags (`[FORMAL_VOUS]`, `[INFORMAL_TU]`) and idiomatic parallel lexicons during fine-tuning so models translate figurative meanings (*« Il pleut des cordes »*) rather than literal calques (*« chats et chiens »*).
4. **Sensitive Content Guardrails & Human-in-the-Loop (HITL) Review**:
   - Integrate toxicity/safety filters and require human expert sign-off for clinical, legal, and crisis-support translations."""
        )
    )

    nb.cells = cells
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Executing notebook {output_path}...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Successfully executed and saved notebook: {output_path}")


if __name__ == "__main__":
    build_and_execute_notebook()

