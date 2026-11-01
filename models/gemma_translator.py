"""
Model 4: Gemma (Decoder-Only Causal Language Model Architecture)
Checkpoint: unsloth/gemma-3-1b-it (Google Gemma 3 1B Instruction-Tuned, 1.00B parameters).

Architecture Highlights:
- Interleaved Local Sliding-Window and Global Causal Self-Attention
- GeGLU Feed-Forward Network, Pre- and Post-RMSNorm, QK-Norm
- 262,144-token multilingual SentencePiece vocabulary
- Adaptation Methods Implemented:
  1. Few-Shot In-Context Prompting (3-shot structured translation prompt)
  2. Parameter-Efficient LoRA Fine-Tuning (r=16, alpha=32 on attention projections)
"""

import time
from typing import List, Dict, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.llama_translator import (
    FEW_SHOT_PROMPT_TEMPLATE,
    clean_decoder_translation,
    finetune_llama_lora,
    translate_llama,
)


def load_gemma_model(
    model_name: str = "unsloth/gemma-3-1b-it",
    device: Optional[torch.device] = None,
):
    """Load Google Gemma decoder-only model and configure left-padded tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if (device is not None and device.type == "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if device is not None:
        model.to(device)
    return model, tokenizer


def finetune_gemma_lora(
    model,
    tokenizer,
    train_pairs: List[Dict[str, str]],
    val_pairs: List[Dict[str, str]],
    device: torch.device,
    epochs: int = 1,
    max_train_samples: int = 2400,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    lr: float = 2e-4,
) -> Tuple[torch.nn.Module, Dict]:
    """Apply Low-Rank Adaptation (LoRA) to Gemma-3-1B-IT on the parallel training corpus."""
    return finetune_llama_lora(
        model=model,
        tokenizer=tokenizer,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        device=device,
        epochs=epochs,
        max_train_samples=max_train_samples,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
    )


@torch.no_grad()
def translate_gemma(
    model,
    tokenizer,
    sentences: List[str],
    device: torch.device,
    batch_size: int = 16,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    mode: str = "lora",
) -> Tuple[List[str], float]:
    """Generate French translations using Gemma-3-1B-IT (supports prompting and LoRA modes)."""
    return translate_llama(
        model=model,
        tokenizer=tokenizer,
        sentences=sentences,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        mode=mode,
    )

