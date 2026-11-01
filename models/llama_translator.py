"""
Model 3: LLaMA (Decoder-Only Causal Language Model Architecture)
Checkpoint: unsloth/Llama-3.2-1B-Instruct (Meta Llama 3.2 1B Instruct, 1.24B parameters).

Architecture Highlights:
- Autoregressive Decoder-Only Transformer with Grouped-Query Attention (GQA)
- Rotary Positional Embeddings (RoPE, theta=500,000), SwiGLU FFN, RMSNorm
- 128,256-token tiktoken BPE vocabulary
- Adaptation Methods Implemented:
  1. Few-Shot In-Context Prompting (3-shot structured translation prompt)
  2. Parameter-Efficient LoRA Fine-Tuning (r=16, alpha=32 on attention projections)
"""

import time
import re
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from utils.metrics import count_parameters


FEW_SHOT_PROMPT_TEMPLATE = """Translate the following English sentence into natural, accurate French. Output only the French translation on a single line.

English: That's exactly what he wanted.
French: C'est exactement ce qu'il voulait.

English: Could you please help me find the station?
French: Pourriez-vous m'aider à trouver la gare, s'il vous plaît ?

English: She bought a new book yesterday morning.
French: Elle a acheté un nouveau livre hier matin.

English: {en_sentence}
French:"""


def clean_decoder_translation(raw_output: str) -> str:
    """Extract only the first line of the French translation from a causal LM generation."""
    text = raw_output.strip()
    # Take first non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    first = lines[0]
    # Stop if model starts another 'English:' turn
    for marker in ["English:", "French:", "Note:", "Translation:", "<|"]:
        if marker in first:
            first = first.split(marker)[0].strip()
    first = first.strip(' "\'')
    return first


def load_llama_model(
    model_name: str = "unsloth/Llama-3.2-1B-Instruct",
    device: Optional[torch.device] = None,
):
    """Load LLaMA decoder-only model and configure left-padded tokenizer for batched generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if (device is not None and device.type == "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if device is not None:
        model.to(device)
    return model, tokenizer


class CausalTranslationDataset(Dataset):
    """Causal LM instruction-tuning dataset masking prompt tokens in labels."""

    def __init__(self, pairs: List[Dict[str, str]], tokenizer, max_len: int = 80):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        item = self.pairs[idx]
        prompt = f"Translate English to French:\nEnglish: {item['en']}\nFrench:"
        full_text = f"{prompt} {item['fr']}{self.tokenizer.eos_token}"

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True, max_length=self.max_len, truncation=True)

        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100  # Compute loss only on French target tokens

        return {"input_ids": full_ids, "labels": labels}


def finetune_llama_lora(
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
    """
    Apply Low-Rank Adaptation (LoRA) to LLaMA-3.2-1B-Instruct on the parallel training corpus.
    """
    base_params = count_parameters(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.to(device)
    lora_params = count_parameters(peft_model)

    subset_train = train_pairs[:max_train_samples]
    subset_val = val_pairs[:300]

    train_ds = CausalTranslationDataset(subset_train, tokenizer)
    val_ds = CausalTranslationDataset(subset_val, tokenizer)
    pad_id = tokenizer.pad_token_id

    def collate_causal(batch):
        max_l = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad_len = max_l - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad_len)
            attention_mask.append([1] * len(x["input_ids"]) + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_causal)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_causal)

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(train_loader) // grad_accum_steps)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1
    )

    history = {"train_loss": [], "val_loss": [], "epoch_times": []}
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        peft_model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step_idx, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                outputs = peft_model(**batch)
                raw_loss = outputs.loss
            loss = raw_loss / grad_accum_steps
            loss.backward()

            total_loss += raw_loss.item()
            n_batches += 1

            if step_idx % grad_accum_steps == 0 or step_idx == len(train_loader):
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler.last_epoch < total_steps - 1:
                    scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_loss / max(n_batches, 1)

        peft_model.eval()
        total_val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    outputs = peft_model(**batch)
                    loss = outputs.loss
                total_val_loss += loss.item()
                n_val += 1

        avg_val_loss = total_val_loss / max(n_val, 1)
        ep_time = time.perf_counter() - ep_start
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["epoch_times"].append(round(ep_time, 2))
        print(
            f"  [LLaMA-3.2-1B LoRA] Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {ep_time:.2f}s"
        )

    total_time = time.perf_counter() - start_time
    return peft_model, {
        "history": history,
        "best_val_loss": round(min(history["val_loss"]) if history["val_loss"] else 0.0, 4),
        "total_train_time_sec": round(total_time, 2),
        "base_params": base_params,
        "params": lora_params,
    }


@torch.no_grad()
def translate_llama(
    model,
    tokenizer,
    sentences: List[str],
    device: torch.device,
    batch_size: int = 16,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    mode: str = "lora",
) -> Tuple[List[str], float]:
    """
    Generate French translations using LLaMA-3.2-1B-Instruct.
    - mode='prompting': Uses 3-shot in-context prompting format.
    - mode='lora': Uses concise fine-tuned instruction format.
    """
    model.eval()
    model.to(device)
    tokenizer.padding_side = "left"
    translations: List[str] = []
    start_time = time.perf_counter()

    for i in range(0, len(sentences), batch_size):
        batch_sents = sentences[i : i + batch_size]
        if mode == "prompting":
            prompts = [FEW_SHOT_PROMPT_TEMPLATE.format(en_sentence=s) for s in batch_sents]
        else:
            prompts = [f"Translate English to French:\nEnglish: {s}\nFrench:" for s in batch_sents]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        input_len = inputs["input_ids"].shape[1]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            if temperature <= 1e-5:
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=30,
                    top_p=0.92,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        new_tokens = gen_ids[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for d in decoded:
            translations.append(clean_decoder_translation(d))

    elapsed = time.perf_counter() - start_time
    return translations, round(elapsed, 4)

