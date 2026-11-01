"""
Model 2: mT5 (Multilingual Text-to-Text Transfer Transformer)
Encoder-Decoder Architecture (Xue et al., 2021) - Checkpoint: google/mt5-small (300M parameters).

Features:
- SentencePiece unigram subword tokenizer (250,112 multilingual tokens)
- Supervised Seq2Seq fine-tuning on English -> French parallel corpus with task prefix
  "translate English to French: "
- Freezes the 250k multilingual token embedding tables (shared & lm_head) and fine-tunes all
  Encoder and Decoder Transformer blocks (44.1M trainable parameters out of 300.2M total)
  for memory-efficient, fast convergence on 8GB VRAM.
- Supports deterministic greedy decoding (T=0.0) and temperature sampling (T>0.0)
"""

import time
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils.metrics import count_parameters


class MT5ParallelDataset(Dataset):
    """Dataset formatting English-French pairs for mT5 Seq2Seq fine-tuning."""

    def __init__(
        self,
        pairs: List[Dict[str, str]],
        tokenizer,
        max_len: int = 40,
        prefix: str = "translate English to French: ",
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prefix = prefix

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = self.pairs[idx]
        return {
            "src_text": self.prefix + item["en"],
            "tgt_text": item["fr"],
        }


def load_mt5_model(model_name: str = "google/mt5-small", device: Optional[torch.device] = None):
    """Load google/mt5-small tokenizer and encoder-decoder model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device is not None:
        model.to(device)
    return model, tokenizer


def finetune_mt5(
    model,
    tokenizer,
    train_pairs: List[Dict[str, str]],
    val_pairs: List[Dict[str, str]],
    device: torch.device,
    epochs: int = 4,
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    lr: float = 1.5e-3,
    max_len: int = 36,
) -> Dict:
    """
    Fine-tune google/mt5-small on the English-French training set.
    Freezes the 250,112-vocab embedding & LM projection matrices while training all
    Encoder and Decoder attention/FFN layers in mixed bfloat16 precision with gradient accumulation.
    """
    # Freeze 250k multilingual token embedding tables to prevent 8GB VRAM overflow
    # and preserve cross-lingual subword alignment while adapting all Encoder-Decoder blocks
    model.shared.requires_grad_(False)
    if hasattr(model, "lm_head"):
        model.lm_head.requires_grad_(False)

    train_ds = MT5ParallelDataset(train_pairs, tokenizer, max_len=max_len)
    val_ds = MT5ParallelDataset(val_pairs, tokenizer, max_len=max_len)

    def collate_fn(batch: List[Dict[str, str]]):
        src_texts = [x["src_text"] for x in batch]
        tgt_texts = [x["tgt_text"] for x in batch]
        model_inputs = tokenizer(
            src_texts,
            max_length=max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenizer(
            text_target=tgt_texts,
            max_length=max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(train_loader) // grad_accum_steps)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.12
    )

    history = {"train_loss": [], "val_loss": [], "epoch_times": []}
    start_total = time.perf_counter()

    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        model.train()
        total_train_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step_idx, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                outputs = model(**batch)
                raw_loss = outputs.loss

            if torch.isnan(raw_loss):
                continue

            loss = raw_loss / grad_accum_steps
            loss.backward()

            total_train_loss += raw_loss.item()
            n_batches += 1

            if step_idx % grad_accum_steps == 0 or step_idx == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                if scheduler.last_epoch < total_steps - 1:
                    scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_train_loss / max(n_batches, 1)

        model.eval()
        total_val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    outputs = model(**batch)
                    loss = outputs.loss
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                    n_val += 1

        avg_val_loss = total_val_loss / max(n_val, 1)
        ep_time = time.perf_counter() - ep_start

        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["epoch_times"].append(round(ep_time, 2))

        print(
            f"  [mT5-small] Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {ep_time:.2f}s"
        )

    total_time = time.perf_counter() - start_total
    return {
        "history": history,
        "best_val_loss": round(min(history["val_loss"]) if history["val_loss"] else 0.0, 4),
        "total_train_time_sec": round(total_time, 2),
        "params": count_parameters(model),
    }


@torch.no_grad()
def translate_mt5(
    model,
    tokenizer,
    sentences: List[str],
    device: torch.device,
    batch_size: int = 32,
    max_new_tokens: int = 36,
    temperature: float = 0.0,
    prefix: str = "translate English to French: ",
) -> Tuple[List[str], float]:
    """
    Generate French translations using the fine-tuned mT5 model.
    Supports Greedy Decoding (temperature=0.0) and Temperature Sampling (temperature>0.0).
    """
    model.eval()
    model.to(device)
    translations: List[str] = []
    start_time = time.perf_counter()

    for i in range(0, len(sentences), batch_size):
        batch_sents = [prefix + s for s in sentences[i : i + batch_size]]
        inputs = tokenizer(
            batch_sents,
            max_length=48,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            if temperature <= 1e-5:
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.15,
                )
            else:
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=30,
                    top_p=0.92,
                    repetition_penalty=1.15,
                )

        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        translations.extend([d.strip() for d in decoded])

    elapsed = time.perf_counter() - start_time
    return translations, round(elapsed, 4)
