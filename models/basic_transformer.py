"""
Model 1: Basic Transformer (Encoder-Decoder Architecture from Scratch)
Implemented in PyTorch following Vaswani et al. (2017) "Attention Is All You Need".
Includes:
- Scaled Token Embeddings + Sinusoidal Positional Encoding
- Multi-Head Self-Attention, Masked Decoder Self-Attention, and Encoder-Decoder Cross-Attention
- Position-wise Feed-Forward Networks with Residual Connections & Layer Normalization
- Training loop with Label Smoothing Cross-Entropy and Cosine Warmup Schedule
- Autoregressive inference supporting Greedy Decoding (T=0.0) and Temperature Sampling (T>0.0)
"""

import math
import time
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_utils import (
    Vocabulary,
    ParallelTranslationDataset,
    collate_parallel_batch,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
)
from utils.metrics import count_parameters


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class BasicTransformerSeq2Seq(nn.Module):
    """
    Encoder-Decoder Transformer built from scratch for Neural Machine Translation.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def generate_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((sz, sz), device=device, dtype=torch.bool), diagonal=1)

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc_out(out)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        device = src.device
        src_pad_mask = src == PAD_IDX
        tgt_pad_mask = tgt_in == PAD_IDX
        tgt_causal_mask = self.generate_causal_mask(tgt_in.size(1), device)

        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt_in) * math.sqrt(self.d_model))

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_causal_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.fc_out(output)


def train_basic_transformer(
    model: BasicTransformerSeq2Seq,
    train_pairs: List[Dict[str, str]],
    val_pairs: List[Dict[str, str]],
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    device: torch.device,
    epochs: int = 18,
    batch_size: int = 128,
    lr: float = 1e-3,
    max_len: int = 32,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Train the Basic Transformer from scratch using Label Smoothing Cross-Entropy
    and AdamW with OneCycle learning rate schedule.
    """
    train_ds = ParallelTranslationDataset(train_pairs, src_vocab, tgt_vocab, max_len=max_len)
    val_ds = ParallelTranslationDataset(val_pairs, src_vocab, tgt_vocab, max_len=max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_parallel_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_parallel_batch
    )

    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
    )

    history = {"train_loss": [], "val_loss": [], "val_ppl": [], "epoch_times": []}
    best_val_loss = float("inf")
    best_state = None
    start_total = time.perf_counter()

    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            optimizer.zero_grad()
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = total_train_loss / max(n_train_batches, 1)

        # Validation pass
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in = tgt[:, :-1]
                tgt_out = tgt[:, 1:]
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                total_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = total_val_loss / max(n_val_batches, 1)
        val_ppl = math.exp(min(avg_val_loss, 20.0))
        ep_time = time.perf_counter() - ep_start

        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_ppl"].append(round(val_ppl, 2))
        history["epoch_times"].append(round(ep_time, 2))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_path:
                torch.save(best_state, save_path)

        if epoch == 1 or epoch % 3 == 0 or epoch == epochs:
            print(
                f"  [Basic Transformer] Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Val PPL: {val_ppl:.2f} | Time: {ep_time:.2f}s"
            )

    total_train_time = time.perf_counter() - start_total
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return {
        "history": history,
        "best_val_loss": round(best_val_loss, 4),
        "total_train_time_sec": round(total_train_time, 2),
        "params": count_parameters(model),
    }


@torch.no_grad()
def translate_basic_transformer(
    model: BasicTransformerSeq2Seq,
    sentences: List[str],
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    device: torch.device,
    max_len: int = 30,
    batch_size: int = 64,
    temperature: float = 0.0,
) -> Tuple[List[str], float]:
    """
    Batched autoregressive translation for the Basic Transformer.
    Supports Greedy Decoding (temperature == 0.0) and Temperature Sampling (temperature > 0.0).
    Returns (list_of_translations, total_inference_time_sec).
    """
    model.eval()
    model.to(device)
    translations: List[str] = []
    start_time = time.perf_counter()

    for i in range(0, len(sentences), batch_size):
        batch_sents = sentences[i : i + batch_size]
        encoded_list = [
            torch.tensor(src_vocab.encode(s, max_len=max_len), dtype=torch.long) for s in batch_sents
        ]
        src_tensor = torch.nn.utils.rnn.pad_sequence(
            encoded_list, batch_first=True, padding_value=PAD_IDX
        ).to(device)
        src_pad_mask = src_tensor == PAD_IDX

        memory = model.encode(src_tensor, src_key_padding_mask=src_pad_mask)
        bsz = src_tensor.size(0)
        ys = torch.full((bsz, 1), SOS_IDX, dtype=torch.long, device=device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            tgt_mask = model.generate_causal_mask(ys.size(1), device)
            tgt_pad_mask = ys == PAD_IDX
            logits = model.decode(
                ys,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask,
            )
            next_logits = logits[:, -1, :]  # (B, V)
            # Suppress <pad> and <sos> during generation
            next_logits[:, PAD_IDX] = -1e9
            next_logits[:, SOS_IDX] = -1e9

            if temperature <= 1e-5:
                next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = next_logits / temperature
                # Top-k filtering (k=25) to prevent tail-noise while preserving temperature effect
                top_k = min(25, scaled_logits.size(-1))
                vals, _ = torch.topk(scaled_logits, top_k, dim=-1)
                min_vals = vals[:, -1].unsqueeze(-1)
                filtered_logits = torch.where(
                    scaled_logits < min_vals,
                    torch.full_like(scaled_logits, -1e9),
                    scaled_logits,
                )
                probs = F.softmax(filtered_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)

            ys = torch.cat([ys, next_tokens], dim=1)
            finished |= next_tokens.squeeze(1) == EOS_IDX
            if finished.all():
                break

        for row in ys.cpu().tolist():
            translations.append(tgt_vocab.decode(row))

    elapsed = time.perf_counter() - start_time
    return translations, round(elapsed, 4)
