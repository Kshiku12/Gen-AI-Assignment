"""
Dataset Downloading, Cleaning, Preprocessing, and Vocabulary Construction
for English -> French Neural Machine Translation (ICT4403 Assignment 3).

Dataset Source:
- Tatoeba / OPUS English-French Parallel Corpus
- Official Links:
  1. https://huggingface.co/datasets/Helsinki-NLP/opus-100 (en-fr config)
  2. https://www.manythings.org/anki/fra-eng.zip (Tab-delimited Tatoeba Project extract)
  3. https://tatoeba.org/en/downloads
"""

import os
import re
import json
import random
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

# Special token definitions for Basic Transformer
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def normalize_text(text: str) -> str:
    """
    Clean and normalize a single English or French sentence:
    - Unicode NFC normalization
    - Replace non-breaking spaces and typographic quotes/apostrophes
    - Collapse redundant whitespace
    """
    text = unicodedata.normalize("NFC", text.strip())
    text = text.replace("\u202f", " ").replace("\xa0", " ")
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("«", '"').replace("»", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_basic(text: str) -> List[str]:
    """
    Word and punctuation-aware tokenizer for the Basic Transformer.
    Separates punctuation marks and French contractions (e.g., l', d', j', qu', c', n', s', m', t')
    into clean discrete tokens.
    """
    text = normalize_text(text.lower())
    # Split French elisions like l'homme -> l' homme
    text = re.sub(r"\b([ldjqcnmst])'", r"\1' ", text)
    # Separate common punctuation into tokens
    text = re.sub(r"([.,!?;:\"()])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def detokenize_basic(tokens: List[str]) -> str:
    """Reconstruct natural surface text from basic word/punctuation tokens."""
    filtered = [t for t in tokens if t not in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN)]
    text = " ".join(filtered)
    # Re-attach French elisions
    text = re.sub(r"\b([ldjqcnmst])'\s+", r"\1'", text)
    # Remove space before standard punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    if text:
        text = text[0].upper() + text[1:]
    return text


class Vocabulary:
    """Vocabulary mapping between tokens and integer IDs for Basic Transformer."""

    def __init__(self, lang_name: str, min_freq: int = 2, max_size: int = 8000):
        self.lang_name = lang_name
        self.min_freq = min_freq
        self.max_size = max_size
        self.itos: List[str] = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}
        self.freqs: Counter = Counter()

    def build_vocabulary(self, sentences: List[str]) -> None:
        for sent in sentences:
            tokens = tokenize_basic(sent)
            self.freqs.update(tokens)

        for token, freq in self.freqs.most_common(self.max_size - len(self.itos)):
            if freq < self.min_freq:
                break
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def encode(self, text: str, max_len: int = 32, add_special_tokens: bool = True) -> List[int]:
        tokens = tokenize_basic(text)
        if add_special_tokens:
            tokens = tokens[: max_len - 2]
            ids = [SOS_IDX] + [self.stoi.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]
        else:
            tokens = tokens[:max_len]
            ids = [self.stoi.get(t, UNK_IDX) for t in tokens]
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for idx in ids:
            if idx == EOS_IDX:
                break
            if idx in (PAD_IDX, SOS_IDX):
                continue
            tokens.append(self.itos[idx] if 0 <= idx < len(self.itos) else UNK_TOKEN)
        return detokenize_basic(tokens)

    def __len__(self) -> int:
        return len(self.itos)

    def to_dict(self) -> Dict:
        return {
            "lang_name": self.lang_name,
            "min_freq": self.min_freq,
            "max_size": self.max_size,
            "itos": self.itos,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Vocabulary":
        vocab = cls(data["lang_name"], data["min_freq"], data["max_size"])
        vocab.itos = data["itos"]
        vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
        return vocab


def download_and_prepare_dataset(
    data_dir: str = "data",
    total_samples: int = 12000,
    min_words: int = 4,
    max_words: int = 18,
    seed: int = 42,
) -> Dict:
    """
    Download and preprocess the English-French parallel dataset.
    Uses Tatoeba / OPUS-100 English-French parallel corpus.
    Saves train (80%), validation (10%), and test (10%) splits along with vocabulary and metadata.
    """
    raw_dir = os.path.join(data_dir, "raw")
    proc_dir = os.path.join(data_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    train_path = os.path.join(proc_dir, "train.json")
    val_path = os.path.join(proc_dir, "val.json")
    test_path = os.path.join(proc_dir, "test.json")
    meta_path = os.path.join(proc_dir, "dataset_metadata.json")
    vocab_path = os.path.join(proc_dir, "vocab.json")

    if all(os.path.exists(p) for p in [train_path, val_path, test_path, meta_path, vocab_path]):
        print("Loaded cached preprocessed dataset from", proc_dir)
        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(val_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        src_vocab = Vocabulary.from_dict(vocab_dict["en"])
        tgt_vocab = Vocabulary.from_dict(vocab_dict["fr"])
        return {
            "train": train_data,
            "val": val_data,
            "test": test_data,
            "metadata": metadata,
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab,
        }

    random.seed(seed)
    raw_pairs: List[Tuple[str, str]] = []
    source_name = ""
    source_url = ""

    # Strategy 1: Try downloading ManyThings / Tatoeba fra-eng.zip or HuggingFace datasets
    anki_zip_path = os.path.join(raw_dir, "fra-eng.zip")
    anki_txt_path = os.path.join(raw_dir, "fra.txt")

    if not os.path.exists(anki_txt_path):
        try:
            print("Attempting to download Tatoeba English-French parallel corpus (ManyThings / Tatoeba)...")
            req = urllib.request.Request(
                "https://www.manythings.org/anki/fra-eng.zip",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, timeout=25) as resp, open(anki_zip_path, "wb") as out_f:
                out_f.write(resp.read())
            with zipfile.ZipFile(anki_zip_path, "r") as zf:
                zf.extract("fra.txt", raw_dir)
            source_name = "Tatoeba English-French Parallel Corpus (ManyThings.org / Tatoeba Project)"
            source_url = "https://www.manythings.org/anki/fra-eng.zip (Mirror: https://huggingface.co/datasets/Helsinki-NLP/opus-100)"
        except Exception as e:
            print(f"Direct zip download note ({e}), using HuggingFace datasets...")

    if os.path.exists(anki_txt_path):
        with open(anki_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    raw_pairs.append((parts[0], parts[1]))
        if not source_name:
            source_name = "Tatoeba English-French Parallel Corpus (ManyThings.org / Tatoeba Project)"
            source_url = "https://www.manythings.org/anki/fra-eng.zip"
    else:
        # Fallback to HuggingFace Helsinki-NLP/opus_books or opus-100
        from datasets import load_dataset
        print("Loading English-French parallel corpus from HuggingFace (Helsinki-NLP/opus-100, en-fr)...")
        ds = load_dataset("Helsinki-NLP/opus-100", "en-fr", split="train[:80000]")
        for item in ds:
            tr = item["translation"]
            raw_pairs.append((tr["en"], tr["fr"]))
        source_name = "OPUS-100 / Tatoeba English-French Parallel Corpus (Helsinki-NLP)"
        source_url = "https://huggingface.co/datasets/Helsinki-NLP/opus-100"

    print(f"Total raw sentence pairs retrieved: {len(raw_pairs):,}")

    # Clean, filter, and deduplicate
    cleaned_pairs = []
    seen_en = set()
    alpha_re = re.compile(r"^[a-zA-ZÀ-ÿ0-9\s.,!?;:'\"()-]+$")

    for en_raw, fr_raw in raw_pairs:
        en = normalize_text(en_raw)
        fr = normalize_text(fr_raw)
        if not en or not fr:
            continue
        en_key = en.lower()
        if en_key in seen_en:
            continue
        if not alpha_re.match(en) or not alpha_re.match(fr):
            continue
        en_words = en.split()
        fr_words = fr.split()
        if not (min_words <= len(en_words) <= max_words and min_words <= len(fr_words) <= max_words):
            continue
        # Length ratio filter to remove noisy alignments
        ratio = len(fr_words) / max(len(en_words), 1)
        if not (0.6 <= ratio <= 1.8):
            continue
        seen_en.add(en_key)
        cleaned_pairs.append({"en": en, "fr": fr})

    print(f"Cleaned & deduplicated pairs passing quality filters: {len(cleaned_pairs):,}")

    # Select a well-distributed subset across length buckets
    if len(cleaned_pairs) > total_samples:
        # Sort by word count to stratify across short, medium, and longer conversational sentences
        # Take from middle-to-rich part so sentences are neither 2-word stubs nor overly obscure
        random.shuffle(cleaned_pairs)
        selected_pairs = cleaned_pairs[:total_samples]
    else:
        selected_pairs = cleaned_pairs

    random.shuffle(selected_pairs)

    n_total = len(selected_pairs)
    n_train = int(0.80 * n_total)
    n_val = int(0.10 * n_total)
    n_test = n_total - n_train - n_val

    train_data = selected_pairs[:n_train]
    val_data = selected_pairs[n_train : n_train + n_val]
    test_data = selected_pairs[n_train + n_val :]

    # Build source (English) and target (French) vocabularies on the training split ONLY
    src_vocab = Vocabulary("en", min_freq=2, max_size=8000)
    tgt_vocab = Vocabulary("fr", min_freq=2, max_size=8000)
    src_vocab.build_vocabulary([item["en"] for item in train_data])
    tgt_vocab.build_vocabulary([item["fr"] for item in train_data])

    # Compute dataset statistics
    en_lengths = [len(tokenize_basic(x["en"])) for x in selected_pairs]
    fr_lengths = [len(tokenize_basic(x["fr"])) for x in selected_pairs]

    # Compute vocabulary token coverage on validation set
    val_en_tokens = [t for x in val_data for t in tokenize_basic(x["en"])]
    val_fr_tokens = [t for x in val_data for t in tokenize_basic(x["fr"])]
    en_cov = 100.0 * sum(1 for t in val_en_tokens if t in src_vocab.stoi) / max(len(val_en_tokens), 1)
    fr_cov = 100.0 * sum(1 for t in val_fr_tokens if t in tgt_vocab.stoi) / max(len(val_fr_tokens), 1)

    metadata = {
        "dataset_name": source_name,
        "source_url": source_url,
        "backup_url": "https://huggingface.co/datasets/Helsinki-NLP/opus-100",
        "source_language": "English (en)",
        "target_language": "French (fr)",
        "raw_pairs_count": len(raw_pairs),
        "cleaned_pairs_count": len(cleaned_pairs),
        "total_selected": n_total,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "en_vocab_size": len(src_vocab),
        "fr_vocab_size": len(tgt_vocab),
        "en_token_coverage_pct": round(en_cov, 2),
        "fr_token_coverage_pct": round(fr_cov, 2),
        "avg_en_tokens": round(sum(en_lengths) / len(en_lengths), 2),
        "avg_fr_tokens": round(sum(fr_lengths) / len(fr_lengths), 2),
        "max_en_tokens": max(en_lengths),
        "max_fr_tokens": max(fr_lengths),
    }

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"en": src_vocab.to_dict(), "fr": tgt_vocab.to_dict()}, f, ensure_ascii=False, indent=2)

    print("Saved preprocessed dataset splits and vocabularies to", proc_dir)
    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "metadata": metadata,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
    }


class ParallelTranslationDataset(Dataset):
    """PyTorch Dataset for the Basic Transformer (Encoder-Decoder)."""

    def __init__(
        self,
        pairs: List[Dict[str, str]],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_len: int = 32,
    ):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.pairs[idx]
        src_ids = self.src_vocab.encode(item["en"], max_len=self.max_len)
        tgt_ids = self.tgt_vocab.encode(item["fr"], max_len=self.max_len)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_parallel_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length source and target sequences in a mini-batch."""
    src_list, tgt_list = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded

