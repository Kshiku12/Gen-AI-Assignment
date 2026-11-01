"""
Evaluation Metrics for Machine Translation Systems (ICT4403 Assignment 3).
Computes Corpus BLEU, chrF, Lexical Diversity (Distinct-1/2), Parameter Counts,
and Latency Statistics.
"""

from typing import List, Dict
import sacrebleu
import torch
import torch.nn as nn


def compute_translation_metrics(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute standardized Corpus BLEU and chrF scores using SacreBLEU,
    along with length ratio and lexical diversity metrics.
    """
    clean_hyps = [h.strip() if h and h.strip() else "." for h in hypotheses]
    clean_refs = [r.strip() for r in references]

    bleu_res = sacrebleu.corpus_bleu(clean_hyps, [clean_refs], force=True)
    chrf_res = sacrebleu.corpus_chrf(clean_hyps, [clean_refs])

    # Compute Distinct-1 and Distinct-2 (for temperature analysis)
    all_unigrams = []
    all_bigrams = []
    total_hyp_words = 0
    total_ref_words = 0

    for h, r in zip(clean_hyps, clean_refs):
        h_words = h.lower().split()
        r_words = r.lower().split()
        total_hyp_words += len(h_words)
        total_ref_words += len(r_words)
        all_unigrams.extend(h_words)
        if len(h_words) >= 2:
            all_bigrams.extend(zip(h_words[:-1], h_words[1:]))

    distinct_1 = len(set(all_unigrams)) / max(len(all_unigrams), 1)
    distinct_2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    len_ratio = total_hyp_words / max(total_ref_words, 1)

    return {
        "bleu": round(float(bleu_res.score), 2),
        "bleu_bp": round(float(bleu_res.bp), 4),
        "bleu_1": round(float(bleu_res.precisions[0]), 2),
        "bleu_2": round(float(bleu_res.precisions[1]), 2),
        "bleu_3": round(float(bleu_res.precisions[2]), 2),
        "bleu_4": round(float(bleu_res.precisions[3]), 2),
        "chrf": round(float(chrf_res.score), 2),
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "length_ratio": round(len_ratio, 4),
    }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total and trainable parameter counts for a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_pct": round(100.0 * trainable_params / max(total_params, 1), 4),
    }

