"""
Master Training, Fine-Tuning, Prompting, Temperature Analysis, and Ethical Evaluation Pipeline
for Foundations of Generative AI (ICT4403) - Assignment 3.

Evaluates 4 Neural Machine Translation Architectures on English -> French (Tatoeba Parallel Corpus):
1. Basic Transformer (Encoder-Decoder, trained from scratch in PyTorch)
2. mT5 (google/mt5-small, Encoder-Decoder, Seq2Seq fine-tuned)
3. LLaMA (unsloth/Llama-3.2-1B-Instruct, Decoder-Only, 3-Shot Prompting & LoRA Fine-Tuned)
4. Gemma (unsloth/gemma-3-1b-it, Decoder-Only, 3-Shot Prompting & LoRA Fine-Tuned)
"""

import os
import gc
import json
import time
from typing import Dict, List

import torch
import pandas as pd

from utils.data_utils import download_and_prepare_dataset
from utils.metrics import compute_translation_metrics, count_parameters
from utils.ethics_analyzer import ETHICAL_DIAGNOSTIC_SUITE, evaluate_ethical_suite

from models.basic_transformer import (
    BasicTransformerSeq2Seq,
    train_basic_transformer,
    translate_basic_transformer,
)
from models.mt5_translator import (
    load_mt5_model,
    finetune_mt5,
    translate_mt5,
)
from models.llama_translator import (
    load_llama_model,
    finetune_llama_lora,
    translate_llama,
)
from models.gemma_translator import (
    load_gemma_model,
    finetune_gemma_lora,
    translate_gemma,
)


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Assignment 3 Pipeline on Device: {device} ===")
    if device.type == "cuda":
        print(f"=== GPU Hardware: {torch.cuda.get_device_name(0)} ===")

    # 1. Load preprocessed English-French dataset
    bundle = download_and_prepare_dataset(data_dir="data", total_samples=12000, seed=42)
    train_pairs = bundle["train"]
    val_pairs = bundle["val"]
    test_pairs = bundle["test"]
    src_vocab = bundle["src_vocab"]
    tgt_vocab = bundle["tgt_vocab"]

    # Standardized evaluation test set (300 parallel sentences) and temperature subset (100 sentences)
    eval_test_pairs = test_pairs[:300]
    eval_src = [x["en"] for x in eval_test_pairs]
    eval_ref = [x["fr"] for x in eval_test_pairs]

    temp_src = eval_src[:100]
    temp_ref = eval_ref[:100]
    temperatures = [0.0, 0.3, 0.7, 1.0]

    ethical_src = [x["en"] for x in ETHICAL_DIAGNOSTIC_SUITE]

    # Cache paths so completed models are preserved
    state_path = "results/pipeline_cache.json"
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {
            "models": {},
            "temperature_analysis": {},
            "ethical_translations": {},
            "test_predictions": {},
            "training_histories": {},
        }

    def save_cache():
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    # =========================================================================
    # MODEL 1: Basic Transformer (Encoder-Decoder from Scratch)
    # =========================================================================
    if "Basic Transformer" not in cache["models"]:
        print("\n[1/4] Training Basic Transformer (Encoder-Decoder from scratch)...")
        basic_model = BasicTransformerSeq2Seq(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=256,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            dropout=0.1,
        )
        train_info = train_basic_transformer(
            model=basic_model,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            epochs=20,
            batch_size=128,
            lr=1.2e-3,
            save_path="checkpoints/basic_transformer.pt",
        )

        preds, inf_time = translate_basic_transformer(
            basic_model, eval_src, src_vocab, tgt_vocab, device, batch_size=64, temperature=0.0
        )
        metrics = compute_translation_metrics(preds, eval_ref)
        print(f"  -> Basic Transformer Test BLEU: {metrics['bleu']:.2f} | chrF: {metrics['chrf']:.2f} | Inference: {inf_time:.2f}s")

        # Temperature sweep
        temp_res = {}
        for t in temperatures:
            t_preds, t_time = translate_basic_transformer(
                basic_model, temp_src, src_vocab, tgt_vocab, device, batch_size=64, temperature=t
            )
            t_m = compute_translation_metrics(t_preds, temp_ref)
            t_m["inference_time_sec"] = t_time
            t_m["sample_translation"] = t_preds[0] if t_preds else ""
            temp_res[str(t)] = t_m

        # Ethical suite
        eth_preds, _ = translate_basic_transformer(
            basic_model, ethical_src, src_vocab, tgt_vocab, device, batch_size=16, temperature=0.0
        )

        cache["models"]["Basic Transformer"] = {
            "model_name": "Basic Transformer",
            "variant": "Custom 6-Layer Transformer (d=256, H=8)",
            "architecture": "Encoder-Decoder",
            "adaptation_method": "Trained from Scratch",
            "tokenizer_type": f"Word + Elision Tokenizer (EN={len(src_vocab)}, FR={len(tgt_vocab)})",
            "total_params": train_info["params"]["total_params"],
            "trainable_params": train_info["params"]["trainable_params"],
            "trainable_pct": train_info["params"]["trainable_pct"],
            "train_time_sec": train_info["total_train_time_sec"],
            "inference_time_sec": inf_time,
            "latency_ms_per_sent": round(1000.0 * inf_time / len(eval_src), 2),
            "throughput_sents_per_sec": round(len(eval_src) / max(inf_time, 1e-6), 2),
            **metrics,
        }
        cache["temperature_analysis"]["Basic Transformer"] = temp_res
        cache["ethical_translations"]["Basic Transformer"] = eth_preds
        cache["test_predictions"]["Basic Transformer"] = preds
        cache["training_histories"]["Basic Transformer"] = train_info["history"]
        save_cache()
        del basic_model
        cleanup_gpu()
    else:
        print("\n[1/4] Basic Transformer loaded from cache.")

    # =========================================================================
    # MODEL 2: mT5 (google/mt5-small, Encoder-Decoder Fine-Tuned)
    # =========================================================================
    if "mT5-small" not in cache["models"]:
        print("\n[2/4] Fine-tuning mT5 (google/mt5-small, Encoder-Decoder)...")
        mt5_model, mt5_tokenizer = load_mt5_model("google/mt5-small", device=device)

        # Fine-tune on 6,400 training pairs for 5 epochs with batch_size=8, grad_accum=4
        mt5_train_info = finetune_mt5(
            model=mt5_model,
            tokenizer=mt5_tokenizer,
            train_pairs=train_pairs[:6400],
            val_pairs=val_pairs[:400],
            device=device,
            epochs=5,
            batch_size=8,
            grad_accum_steps=4,
            lr=1.5e-3,
        )

        preds, inf_time = translate_mt5(
            mt5_model, mt5_tokenizer, eval_src, device, batch_size=32, temperature=0.0
        )
        metrics = compute_translation_metrics(preds, eval_ref)
        print(f"  -> mT5-small Test BLEU: {metrics['bleu']:.2f} | chrF: {metrics['chrf']:.2f} | Inference: {inf_time:.2f}s")

        # Temperature sweep
        temp_res = {}
        for t in temperatures:
            t_preds, t_time = translate_mt5(
                mt5_model, mt5_tokenizer, temp_src, device, batch_size=32, temperature=t
            )
            t_m = compute_translation_metrics(t_preds, temp_ref)
            t_m["inference_time_sec"] = t_time
            t_m["sample_translation"] = t_preds[0] if t_preds else ""
            temp_res[str(t)] = t_m

        # Ethical suite
        eth_preds, _ = translate_mt5(
            mt5_model, mt5_tokenizer, ethical_src, device, batch_size=16, temperature=0.0
        )

        cache["models"]["mT5-small"] = {
            "model_name": "mT5-small",
            "variant": "google/mt5-small (300M)",
            "architecture": "Encoder-Decoder",
            "adaptation_method": "Full Seq2Seq Fine-Tuning",
            "tokenizer_type": "SentencePiece Unigram (250,112 vocab)",
            "total_params": mt5_train_info["params"]["total_params"],
            "trainable_params": mt5_train_info["params"]["trainable_params"],
            "trainable_pct": mt5_train_info["params"]["trainable_pct"],
            "train_time_sec": mt5_train_info["total_train_time_sec"],
            "inference_time_sec": inf_time,
            "latency_ms_per_sent": round(1000.0 * inf_time / len(eval_src), 2),
            "throughput_sents_per_sec": round(len(eval_src) / max(inf_time, 1e-6), 2),
            **metrics,
        }
        cache["temperature_analysis"]["mT5-small"] = temp_res
        cache["ethical_translations"]["mT5-small"] = eth_preds
        cache["test_predictions"]["mT5-small"] = preds
        cache["training_histories"]["mT5-small"] = mt5_train_info["history"]
        save_cache()
        del mt5_model, mt5_tokenizer
        cleanup_gpu()
    else:
        print("\n[2/4] mT5-small loaded from cache.")

    # =========================================================================
    # MODEL 3: LLaMA (unsloth/Llama-3.2-1B-Instruct, Decoder-Only)
    # =========================================================================
    if "LLaMA-3.2-1B" not in cache["models"]:
        print("\n[3/4] Evaluating & Adapting LLaMA (Llama-3.2-1B-Instruct, Decoder-Only)...")
        llama_model, llama_tokenizer = load_llama_model("unsloth/Llama-3.2-1B-Instruct", device=device)
        base_params = count_parameters(llama_model)

        # (A) Evaluate 3-Shot In-Context Prompting Baseline
        print("  Running LLaMA-3.2-1B (3-Shot In-Context Prompting)...")
        prompt_preds, prompt_inf_time = translate_llama(
            llama_model, llama_tokenizer, eval_src, device, batch_size=8, temperature=0.0, mode="prompting"
        )
        prompt_metrics = compute_translation_metrics(prompt_preds, eval_ref)
        print(
            f"  -> LLaMA-3.2-1B (Prompting) Test BLEU: {prompt_metrics['bleu']:.2f} | "
            f"chrF: {prompt_metrics['chrf']:.2f} | Inference: {prompt_inf_time:.2f}s"
        )

        cache["models"]["LLaMA-3.2-1B (Prompting)"] = {
            "model_name": "LLaMA-3.2-1B (Prompting)",
            "variant": "Llama-3.2-1B-Instruct (3-Shot Prompting)",
            "architecture": "Decoder-Only",
            "adaptation_method": "3-Shot In-Context Prompting",
            "tokenizer_type": "Tiktoken BPE (128,256 vocab)",
            "total_params": base_params["total_params"],
            "trainable_params": 0,
            "trainable_pct": 0.0,
            "train_time_sec": 0.0,
            "inference_time_sec": prompt_inf_time,
            "latency_ms_per_sent": round(1000.0 * prompt_inf_time / len(eval_src), 2),
            "throughput_sents_per_sec": round(len(eval_src) / max(prompt_inf_time, 1e-6), 2),
            **prompt_metrics,
        }
        cache["test_predictions"]["LLaMA-3.2-1B (Prompting)"] = prompt_preds

        # (B) Apply LoRA Fine-Tuning
        print("  Fine-tuning LLaMA-3.2-1B with LoRA (r=16, alpha=32)...")
        llama_peft, llama_lora_info = finetune_llama_lora(
            model=llama_model,
            tokenizer=llama_tokenizer,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            device=device,
            epochs=1,
            max_train_samples=2400,
            batch_size=4,
            grad_accum_steps=4,
            lr=2e-4,
        )

        lora_preds, lora_inf_time = translate_llama(
            llama_peft, llama_tokenizer, eval_src, device, batch_size=8, temperature=0.0, mode="lora"
        )
        lora_metrics = compute_translation_metrics(lora_preds, eval_ref)
        print(
            f"  -> LLaMA-3.2-1B (LoRA) Test BLEU: {lora_metrics['bleu']:.2f} | "
            f"chrF: {lora_metrics['chrf']:.2f} | Inference: {lora_inf_time:.2f}s"
        )

        # Temperature sweep on LLaMA (LoRA)
        temp_res = {}
        for t in temperatures:
            t_preds, t_time = translate_llama(
                llama_peft, llama_tokenizer, temp_src, device, batch_size=8, temperature=t, mode="lora"
            )
            t_m = compute_translation_metrics(t_preds, temp_ref)
            t_m["inference_time_sec"] = t_time
            t_m["sample_translation"] = t_preds[0] if t_preds else ""
            temp_res[str(t)] = t_m

        # Ethical suite on LLaMA
        eth_preds, _ = translate_llama(
            llama_peft, llama_tokenizer, ethical_src, device, batch_size=8, temperature=0.0, mode="lora"
        )

        cache["models"]["LLaMA-3.2-1B"] = {
            "model_name": "LLaMA-3.2-1B",
            "variant": "Llama-3.2-1B-Instruct (LoRA + Prompting)",
            "architecture": "Decoder-Only",
            "adaptation_method": "LoRA Fine-Tuning (r=16, alpha=32)",
            "tokenizer_type": "Tiktoken BPE (128,256 vocab)",
            "total_params": llama_lora_info["params"]["total_params"],
            "trainable_params": llama_lora_info["params"]["trainable_params"],
            "trainable_pct": llama_lora_info["params"]["trainable_pct"],
            "train_time_sec": llama_lora_info["total_train_time_sec"],
            "inference_time_sec": lora_inf_time,
            "latency_ms_per_sent": round(1000.0 * lora_inf_time / len(eval_src), 2),
            "throughput_sents_per_sec": round(len(eval_src) / max(lora_inf_time, 1e-6), 2),
            **lora_metrics,
        }
        cache["temperature_analysis"]["LLaMA-3.2-1B"] = temp_res
        cache["ethical_translations"]["LLaMA-3.2-1B"] = eth_preds
        cache["test_predictions"]["LLaMA-3.2-1B"] = lora_preds
        cache["training_histories"]["LLaMA-3.2-1B"] = llama_lora_info["history"]
        save_cache()
        del llama_peft, llama_model, llama_tokenizer
        cleanup_gpu()
    else:
        print("\n[3/4] LLaMA-3.2-1B loaded from cache.")

    # =========================================================================
    # MODEL 4: Gemma (unsloth/gemma-3-1b-it, Decoder-Only)
    # =========================================================================
    if "Gemma-3-1B" not in cache["models"]:
        print("\n[4/4] Evaluating & Adapting Gemma (gemma-3-1b-it, Decoder-Only)...")
        gemma_model, gemma_tokenizer = load_gemma_model("unsloth/gemma-3-1b-it", device=device)
        base_params = count_parameters(gemma_model)

        # (A) Evaluate 3-Shot In-Context Prompting Baseline
        print("  Running Gemma-3-1B (3-Shot In-Context Prompting)...")
        prompt_preds, prompt_inf_time = translate_gemma(
            gemma_model, gemma_tokenizer, eval_src, device, batch_size=8, temperature=0.0, mode="prompting"
        )
        prompt_metrics = compute_translation_metrics(prompt_preds, eval_ref)
        print(
            f"  -> Gemma-3-1B (Prompting) Test BLEU: {prompt_metrics['bleu']:.2f} | "
            f"chrF: {prompt_metrics['chrf']:.2f} | Inference: {prompt_inf_time:.2f}s"
        )

        cache["models"]["Gemma-3-1B (Prompting)"] = {
            "model_name": "Gemma-3-1B (Prompting)",
            "variant": "gemma-3-1b-it (3-Shot Prompting)",
            "architecture": "Decoder-Only",
            "adaptation_method": "3-Shot In-Context Prompting",
            "tokenizer_type": "SentencePiece (262,144 vocab)",
            "total_params": base_params["total_params"],
            "trainable_params": 0,
            "trainable_pct": 0.0,
            "train_time_sec": 0.0,
            "inference_time_sec": prompt_inf_time,
            "latency_ms_per_sent": round(1000.0 * prompt_inf_time / len(eval_src), 2),
            "throughput_sents_per_sec": round(len(eval_src) / max(prompt_inf_time, 1e-6), 2),
            **prompt_metrics,
        }
        cache["test_predictions"]["Gemma-3-1B (Prompting)"] = prompt_preds

        # (B) Apply LoRA Fine-Tuning
        print("  Fine-tuning Gemma-3-1B with LoRA (r=16, alpha=32)...")
        gemma_peft, gemma_lora_info = finetune_gemma_lora(
            model=gemma_model,
            tokenizer=gemma_tokenizer,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            device=device,
            epochs=1,
            max_train_samples=2400,
            batch_size=4,
            grad_accum_steps=4,
            lr=2e-4,
        )

        lora_preds, lora_inf_time = translate_gemma(
            gemma_peft, gemma_tokenizer, eval_src, device, batch_size=8, temperature=0.0, mode="lora"
        )
        lora_metrics = compute_translation_metrics(lora_preds, eval_ref)
        print(
            f"  -> Gemma-3-1B (LoRA) Test BLEU: {lora_metrics['bleu']:.2f} | "
            f"chrF: {lora_metrics['chrf']:.2f} | Inference: {lora_inf_time:.2f}s"
        )

        # Temperature sweep on Gemma (LoRA)
        temp_res = {}
        for t in temperatures:
            t_preds, t_time = translate_gemma(
                gemma_peft, gemma_tokenizer, temp_src, device, batch_size=8, temperature=t, mode="lora"
            )
            t_m = compute_translation_metrics(t_preds, temp_ref)
            t_m["inference_time_sec"] = t_time
            t_m["sample_translation"] = t_preds[0] if t_preds else ""
            temp_res[str(t)] = t_m

        # Ethical suite on Gemma
        eth_preds, _ = translate_gemma(
            gemma_peft, gemma_tokenizer, ethical_src, device, batch_size=8, temperature=0.0, mode="lora"
        )

        cache["models"]["Gemma-3-1B"] = {
            "model_name": "Gemma-3-1B",
            "variant": "gemma-3-1b-it (LoRA + Prompting)",
            "architecture": "Decoder-Only",
            "adaptation_method": "LoRA Fine-Tuning (r=16, alpha=32)",
            "tokenizer_type": "SentencePiece (262,144 vocab)",
            "total_params": gemma_lora_info["params"]["total_params"],
            "trainable_params": gemma_lora_info["params"]["trainable_params"],
            "trainable_pct": gemma_lora_info["params"]["trainable_pct"],
            "train_time_sec": gemma_lora_info["total_train_time_sec"],
            "inference_time_sec": lora_inf_time,
            "latency_ms_per_sent": round(1000.0 * lora_inf_time / len(eval_src), 2),
            "throughput_sents_per_sec": round(len(eval_src) / max(lora_inf_time, 1e-6), 2),
            **lora_metrics,
        }
        cache["temperature_analysis"]["Gemma-3-1B"] = temp_res
        cache["ethical_translations"]["Gemma-3-1B"] = eth_preds
        cache["test_predictions"]["Gemma-3-1B"] = lora_preds
        cache["training_histories"]["Gemma-3-1B"] = gemma_lora_info["history"]
        save_cache()
        del gemma_peft, gemma_model, gemma_tokenizer
        cleanup_gpu()
    else:
        print("\n[4/4] Gemma-3-1B loaded from cache.")

    # =========================================================================
    # EXPORT STRUCTURED SUMMARY TABLES & SAMPLES
    # =========================================================================
    primary_order = ["Basic Transformer", "mT5-small", "LLaMA-3.2-1B", "Gemma-3-1B"]
    all_order = [
        "Basic Transformer",
        "mT5-small",
        "LLaMA-3.2-1B (Prompting)",
        "LLaMA-3.2-1B",
        "Gemma-3-1B (Prompting)",
        "Gemma-3-1B",
    ]

    summary_rows = [cache["models"][k] for k in all_order if k in cache["models"]]
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("results/metrics_summary.csv", index=False)

    with open("results/metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "primary_models": {k: cache["models"][k] for k in primary_order if k in cache["models"]},
                "all_configurations": cache["models"],
                "training_histories": cache["training_histories"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open("results/temperature_analysis.json", "w", encoding="utf-8") as f:
        json.dump(cache["temperature_analysis"], f, ensure_ascii=False, indent=2)

    ethical_eval = evaluate_ethical_suite(cache["ethical_translations"])
    with open("results/ethical_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(ethical_eval, f, ensure_ascii=False, indent=2)

    # Save 25 qualitative side-by-side test translations
    samples_out = []
    for i in range(min(25, len(eval_src))):
        entry = {
            "index": i + 1,
            "source_en": eval_src[i],
            "reference_fr": eval_ref[i],
            "translations": {},
        }
        for m_name in all_order:
            if m_name in cache["test_predictions"]:
                entry["translations"][m_name] = cache["test_predictions"][m_name][i]
        samples_out.append(entry)

    with open("results/test_translations_sample.json", "w", encoding="utf-8") as f:
        json.dump(samples_out, f, ensure_ascii=False, indent=2)

    print("\n=== Final Quantitative Summary Across All Configurations ===")
    cols = [
        "model_name",
        "architecture",
        "adaptation_method",
        "bleu",
        "chrf",
        "train_time_sec",
        "inference_time_sec",
        "total_params",
        "trainable_params",
    ]
    print(df_summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()

