#!/usr/bin/env python3
import argparse
import datetime
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfFileSystem
from tqdm import tqdm

try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    llama_cpp = None
    LLAMA_CPP_AVAILABLE = False

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

def chance_normalize(raw_score: float, chance: float) -> float:
    return max(0.0, (raw_score - chance) / (1.0 - chance))

def compute_intelligence_index(
    hellaswag_acc: float,
    arc_easy_acc: float,
    arc_challenge_acc: float,
    piqa_acc: float,
    arithmark_acc: float,
) -> Dict[str, Union[float, Dict[str, float]]]:
    arc_combined = (arc_easy_acc + arc_challenge_acc) / 2.0
    s_hellaswag = chance_normalize(hellaswag_acc, 0.25)
    s_arc_easy = chance_normalize(arc_easy_acc, 0.25)
    s_arc_challenge = chance_normalize(arc_challenge_acc, 0.25)
    s_arc = chance_normalize(arc_combined, 0.25)
    s_piqa = chance_normalize(piqa_acc, 0.50)
    s_arith = chance_normalize(arithmark_acc, 0.25)

    weighted_sum = (1.0 * s_hellaswag) + (1.0 * s_arc) + (1.0 * s_piqa) + (0.65 * s_arith)
    intelligence_index = (weighted_sum / 3.65) * 100.0

    return {
        "intelligence_index": round(intelligence_index, 2),
        "weights": {
            "hellaswag": 1.0,
            "arc_avg": 1.0,
            "piqa": 1.0,
            "arithmark3": 0.65,
            "denominator": 3.65,
        },
        "scores_raw": {
            "hellaswag_acc_norm": round(hellaswag_acc * 100.0, 2),
            "arc_easy_acc_norm": round(arc_easy_acc * 100.0, 2),
            "arc_challenge_acc_norm": round(arc_challenge_acc * 100.0, 2),
            "piqa_acc_norm": round(piqa_acc * 100.0, 2),
            "arithmark3_acc": round(arithmark_acc * 100.0, 2),
        },
        "scores_normalized": {
            "hellaswag_norm": round(s_hellaswag * 100.0, 2),
            "arc_easy_norm": round(s_arc_easy * 100.0, 2),
            "arc_challenge_norm": round(s_arc_challenge * 100.0, 2),
            "piqa_norm": round(s_piqa * 100.0, 2),
            "arithmark3_norm": round(s_arith * 100.0, 2),
            "arc_avg_norm": round(s_arc * 100.0, 2),
        },
    }

class EvaluatorBackend:
    def score_continuations(self, context: str, endings: List[str]) -> int:
        raise NotImplementedError

class PyTorchBackend(EvaluatorBackend):
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.model.eval()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

    def score_continuations(self, context: str, endings: List[str]) -> int:
        ctx_ids = self.tokenizer(context, add_special_tokens=False).input_ids
        if not ctx_ids:
            ctx_ids = [self.tokenizer.bos_token_id or 1]

        log_likelihoods = []
        for ending in endings:
            end_ids = self.tokenizer(ending, add_special_tokens=False).input_ids
            if not end_ids:
                log_likelihoods.append(-1e9)
                continue

            full_ids = ctx_ids + end_ids
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits[0]

            ctx_len = len(ctx_ids)
            end_len = len(end_ids)
            pred_logits = logits[ctx_len - 1 : ctx_len + end_len - 1]
            target_tokens = torch.tensor(end_ids, dtype=torch.long, device=self.device)

            log_probs = F.log_softmax(pred_logits, dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
            log_likelihoods.append(token_log_probs.sum().item())

        return int(torch.tensor(log_likelihoods).argmax().item())

class LlamaCppBackend(EvaluatorBackend):
    def __init__(self, gguf_path: str, n_ctx: int = 1024, n_gpu_layers: int = -1):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is required for GGUF evaluation. Install with pip install llama-cpp-python.")
        self.llm = llama_cpp.Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            logits_all=True,
            verbose=False,
        )

    def score_continuations(self, context: str, endings: List[str]) -> int:
        ctx_tokens = self.llm.tokenize(context.encode("utf-8"), add_bos=True)
        if not ctx_tokens:
            ctx_tokens = [1]

        log_likelihoods = []
        for ending in endings:
            end_tokens = self.llm.tokenize(ending.encode("utf-8"), add_bos=False)
            if not end_tokens:
                log_likelihoods.append(-1e9)
                continue

            all_tokens = ctx_tokens + end_tokens
            if len(all_tokens) > self.llm.n_ctx():
                # Truncate context from the left if necessary
                excess = len(all_tokens) - self.llm.n_ctx()
                trimmed_ctx = ctx_tokens[excess:]
                all_tokens = trimmed_ctx + end_tokens
                ctx_len = len(trimmed_ctx)
            else:
                ctx_len = len(ctx_tokens)

            self.llm.reset()
            self.llm.eval(all_tokens)
            scores = np.array(self.llm.scores[:len(all_tokens)])

            log_prob = 0.0
            for i, tok in enumerate(end_tokens):
                pos = ctx_len - 1 + i
                logits = scores[pos]
                shift_logits = logits - np.max(logits)
                lprobs = shift_logits - np.log(np.sum(np.exp(shift_logits)))
                log_prob += float(lprobs[tok])
            log_likelihoods.append(log_prob)

        return int(np.argmax(log_likelihoods))

def evaluate_arithmark_subset(
    backend: EvaluatorBackend,
    max_samples: Optional[int] = None,
) -> float:
    print("Loading ArithMark 3.0 benchmark dataset from Hugging Face Hub...")
    fs = HfFileSystem()
    examples = []
    with fs.open("datasets/AxiomicLabs/Arithmark-3.0/arithmark-3.jsonl", "r") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and max_samples > 0 and idx >= max_samples:
                break
            line_str = line.strip()
            if not line_str:
                continue
            examples.append(json.loads(line_str))

    correct = 0
    total = len(examples)
    for ex in tqdm(examples, desc="Evaluating ArithMark 3.0"):
        ctx = ex["ctx"]
        endings = ex["endings"]
        label = int(ex["label"])
        pred = backend.score_continuations(ctx, endings)
        if pred == label:
            correct += 1

    acc = correct / max(1, total)
    print(f"ArithMark 3.0 Accuracy: {acc * 100.0:.2f}% ({correct}/{total})")
    return acc

def evaluate_hellaswag_subset(
    backend: EvaluatorBackend,
    max_samples: Optional[int] = None,
) -> float:
    print("Loading HellaSwag validation dataset from Hugging Face...")
    ds = iter(load_dataset("hellaswag", split="validation", streaming=True))
    correct = 0
    total = 0

    limit_desc = max_samples if (max_samples and max_samples > 0) else "full"
    pbar = tqdm(total=max_samples if (max_samples and max_samples > 0) else None, desc=f"Evaluating HellaSwag ({limit_desc})")
    for ex in ds:
        if max_samples is not None and max_samples > 0 and total >= max_samples:
            break
        ctx = ex["ctx"]
        endings = ex["endings"]
        label = int(ex["label"])
        pred = backend.score_continuations(ctx, endings)
        if pred == label:
            correct += 1
        total += 1
        pbar.update(1)
    pbar.close()

    acc = correct / max(1, total)
    print(f"HellaSwag Accuracy: {acc * 100.0:.2f}% ({correct}/{total})")
    return acc

def evaluate_arc_subset(
    backend: EvaluatorBackend,
    config_name: str = "ARC-Easy",
    max_samples: Optional[int] = None,
) -> float:
    print(f"Loading AI2_ARC {config_name} test dataset from Hugging Face...")
    ds = iter(load_dataset("ai2_arc", config_name, split="test", streaming=True))
    correct = 0
    total = 0

    limit_desc = max_samples if (max_samples and max_samples > 0) else "full"
    pbar = tqdm(total=max_samples if (max_samples and max_samples > 0) else None, desc=f"Evaluating {config_name} ({limit_desc})")
    for ex in ds:
        if max_samples is not None and max_samples > 0 and total >= max_samples:
            break
        q = ex["question"]
        choices = ex["choices"]
        labels = choices["label"]
        texts = choices["text"]
        ans_key = ex["answerKey"]
        if ans_key not in labels:
            continue
        label_idx = labels.index(ans_key)
        endings = [f" {t}" for t in texts]
        pred = backend.score_continuations(q, endings)
        if pred == label_idx:
            correct += 1
        total += 1
        pbar.update(1)
    pbar.close()

    acc = correct / max(1, total)
    print(f"{config_name} Accuracy: {acc * 100.0:.2f}% ({correct}/{total})")
    return acc

def evaluate_piqa_subset(
    backend: EvaluatorBackend,
    max_samples: Optional[int] = None,
) -> float:
    print("Loading PIQA validation dataset from Hugging Face...")
    ds = iter(load_dataset("piqa", split="validation", streaming=True, trust_remote_code=True))
    correct = 0
    total = 0

    limit_desc = max_samples if (max_samples and max_samples > 0) else "full"
    pbar = tqdm(total=max_samples if (max_samples and max_samples > 0) else None, desc=f"Evaluating PIQA ({limit_desc})")
    for ex in ds:
        if max_samples is not None and max_samples > 0 and total >= max_samples:
            break
        goal = ex["goal"]
        endings = [f" {ex['sol1']}", f" {ex['sol2']}"]
        label = int(ex["label"])
        pred = backend.score_continuations(goal, endings)
        if pred == label:
            correct += 1
        total += 1
        pbar.update(1)
    pbar.close()

    acc = correct / max(1, total)
    print(f"PIQA Accuracy: {acc * 100.0:.2f}% ({correct}/{total})")
    return acc

def main():
    parser = argparse.ArgumentParser(description="Evaluate Open SLM Leaderboard metrics (PyTorch or GGUF via llama-cpp-python)")
    parser.add_argument("--model-path", type=str, default="models/mark-slm-v3-150m", help="Path to HF model directory or .gguf file")
    parser.add_argument("--gguf-path", type=str, default=None, help="Explicit path to GGUF file to evaluate via llama-cpp-python")
    parser.add_argument("--samples", type=int, default=0, help="Number of samples to evaluate (0 or negative for full split)")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save canonical results JSON (defaults to <model-path>/open_slm_leaderboard_results.json)")
    args = parser.parse_args()

    max_samples = args.samples if args.samples > 0 else None

    # Determine backend
    gguf_candidate = args.gguf_path
    if not gguf_candidate and args.model_path.endswith(".gguf"):
        gguf_candidate = args.model_path
    elif not gguf_candidate and os.path.isdir(args.model_path):
        # Look for gguf in directory if user requests or default
        ggufs = list(Path(args.model_path).glob("*.gguf"))
        if ggufs and not (Path(args.model_path) / "config.json").exists():
            gguf_candidate = str(ggufs[0])

    if gguf_candidate:
        print(f"Initializing LlamaCppBackend with {gguf_candidate}...")
        backend = LlamaCppBackend(gguf_candidate)
        model_name = Path(gguf_candidate).stem
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing PyTorchBackend with {args.model_path} on {device}...")
        backend = PyTorchBackend(args.model_path, device=device)
        model_name = Path(args.model_path).name

    # Run evaluations
    arith_acc = evaluate_arithmark_subset(backend, max_samples=max_samples)
    hellaswag_acc = evaluate_hellaswag_subset(backend, max_samples=max_samples)
    arc_easy_acc = evaluate_arc_subset(backend, config_name="ARC-Easy", max_samples=max_samples)
    arc_challenge_acc = evaluate_arc_subset(backend, config_name="ARC-Challenge", max_samples=max_samples)
    piqa_acc = evaluate_piqa_subset(backend, max_samples=max_samples)

    summary = compute_intelligence_index(
        hellaswag_acc=hellaswag_acc,
        arc_easy_acc=arc_easy_acc,
        arc_challenge_acc=arc_challenge_acc,
        piqa_acc=piqa_acc,
        arithmark_acc=arith_acc,
    )

    canonical_result = {
        "benchmark": "Open SLM Leaderboard",
        "organization": "AxiomicLabs & Miceworks",
        "model_id": f"miceworks/{model_name}",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **summary,
    }

    print("\n" + "=" * 65)
    print("OPEN SLM LEADERBOARD INTELLIGENCE INDEX BENCHMARK REPORT")
    print("=" * 65)
    print(f"Model ID:               {canonical_result['model_id']}")
    print(f"Intelligence Index:     {summary['intelligence_index']:.2f}")
    print(f"ArithMark 3.0:          {summary['scores_raw']['arithmark3_acc']:.2f}% (Normalized: {summary['scores_normalized']['arithmark3_norm']:.2f}%)")
    print(f"ARC-Easy:               {summary['scores_raw']['arc_easy_acc_norm']:.2f}%")
    print(f"ARC-Challenge:          {summary['scores_raw']['arc_challenge_acc_norm']:.2f}%")
    print(f"ARC-Combined:           {(summary['scores_raw']['arc_easy_acc_norm'] + summary['scores_raw']['arc_challenge_acc_norm']) / 2.0:.2f}% (Normalized: {summary['scores_normalized']['arc_avg_norm']:.2f}%)")
    print(f"PIQA:                   {summary['scores_raw']['piqa_acc_norm']:.2f}% (Normalized: {summary['scores_normalized']['piqa_norm']:.2f}%)")
    print(f"HellaSwag:              {summary['scores_raw']['hellaswag_acc_norm']:.2f}% (Normalized: {summary['scores_normalized']['hellaswag_norm']:.2f}%)")
    print("=" * 65)

    out_path = Path(args.output_json) if args.output_json else (Path(args.model_path) / "open_slm_leaderboard_results.json" if os.path.isdir(args.model_path) else Path("open_slm_leaderboard_results.json"))
    with open(out_path, "w") as f:
        json.dump(canonical_result, f, indent=2)
    print(f"Saved benchmark summary to {out_path}")

if __name__ == "__main__":
    main()
