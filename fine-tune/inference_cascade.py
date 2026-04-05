#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_cascade.py
═══════════════════════════════════════════════════════════════════════════════
Cascade + Ensemble Inference cho AMRBART Vietnamese AMR Parser

Pipeline:
  1. Chạy model-A (nhanh/nhỏ, e.g. checkpoint-852)
  2. Kiểm tra chất lượng output (thin graph detection)
  3. Nếu thin → chạy model-B (mạnh hơn, e.g. checkpoint-9256)
  4. [Optional] Ensemble: chọn output tốt hơn từ cả 2 model

Usage:
  python inference_cascade.py \
      --model_a /path/to/checkpoint-852 \
      --model_b /path/to/checkpoint-9256 \
      --sentence "Hãy cùng lắng_nghe bài hát này"

  python inference_cascade.py \
      --model_a /path/to/checkpoint-852 \
      --model_b /path/to/checkpoint-9256 \
      --input_file sentences.txt \
      --output_file output_cascade.txt \
      --mode ensemble   # hoặc cascade (default)
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import time
from typing import List, Tuple, Optional

import torch
import penman
from tqdm import tqdm

# Thêm fine-tune dir vào sys.path để import local modules
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from transformers import BartForConditionalGeneration
from model_interface.tokenization_bart import AMRBartTokenizer

DUMMY = '(z / amr-empty)'


# ─────────────────────────────────────────────────────────────────────────────
# Quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def count_amr_edges(amr_str: str) -> int:
    """Đếm số edges (không tính :instance) trong graph."""
    try:
        graph = penman.decode(amr_str)
        return sum(1 for t in graph.triples if t[1] != ':instance')
    except Exception:
        return 0


def is_thin_graph(amr_str: str, source_sentence: str,
                  edges_threshold: int = 2,
                  length_ratio: float = 1.5,
                  length_floor: int = 15) -> bool:
    """
    Phát hiện thin graph dựa trên nhiều heuristics:
      1. Chứa 'amr-unknown' hoặc 'amr-empty'
      2. Số edges quá ít so với nguồn
      3. Tổng tokens AMR quá ngắn so với số từ đầu vào
    """
    if amr_str in (DUMMY, '(z / amr-empty)', '(z / amr-unknown)'):
        return True

    if 'amr-empty' in amr_str:
        return True

    edge_count = count_amr_edges(amr_str)
    if edge_count <= edges_threshold:
        return True

    n_words = len(source_sentence.split())
    min_amr_tokens = max(length_floor, int(n_words * length_ratio))
    amr_token_count = len(amr_str.split())
    if amr_token_count < min_amr_tokens:
        return True

    return False


def score_amr_quality(amr_str: str, source_sentence: str) -> float:
    """
    Tính điểm chất lượng cho output AMR (không cần gold).
    Dùng để so sánh 2 outputs trong ensemble mode.

    Score = weighted sum of:
      - edge_count: nhiều edges = phong phú hơn
      - is_parseable: có parse được bằng penman không
      - no_amr_unknown: không có mode amr-unknown
      - length_ratio: độ dài output so với input
    """
    try:
        graph = penman.decode(amr_str)
        parseable = 1.0
    except Exception:
        return 0.0

    n_edges = count_amr_edges(amr_str)
    n_words = len(source_sentence.split())

    no_unk   = 0.0 if 'amr-unknown' in amr_str else 1.0
    no_empty = 0.0 if 'amr-empty' in amr_str else 1.0
    edge_score = min(1.0, n_edges / max(1, n_words))   # normalize bởi số từ

    return (0.5 * edge_score + 0.3 * no_unk + 0.2 * no_empty)


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def remove_amr_unknown_mode(graph: penman.Graph) -> penman.Graph:
    """Xóa edge ':mode amr-unknown' khỏi graph (giữ phần còn lại)."""
    keep = [
        t for t in graph.triples
        if not (t[1] == ':mode' and str(t[2]) == 'amr-unknown')
    ]
    if len(keep) == len(graph.triples) or len(keep) == 0:
        return graph
    return penman.Graph(keep)


# ─────────────────────────────────────────────────────────────────────────────
# Single model inference
# ─────────────────────────────────────────────────────────────────────────────

def decode_to_amr(token_ids: list, tokenizer) -> str:
    """Decode token IDs → Penman AMR string."""
    seq = list(token_ids)
    if not seq:
        return DUMMY

    seq[0] = tokenizer.bos_token_id
    clean = []
    for tok in seq:
        if tok == tokenizer.pad_token_id:
            break
        mapped = tokenizer.eos_token_id if tok == tokenizer.amr_eos_token_id else tok
        clean.append(mapped)
        if mapped == tokenizer.eos_token_id:
            break

    try:
        graph, _, _ = tokenizer.decode_amr(clean, restore_name_ops=False)
        graph = remove_amr_unknown_mode(graph)
        enc = penman.encode(graph)
        return enc if enc.strip() not in ('()', '') else DUMMY
    except Exception:
        return DUMMY


def parse_batch(
    sentences: List[str],
    model,
    tokenizer,
    num_beams: int = 10,
    num_beam_groups: int = 5,
    diversity_penalty: float = 0.8,
    max_new_tokens: int = 400,
    batch_size: int = 2,
    device: str = 'cuda',
) -> List[str]:
    """Parse một batch câu bằng Diverse Beam Search."""
    model.eval()
    model.to(device)
    results = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i: i + batch_size]

        # Dynamic min_length
        min_lens = [max(30, int(len(s.split()) * 3.5)) for s in batch]
        dynamic_min = min(min_lens)

        inputs = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                min_length=dynamic_min,
                max_new_tokens=max_new_tokens,
                forced_bos_token_id=tokenizer.amr_bos_token_id,
                early_stopping=True,
            )

        for seq in outputs:
            results.append(decode_to_amr(seq.tolist(), tokenizer))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Cascade pipeline
# ─────────────────────────────────────────────────────────────────────────────

def cascade_inference(
    sentences: List[str],
    model_a, tokenizer_a,
    model_b, tokenizer_b,
    num_beams: int = 10,
    num_beam_groups: int = 5,
    diversity_penalty: float = 0.8,
    max_new_tokens: int = 400,
    batch_size: int = 2,
    device: str = 'cuda',
) -> Tuple[List[str], List[str]]:
    """
    Cascade: model-A trước → nếu thin → model-B làm fallback.

    Trả về:
        results: list output AMR
        sources: ['model_a', 'model_b', ...] — biết câu nào dùng model nào
    """
    print(f"\n[Cascade] Bước 1: Chạy model-A cho {len(sentences)} câu...")
    results_a = []
    for i in tqdm(range(0, len(sentences), batch_size), desc='Model-A'):
        batch = sentences[i: i + batch_size]
        batch_results = parse_batch(
            batch, model_a, tokenizer_a,
            num_beams=num_beams, num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            max_new_tokens=max_new_tokens, batch_size=len(batch), device=device,
        )
        results_a.extend(batch_results)

    # Phát hiện câu cần fallback
    need_fallback = [
        i for i, (sent, amr) in enumerate(zip(sentences, results_a))
        if is_thin_graph(amr, sent)
    ]

    print(f"\n[Cascade] {len(need_fallback)}/{len(sentences)} câu thin → fallback Model-B")

    results   = list(results_a)
    sources   = ['model_a'] * len(sentences)

    if need_fallback:
        fallback_sents = [sentences[i] for i in need_fallback]
        print(f"[Cascade] Bước 2: Chạy model-B cho {len(fallback_sents)} câu cần fallback...")
        results_b = []
        for i in tqdm(range(0, len(fallback_sents), batch_size), desc='Model-B fallback'):
            batch = fallback_sents[i: i + batch_size]
            batch_results = parse_batch(
                batch, model_b, tokenizer_b,
                num_beams=num_beams, num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                max_new_tokens=max_new_tokens, batch_size=len(batch), device=device,
            )
            results_b.extend(batch_results)

        for idx, amr_b in zip(need_fallback, results_b):
            results[idx] = amr_b
            sources[idx] = 'model_b'

    return results, sources


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble pipeline (cả 2 model → pick tốt hơn)
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_inference(
    sentences: List[str],
    model_a, tokenizer_a,
    model_b, tokenizer_b,
    num_beams: int = 10,
    num_beam_groups: int = 5,
    diversity_penalty: float = 0.8,
    max_new_tokens: int = 400,
    batch_size: int = 2,
    device: str = 'cuda',
) -> Tuple[List[str], List[str]]:
    """
    Ensemble: sinh output từ CẢ 2 model → chọn cái chất lượng tốt hơn.
    Hai model "chia sẻ thông tin" qua candidate selection.
    """
    print(f"\n[Ensemble] Chạy model-A ({len(sentences)} câu)...")
    results_a = []
    for i in tqdm(range(0, len(sentences), batch_size), desc='Model-A'):
        batch = sentences[i: i + batch_size]
        results_a.extend(parse_batch(
            batch, model_a, tokenizer_a,
            num_beams=num_beams, num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            max_new_tokens=max_new_tokens, batch_size=len(batch), device=device,
        ))

    print(f"\n[Ensemble] Chạy model-B ({len(sentences)} câu)...")
    results_b = []
    for i in tqdm(range(0, len(sentences), batch_size), desc='Model-B'):
        batch = sentences[i: i + batch_size]
        results_b.extend(parse_batch(
            batch, model_b, tokenizer_b,
            num_beams=num_beams, num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            max_new_tokens=max_new_tokens, batch_size=len(batch), device=device,
        ))

    # Chọn output tốt hơn dựa trên quality score
    print("\n[Ensemble] Chọn output tốt hơn từ 2 model...")
    results = []
    sources = []
    for sent, amr_a, amr_b in zip(sentences, results_a, results_b):
        score_a = score_amr_quality(amr_a, sent)
        score_b = score_amr_quality(amr_b, sent)
        if score_b > score_a:
            results.append(amr_b)
            sources.append('model_b')
        else:
            results.append(amr_a)
            sources.append('model_a')

    a_count = sources.count('model_a')
    b_count = sources.count('model_b')
    print(f"  ✓ Model-A thắng: {a_count} | Model-B thắng: {b_count}")

    return results, sources


# ─────────────────────────────────────────────────────────────────────────────
# Load model helper
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(checkpoint_path: str, device: str):
    """Load BartForConditionalGeneration + AMRBartTokenizer từ checkpoint."""
    print(f"  Loading tokenizer from: {checkpoint_path}")
    tokenizer = AMRBartTokenizer.from_pretrained(checkpoint_path, use_fast=False)

    print(f"  Loading model from: {checkpoint_path}")
    model = BartForConditionalGeneration.from_pretrained(checkpoint_path)
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✓ {params:.0f}M params loaded")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='AMRBART Cascade/Ensemble Inference — 2 models'
    )
    parser.add_argument('--model_a', required=True,
                        help='Checkpoint model A (chạy trước, thường nhỏ/nhanh hơn)')
    parser.add_argument('--model_b', required=True,
                        help='Checkpoint model B (fallback/ensemble, thường mạnh hơn)')
    parser.add_argument('--sentence', default=None,
                        help='Parse một câu đơn')
    parser.add_argument('--input_file', default=None,
                        help='File chứa câu (mỗi dòng 1 câu)')
    parser.add_argument('--output_file', default='output_cascade.txt',
                        help='File output AMR')
    parser.add_argument('--mode', choices=['cascade', 'ensemble'], default='cascade',
                        help='cascade: A trước → B nếu thin | ensemble: A+B → pick tốt hơn')
    parser.add_argument('--num_beams', type=int, default=10)
    parser.add_argument('--num_beam_groups', type=int, default=5)
    parser.add_argument('--diversity_penalty', type=float, default=0.8)
    parser.add_argument('--max_new_tokens', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*65}")
    print(f" AMRBART Cascade/Ensemble Inference")
    print(f"{'='*65}")
    print(f" Mode:         {args.mode.upper()}")
    print(f" Model A:      {args.model_a}")
    print(f" Model B:      {args.model_b}")
    print(f" Device:       {device}")
    print(f" num_beams:    {args.num_beams} ({args.num_beam_groups} groups)")
    print(f" diversity:    {args.diversity_penalty}")
    print(f"{'='*65}\n")

    # Load cả 2 model
    print("[1/2] Loading Model A...")
    model_a, tokenizer_a = load_model_and_tokenizer(args.model_a, device)
    model_a.to(device)

    print("\n[2/2] Loading Model B...")
    model_b, tokenizer_b = load_model_and_tokenizer(args.model_b, device)
    model_b.to(device)

    # Collect sentences
    if args.sentence:
        sentences = [args.sentence]
    elif args.input_file:
        with open(args.input_file, encoding='utf-8') as f:
            sentences = [l.strip() for l in f if l.strip()]
        print(f"\nLoaded {len(sentences)} sentences from {args.input_file}")
    else:
        parser.error("Cần --sentence hoặc --input_file")

    # Run inference
    t0 = time.time()

    if args.mode == 'cascade':
        results, sources = cascade_inference(
            sentences, model_a, tokenizer_a, model_b, tokenizer_b,
            num_beams=args.num_beams, num_beam_groups=args.num_beam_groups,
            diversity_penalty=args.diversity_penalty,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size, device=device,
        )
    else:  # ensemble
        results, sources = ensemble_inference(
            sentences, model_a, tokenizer_a, model_b, tokenizer_b,
            num_beams=args.num_beams, num_beam_groups=args.num_beam_groups,
            diversity_penalty=args.diversity_penalty,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size, device=device,
        )

    elapsed = time.time() - t0

    # Output
    print(f"\n{'─'*65}")
    if args.sentence:
        print(f"# ::id 0")
        print(f"# ::annotator {args.mode}-amr [{sources[0]}]")
        print(f"# ::snt {args.sentence}")
        print(results[0])
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for i, (sent, amr, src) in enumerate(zip(sentences, results, sources)):
                f.write(f"# ::id {i}\n# ::annotator {args.mode}-amr [{src}]\n")
                f.write(f"# ::snt {sent}\n{amr}\n\n")

        a_count = sources.count('model_a')
        b_count = sources.count('model_b')
        print(f"\n✅ Done: {len(results)} sentences in {elapsed:.1f}s "
              f"({elapsed/len(results):.2f}s/sent)")
        print(f"   Model-A used: {a_count} | Model-B used: {b_count}")
        print(f"   Output: {args.output_file}")


if __name__ == '__main__':
    main()
