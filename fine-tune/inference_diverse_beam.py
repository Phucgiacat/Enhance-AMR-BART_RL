"""
inference_diverse_beam.py
─────────────────────────────────────────────────────────────────────────────
Inference script cho AMRBART Vietnamese AMR Parser.
Sử dụng Diverse Beam Search để cải thiện chất lượng parse so với beam search
thông thường — đặc biệt hiệu quả với câu phức tạp và graph sâu.

Diverse Beam Search chia `num_beams` thành `num_beam_groups` nhóm riêng biệt.
Mỗi nhóm tìm kiếm theo hướng khác → tránh tất cả beams đổ về path ngắn nhất.

Usage:
    python inference_diverse_beam.py \
        --model_path /path/to/checkpoint \
        --input_file sentences.txt \
        --output_file parsed_amr.txt \
        --num_beams 10 \
        --num_beam_groups 5 \
        --diversity_penalty 0.8

    # Hoặc parse 1 câu trực tiếp:
    python inference_diverse_beam.py \
        --model_path /path/to/checkpoint \
        --sentence "Hãy cùng lắng_nghe bài hát mà Coca-Cola đã tạo ra"
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import os
import sys
import torch
import penman
from typing import List
from tqdm import tqdm

# ─── Đảm bảo import đúng từ thư mục fine-tune ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Decode token IDs → AMR string (tái sử dụng logic từ main.py)
# ─────────────────────────────────────────────────────────────────────────────

def decode_to_amr(token_ids: list, tokenizer) -> str:
    """Decode token IDs thành Penman AMR string."""
    DUMMY = '(z / amr-empty)'
    seq = list(token_ids)
    if not seq:
        return DUMMY

    # Inject BOS nếu cần
    seq[0] = tokenizer.bos_token_id

    # Truncate tại PAD/EOS
    clean = []
    for tok in seq:
        if tok == tokenizer.pad_token_id:
            break
        mapped = tokenizer.eos_token_id if tok == tokenizer.amr_eos_token_id else tok
        clean.append(mapped)
        if mapped == tokenizer.eos_token_id:
            break

    try:
        graph, status, _ = tokenizer.decode_amr(clean, restore_name_ops=False)
        encoded = penman.encode(graph)
        return encoded if encoded.strip() not in ('()', '') else DUMMY
    except Exception:
        return DUMMY


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic min_length estimation
# ─────────────────────────────────────────────────────────────────────────────

def compute_min_length(sentence: str, ratio: float = 3.5, floor: int = 30) -> int:
    """
    Ước tính min_length cho AMR output dựa trên độ dài câu đầu vào.

    Lưu ý quan trọng về AMR tokens:
      - 1 từ tiếng Việt → có thể 1-3 AMR subword tokens
      - ':mode', ':ARG0', '<pointer:N>' mỗi cái là 1 token đặc biệt
      - Thin graph '(z1/X :mode amr-unknown)' ≈ 12-18 tokens trong AMRBartTokenizer
      → ratio=3.5, floor=30 để chắc chắn vượt qua thin graph

    Calibration (9 từ test case):
      "Hãy cùng lắng_nghe bài hát mà Coca-Cola đã tạo ra"
      → 9 words × 3.5 = 31.5 → max(30, 31) = 31 tokens  ✅ > thin graph (12-18)

    Scale theo câu dài:
      5  từ → max(30, 17 ) = 30  tokens
      10 từ → max(30, 35 ) = 35  tokens
      15 từ → max(30, 52 ) = 52  tokens
      25 từ → max(30, 87 ) = 87  tokens
    """
    n_words = len(sentence.split())
    return max(floor, int(n_words * ratio))


# ─────────────────────────────────────────────────────────────────────────────
# Core inference function
# ─────────────────────────────────────────────────────────────────────────────

def parse_sentences(
    model,
    tokenizer,
    sentences: List[str],
    num_beams: int = 10,
    num_beam_groups: int = 5,
    diversity_penalty: float = 0.8,
    max_new_tokens: int = 400,
    min_length_ratio: float = 1.5,   # ← dynamic min_length ratio (words × ratio)
    min_length_floor: int = 15,       # ← minimum tokens bất kể câu ngắn cỡ nào
    batch_size: int = 4,
    device: str = "cuda",
) -> List[str]:
    """
    Parse danh sách câu → AMR graphs dùng Diverse Beam Search.

    min_length được tính động per batch:
        min_length = max(floor, min_words_in_batch × ratio)
    Lấy min của batch để đảm bảo an toàn cho câu ngắn nhất.
    """
    assert num_beams % num_beam_groups == 0, (
        f"num_beams ({num_beams}) phải chia hết cho num_beam_groups ({num_beam_groups})"
    )

    model.eval()
    model.to(device)
    results = []

    # ── Ban 'amr-unknown' để buộc model sinh graph thực sự ───────────────────
    # amr-unknown là fallback token — ban nó buộc model chọn structure thay thế
    bad_words_ids = None
    try:
        unk_ids = tokenizer.encode("amr-unknown", add_special_tokens=False)
        if unk_ids:
            bad_words_ids = [unk_ids]  # format: list of list of token ids
            print(f"Banning token: 'amr-unknown' = {unk_ids}")
    except Exception:
        pass

    for i in tqdm(range(0, len(sentences), batch_size), desc="Parsing"):
        batch_sents = sentences[i: i + batch_size]

        # ── Tính dynamic min_length cho batch này ────────────────────────────
        # Lấy min của batch → an toàn cho câu ngắn nhất trong batch
        batch_min_lengths = [
            compute_min_length(s, ratio=min_length_ratio, floor=min_length_floor)
            for s in batch_sents
        ]
        dynamic_min = min(batch_min_lengths)

        # Tokenize
        inputs = tokenizer(
            batch_sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],

                # ── Diverse Beam Search ──────────────────────────────────
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                # ── Dynamic min_length per batch ─────────────────────────
                min_length=dynamic_min,
                # ── Ban 'amr-unknown' hoàn toàn ──────────────────────────
                bad_words_ids=bad_words_ids,
                # ────────────────────────────────────────────────────────

                max_new_tokens=max_new_tokens,
                forced_bos_token_id=tokenizer.amr_bos_token_id,
                early_stopping=True,
            )

        for seq in outputs:
            amr_str = decode_to_amr(seq.tolist(), tokenizer)
            results.append(amr_str)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty print cho single sentence
# ─────────────────────────────────────────────────────────────────────────────

def pretty_print_result(sentence: str, amr_str: str, idx: int = 0):
    print(f"\n{'─'*60}")
    print(f"# ::id {idx}")
    print(f"# ::annotator diverse-beam-amr")
    print(f"# ::snt {sentence}")
    print(amr_str)
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AMRBART Inference với Diverse Beam Search"
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Đường dẫn đến checkpoint (local hoặc HuggingFace hub)"
    )
    parser.add_argument(
        "--sentence", default=None,
        help="Parse 1 câu trực tiếp từ command line"
    )
    parser.add_argument(
        "--input_file", default=None,
        help="File text, mỗi dòng 1 câu"
    )
    parser.add_argument(
        "--output_file", default="parsed_amr.txt",
        help="File output AMR"
    )
    parser.add_argument("--num_beams",       type=int,   default=10)
    parser.add_argument("--num_beam_groups", type=int,   default=5)
    parser.add_argument("--diversity_penalty", type=float, default=0.8)
    parser.add_argument("--max_new_tokens",  type=int,   default=400)
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading model from: {args.model_path}")
    print(f"Device: {args.device}")

    from transformers import BartForConditionalGeneration
    from model_interface.tokenization_bart import AMRBartTokenizer

    # Load tokenizer — giống hệt main.py
    tokenizer = AMRBartTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
    )

    model = BartForConditionalGeneration.from_pretrained(args.model_path)
    model.eval()

    # ── Collect sentences ───────────────────────────────────────────────────
    if args.sentence:
        sentences = [args.sentence]
    elif args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Cần --sentence hoặc --input_file")

    print(f"\nParsing {len(sentences)} câu với Diverse Beam Search:")
    print(f"  num_beams={args.num_beams} | num_beam_groups={args.num_beam_groups}")
    print(f"  diversity_penalty={args.diversity_penalty}\n")

    # ── Run inference ────────────────────────────────────────────────────────
    results = parse_sentences(
        model=model,
        tokenizer=tokenizer,
        sentences=sentences,
        num_beams=args.num_beams,
        num_beam_groups=args.num_beam_groups,
        diversity_penalty=args.diversity_penalty,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
    )

    # ── Output ───────────────────────────────────────────────────────────────
    if args.sentence:
        # Single sentence → pretty print
        pretty_print_result(args.sentence, results[0], idx=0)
    else:
        # Batch → write to file
        with open(args.output_file, "w", encoding="utf-8") as f:
            for i, (sent, amr) in enumerate(zip(sentences, results)):
                f.write(f"# ::id {i}\n")
                f.write(f"# ::annotator diverse-beam-amr\n")
                f.write(f"# ::snt {sent}\n")
                f.write(amr + "\n\n")
        print(f"Saved {len(results)} graphs → {args.output_file}")

        # Stats
        empty = sum(1 for r in results if 'amr-empty' in r)
        thin  = sum(
            1 for r in results
            if 'amr-unknown' in r and r.count(':') <= 2
        )
        print(f"\nStats:")
        print(f"  Total:          {len(results)}")
        print(f"  amr-empty:      {empty} ({100*empty/len(results):.1f}%)")
        print(f"  thin + unk:     {thin}  ({100*thin/len(results):.1f}%)")
        print(f"  Valid graphs:   {len(results)-empty} ({100*(len(results)-empty)/len(results):.1f}%)")


if __name__ == "__main__":
    main()
