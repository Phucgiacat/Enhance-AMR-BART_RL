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
    batch_size: int = 4,
    device: str = "cuda",
) -> List[str]:
    """
    Parse danh sách câu → AMR graphs dùng Diverse Beam Search.

    Args:
        num_beams:         Tổng số beams (phải chia hết cho num_beam_groups)
        num_beam_groups:   Số nhóm beams — mỗi nhóm explore hướng khác
        diversity_penalty: Càng cao → các nhóm càng đa dạng (0.5-1.5)
        max_new_tokens:    Độ dài tối đa output
        batch_size:        Số câu xử lý một lúc (giảm nếu OOM)
    """
    assert num_beams % num_beam_groups == 0, (
        f"num_beams ({num_beams}) phải chia hết cho num_beam_groups ({num_beam_groups})"
    )

    model.eval()
    model.to(device)
    results = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="Parsing"):
        batch_sents = sentences[i: i + batch_size]

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

    from transformers import MBartForConditionalGeneration
    from spring_amr.tokenization_bart import AMRBartTokenizer

    # Load tokenizer theo đúng pattern AMRBART (giống main.py)
    base_model = "facebook/bart-large"
    tokenizer = AMRBartTokenizer.from_pretrained(
        base_model,
        collapse_name_ops=False,
        use_pointer_tokens=True,
        raw_graph=False,
    )

    model = MBartForConditionalGeneration.from_pretrained(args.model_path)
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
