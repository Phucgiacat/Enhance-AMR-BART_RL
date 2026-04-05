#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# inference-cascade.sh
# Cascade / Ensemble Inference với 2 AMRBART checkpoints
#
# USAGE:
#   bash inference-cascade.sh <MODEL_A> <MODEL_B> [OPTIONS]
#
# EXAMPLES:
#   # Cascade (A trước, fallback B nếu thin):
#   bash inference-cascade.sh /ckpt-852 /ckpt-9256 \
#       --sentence "Hãy cùng lắng_nghe bài hát này"
#
#   # Ensemble (cả 2 → chọn tốt hơn):
#   bash inference-cascade.sh /ckpt-852 /ckpt-9256 \
#       --input_file input.txt --mode ensemble
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
NUM_BEAMS=10
NUM_BEAM_GROUPS=5
DIVERSITY=0.8
MAX_TOKENS=400
BATCH_SIZE=2
MODE="cascade"

# ── Parse args ────────────────────────────────────────────────────────────────
MODEL_A="${1:?Usage: $0 <MODEL_A> <MODEL_B> [--sentence X | --input_file F] [--mode cascade|ensemble]}"
MODEL_B="${2:?Usage: $0 <MODEL_A> <MODEL_B> [--sentence X | --input_file F] [--mode cascade|ensemble]}"
shift 2

SENTENCE=""
INPUT_FILE=""
OUTPUT_FILE="output_cascade.txt"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sentence)         SENTENCE="$2";    shift 2 ;;
        --input_file)       INPUT_FILE="$2";  shift 2 ;;
        --output_file)      OUTPUT_FILE="$2"; shift 2 ;;
        --mode)             MODE="$2";        shift 2 ;;
        --num_beams)        NUM_BEAMS="$2";   shift 2 ;;
        --num_beam_groups)  NUM_BEAM_GROUPS="$2"; shift 2 ;;
        --diversity_penalty) DIVERSITY="$2";  shift 2 ;;
        --max_new_tokens)   MAX_TOKENS="$2";  shift 2 ;;
        --batch_size)       BATCH_SIZE="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  AMRBART Cascade/Ensemble Inference                         ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Mode:      %-48s║\n" "$MODE"
printf "║  Model A:   %-48s║\n" "$(basename $MODEL_A)"
printf "║  Model B:   %-48s║\n" "$(basename $MODEL_B)"
printf "║  Beams:     %-48s║\n" "$NUM_BEAMS (groups: $NUM_BEAM_GROUPS)"
printf "║  Diversity: %-48s║\n" "$DIVERSITY"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Build python args ─────────────────────────────────────────────────────────
PY_ARGS=(
    "$SCRIPT_DIR/inference_cascade.py"
    --model_a "$MODEL_A"
    --model_b "$MODEL_B"
    --mode "$MODE"
    --num_beams "$NUM_BEAMS"
    --num_beam_groups "$NUM_BEAM_GROUPS"
    --diversity_penalty "$DIVERSITY"
    --max_new_tokens "$MAX_TOKENS"
    --batch_size "$BATCH_SIZE"
    --output_file "$OUTPUT_FILE"
)

if [[ -n "$SENTENCE" ]]; then
    PY_ARGS+=(--sentence "$SENTENCE")
elif [[ -n "$INPUT_FILE" ]]; then
    PY_ARGS+=(--input_file "$INPUT_FILE")
else
    echo "❌  Cần --sentence hoặc --input_file"
    exit 1
fi

python "${PY_ARGS[@]}"
STATUS=$?

if [[ $STATUS -eq 0 ]]; then
    echo ""
    echo "✅ Inference hoàn tất!"
else
    echo ""
    echo "❌ Inference thất bại (exit code: $STATUS)"
    exit $STATUS
fi
