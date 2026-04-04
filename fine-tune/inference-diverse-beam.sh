#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# inference-diverse-beam.sh
# Inference script for AMRBART Vietnamese AMR Parser
# Uses Diverse Beam Search for better quality than standard beam search
#
# Usage:
#   # Parse from file:
#   bash inference-diverse-beam.sh /path/to/checkpoint sentences.txt
#
#   # Parse single sentence:
#   bash inference-diverse-beam.sh /path/to/checkpoint --sentence "câu cần parse"
#
#   # Use best checkpoint automatically:
#   bash inference-diverse-beam.sh best sentences.txt
# ─────────────────────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES=0
RootDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# ── Checkpoint ────────────────────────────────────────────────────────────────
CHECKPOINT=$1

# Nếu truyền 'best', tự động tìm best checkpoint từ trainer_state.json
if [ "$CHECKPOINT" = "best" ]; then
    OUTPUT_DIR="/content/drive/MyDrive/output_grpo/AMRBART-GRPO_v2"
    TRAINER_STATE="$OUTPUT_DIR/trainer_state.json"
    if [ -f "$TRAINER_STATE" ]; then
        CHECKPOINT=$(python3 -c "
import json
with open('$TRAINER_STATE') as f:
    state = json.load(f)
print(state['best_model_checkpoint'])
")
        echo "Auto-detected best checkpoint: $CHECKPOINT"
    else
        echo "ERROR: trainer_state.json not found at $OUTPUT_DIR"
        exit 1
    fi
fi

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: bash inference-diverse-beam.sh <checkpoint_path|best> <input_file|--sentence 'text'>"
    exit 1
fi

# ── Input mode ────────────────────────────────────────────────────────────────
if [ "$2" = "--sentence" ]; then
    # Single sentence mode
    SENTENCE="$3"
    if [ -z "$SENTENCE" ]; then
        echo "ERROR: --sentence requires text argument"
        exit 1
    fi
    INPUT_ARGS="--sentence \"$SENTENCE\""
else
    # File mode
    INPUT_FILE=${2:-"sentences.txt"}
    OUTPUT_FILE=${3:-"${RootDir}/outputs/parsed_amr_diverse_beam.txt"}
    mkdir -p "$(dirname $OUTPUT_FILE)"
    INPUT_ARGS="--input_file $INPUT_FILE --output_file $OUTPUT_FILE"
fi

# ── Diverse Beam Search Parameters ───────────────────────────────────────────
NUM_BEAMS=10            # Tổng số beams (phải chia hết cho num_beam_groups)
NUM_BEAM_GROUPS=5       # Số nhóm beams — mỗi nhóm explore hướng khác nhau
DIVERSITY_PENALTY=0.8   # Độ đa dạng giữa các nhóm (0.5=nhẹ, 1.5=mạnh)
MAX_NEW_TOKENS=400      # Độ dài tối đa output (tokens)
BATCH_SIZE=4            # Giảm xuống 2 nếu OOM trên Colab T4

# ── Run ──────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " AMRBART Inference — Diverse Beam Search"
echo "============================================================"
echo " Checkpoint:       $CHECKPOINT"
echo " num_beams:        $NUM_BEAMS"
echo " num_beam_groups:  $NUM_BEAM_GROUPS"
echo " diversity_penalty:$DIVERSITY_PENALTY"
echo " max_new_tokens:   $MAX_NEW_TOKENS"
echo " batch_size:       $BATCH_SIZE"
echo "============================================================"

python -u $RootDir/inference_diverse_beam.py \
    --model_path $CHECKPOINT \
    $INPUT_ARGS \
    --num_beams $NUM_BEAMS \
    --num_beam_groups $NUM_BEAM_GROUPS \
    --diversity_penalty $DIVERSITY_PENALTY \
    --max_new_tokens $MAX_NEW_TOKENS \
    --batch_size $BATCH_SIZE \
    --device cuda

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Inference hoàn tất!"
    if [ -n "$OUTPUT_FILE" ]; then
        echo "   Output: $OUTPUT_FILE"
    fi
else
    echo "❌ Inference thất bại (exit code: $EXIT_CODE)"
fi
exit $EXIT_CODE
