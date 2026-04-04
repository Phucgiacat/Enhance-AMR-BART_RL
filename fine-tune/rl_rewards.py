import sys
import penman
import re
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
import smatch

DUMMY_AMR = '(z / amr-empty)'

# ─────────────────────────────────────────────────────────────────────────────
# 1. SMATCH REWARD
# ─────────────────────────────────────────────────────────────────────────────

def compute_smatch_reward(pred_amr_str: str, gold_amr_str: str) -> float:
    """
    Tính điểm Smatch F1 giữa Generated Graph và Gold Graph.
    Theo GRPO paper, kết quả được tính bình phương (F1²) để phạt mạnh sai lớn.
    """
    try:
        pred_f = StringIO(pred_amr_str)
        gold_f = StringIO(gold_amr_str)
        score = None
        for score in smatch.score_amr_pairs(pred_f, gold_f):
            pass
        f1 = score[2] if score else 0.0
        return f1 ** 2
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. STRUCTURAL REWARDS (Parsability, Frame, AND-OR)
# ─────────────────────────────────────────────────────────────────────────────

def validate_amr_structure(amr_str: str):
    """
    Đánh giá 3 tín hiệu cấu trúc:
    - Parsability: Decode được bằng Penman không?
    - Frame-Argument: Node có ARGx có hợp lệ không (ARG0-ARG5)?
    - AND-OR logic: Toán tử opN có liên tiếp không?

    Đặc điểm Vietnamese AMR:
    - Không có PropBank sense-ID (-01, -02), chỉ có -91
    - Predicate được nhận diện bằng sự hiện diện của ARGx edge
    """
    try:
        if amr_str.strip() == DUMMY_AMR:
            return 0.0, 0.0, 0.0
        graph = penman.decode(amr_str + " ")
        parsability = 1.0
    except Exception:
        return 0.0, 0.0, 0.0

    triples = graph.triples
    total_frames = 0
    valid_frames = 0
    total_and_or = 0
    valid_and_or = 0

    node_edges = {}
    node_concepts = {}
    for src, rel, tgt in triples:
        if rel == ':instance':
            node_concepts[src] = tgt.lower() if isinstance(tgt, str) else tgt
        else:
            node_edges.setdefault(src, []).append(rel)

    for src, concept in node_concepts.items():
        edges = node_edges.get(src, [])

        # Frame-Argument: bất kỳ node có ARGx đều coi là predicate
        args = [int(r[4:]) for r in edges if r.startswith(':ARG') and r[4:].isdigit()]
        if args:
            total_frames += 1
            # ARG > 5 thường là lỗi (trong cả English và Vietnamese AMR)
            if not any(a > 5 for a in args):
                valid_frames += 1

        # AND-OR logic
        if concept in ('and', 'or'):
            total_and_or += 1
            ops = [int(r[3:]) for r in edges if r.startswith(':op') and r[3:].isdigit()]
            if len(ops) == 1 and ops[0] == 2:
                # Trường hợp đặc biệt AMR 3.0: sub-conjunction chỉ có :op2
                valid_and_or += 1
            elif len(ops) > 1:
                ops_sorted = sorted(ops)
                if ops_sorted == list(range(1, len(ops_sorted) + 1)):
                    valid_and_or += 1

    frame_reward  = (valid_frames  / total_frames)  if total_frames  > 0 else 1.0
    and_or_reward = (valid_and_or  / total_and_or)  if total_and_or  > 0 else 1.0

    return parsability, frame_reward, and_or_reward


# ─────────────────────────────────────────────────────────────────────────────
# 3. LENGTH BALANCE REWARD (thay thế coverage 1 chiều)
# ─────────────────────────────────────────────────────────────────────────────

def compute_length_balance_reward(pred_amr_str: str, gold_amr_str: str) -> float:
    """
    Phạt CANE HAI CHIỀU:
    - Graph quá ngắn (mỏng): thiếu edge → reward thấp tuyến tính
    - Graph quá dài (babble): thừa edge gấp đôi gold → reward về 0

    Công thức:
        ratio = num_pred_edges / max(num_gold_edges, 1)
        if ratio <= 1.0: reward = ratio          [quá ngắn]
        else:            reward = max(0, 2-ratio) [quá dài]

    Đặc điểm: reward = 1.0 khi pred có số edge bằng gold (lý tưởng).
    """
    try:
        if pred_amr_str.strip() == DUMMY_AMR:
            return 0.0
        if gold_amr_str.strip() == DUMMY_AMR:
            return 1.0  # Không có tiêu chuẩn → không phạt

        pred_graph = penman.decode(pred_amr_str + " ")
        gold_graph = penman.decode(gold_amr_str + " ")

        pred_edges = [t for t in pred_graph.triples if t[1] != ':instance']
        gold_edges = [t for t in gold_graph.triples if t[1] != ':instance']

        num_pred = len(pred_edges)
        num_gold = max(len(gold_edges), 1)
        ratio = num_pred / num_gold

        if ratio <= 1.0:
            return ratio           # quá ngắn → tuyến tính
        else:
            return max(0.0, 2.0 - ratio)  # quá dài → về 0 khi gấp đôi
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONCEPT QUALITY REWARD
# ─────────────────────────────────────────────────────────────────────────────

# Các concept AMR chuẩn không bị kiểm tra
_VALID_AMR_SPECIALS = {
    'amr-unknown', 'amr-empty', 'and', 'or', 'multi-sentence',
    'have-org-role-91', 'have-rel-role-91', 'have-quant-91',
    'have-degree-91', 'have-condition-91', 'include-91',
    'cause-01', 'obligate-91', 'possible-91', 'recommend-91',
    'name', 'date-entity', 'ordinal-entity', 'temporal-quantity',
    'mass-quantity', 'distance-quantity', 'monetary-quantity',
    'rate-entity-91', 'percentage-entity', 'url-entity',
    'contrast', 'concession', 'condition', 'expressive', 'imperative',
}

# Regex phát hiện concept bị vỡ UTF-8 hoặc rác
_GARBAGE_PATTERN = re.compile(
    r'[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]'  # replacement char, control chars
    r'|</?lit>'                              # literal tag bị lọt ra
    r'|\[.*?\]'                              # bracket artifact
)

def compute_concept_quality_reward(amr_str: str) -> float:
    """
    Đánh giá chất lượng concept nodes:
    - Phạt concept chứa ký tự UTF-8 vỡ (\\ufffd) — do temperature sampling
    - Phạt concept quá dài (> 40 ký tự thường là garbage concatenation)
    - Phạt concept chứa dấu ngoặc lọt vào (parsing artifact)

    Phù hợp với Vietnamese AMR: concept là từ/cụm từ tiếng Việt
    hoặc standard AMR special nodes (amr-unknown, name, date-entity...).
    Không yêu cầu sense-ID vì Vietnamese AMR không có PropBank sense.
    """
    try:
        if amr_str.strip() == DUMMY_AMR:
            return 0.0

        graph = penman.decode(amr_str + " ")
        concepts = [
            str(t[2]) for t in graph.triples
            if t[1] == ':instance' and t[2] is not None
        ]

        if not concepts:
            return 0.5  # Không có concept → không đủ thông tin

        bad_count = 0
        for c in concepts:
            c_lower = c.lower().strip('"')
            if c_lower in _VALID_AMR_SPECIALS:
                continue
            # Kiểm tra garbage
            if _GARBAGE_PATTERN.search(c):
                bad_count += 1
            elif len(c) > 40:
                bad_count += 1
            # Concept chứa ngoặc hoặc dấu 2 chấm lạ (artifact)
            elif re.search(r'[():]', c):
                bad_count += 1

        quality = 1.0 - (bad_count / len(concepts))
        return max(0.0, quality)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. REENTRANCY REWARD
# ─────────────────────────────────────────────────────────────────────────────

def _count_reentrancies(graph) -> int:
    """
    Đếm số node được tham chiếu > 1 lần trong graph (co-reference).
    Một node là reentrant nếu nó xuất hiện làm TGT nhiều hơn 1 lần
    trong các non-instance triples.
    """
    tgt_counts = {}
    for src, rel, tgt in graph.triples:
        if rel != ':instance' and isinstance(tgt, str) and tgt.startswith('z'):
            tgt_counts[tgt] = tgt_counts.get(tgt, 0) + 1
    return sum(1 for cnt in tgt_counts.values() if cnt > 1)

def compute_reentrancy_reward(pred_amr_str: str, gold_amr_str: str) -> float:
    """
    So sánh số reentrant nodes giữa pred và gold.
    Phạt cả trường hợp:
    - Pred thiếu reentrancy khi gold có nhiều (model quên co-reference)
    - Pred thừa reentrancy khi gold ít (model hallucinate co-reference)

    Công thức:
        ratio = pred_reentrant / max(gold_reentrant, 1)
        reward = max(0, 1 - |ratio - 1|)   [đỉnh tại ratio=1.0]

    Ghi chú: Vietnamese AMR ~18k data — reentrancy ít phổ biến hơn English.
    Weight thấp (5%) để tránh over-penalize.
    """
    try:
        if pred_amr_str.strip() == DUMMY_AMR:
            return 0.0
        if gold_amr_str.strip() == DUMMY_AMR:
            return 1.0

        pred_graph = penman.decode(pred_amr_str + " ")
        gold_graph = penman.decode(gold_amr_str + " ")

        pred_r = _count_reentrancies(pred_graph)
        gold_r = _count_reentrancies(gold_graph)

        if gold_r == 0:
            # Gold không có reentrancy → reward 1.0 nếu pred cũng không có
            # nhưng phạt nhẹ nếu pred tự thêm vào (hallucinate)
            return 1.0 if pred_r == 0 else max(0.0, 1.0 - 0.2 * pred_r)

        ratio = pred_r / gold_r
        return max(0.0, 1.0 - abs(ratio - 1.0))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODE CONSISTENCY REWARD
# ─────────────────────────────────────────────────────────────────────────────

_VALID_MODES = {'imperative', 'expressive', 'interrogative'}

def compute_mode_reward(pred_amr_str: str, gold_amr_str: str) -> float:
    """
    Phạt trường hợp model lười: sinh ':mode amr-unknown' khi graph quá mỏng.
    Ví dụ: câu 'Hãy...' nên là ':mode imperative', không phải ':mode amr-unknown'.

    Logic:
    - Nếu pred không có :mode → không phạt (1.0)
    - Nếu pred có ':mode amr-unknown' VÀ graph rất mỏng (ít hơn 3 edges) → phạt
    - Nếu gold có :mode cụ thể mà pred dùng amr-unknown → phạt
    - Nếu pred dùng :mode hợp lệ (imperative/expressive/interrogative) → reward cao
    """
    try:
        if pred_amr_str.strip() == DUMMY_AMR:
            return 0.0

        pred_graph = penman.decode(pred_amr_str + " ")
        pred_triples = pred_graph.triples

        # Tìm :mode trong pred
        pred_mode = None
        for src, rel, tgt in pred_triples:
            if rel == ':mode':
                pred_mode = str(tgt).lower() if isinstance(tgt, str) else str(tgt)
                break

        # Không có :mode → OK
        if pred_mode is None:
            return 1.0

        # Dùng mode đúng chuẩn → reward cao
        if pred_mode in _VALID_MODES:
            return 1.0

        # Dùng amr-unknown: phạt nếu graph mỏng (< 3 non-instance edges)
        if pred_mode == 'amr-unknown':
            non_inst = [t for t in pred_triples if t[1] != ':instance']
            if len(non_inst) <= 2:
                # Graph rất mỏng + amr-unknown = model đang lười
                return 0.3

            # Graph đầy đủ nhưng không chắc mode → chấp nhận
            return 0.7

        return 0.8  # mode khác lạ nhưng không phải amr-unknown
    except Exception:
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# REWARD AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def _compute_single_reward(args):
    p, g = args
    r_smatch          = compute_smatch_reward(p, g)
    r_parse, r_frame, r_andor = validate_amr_structure(p)
    r_length_balance  = compute_length_balance_reward(p, g)
    r_concept_quality = compute_concept_quality_reward(p)
    r_reentrancy      = compute_reentrancy_reward(p, g)
    r_mode            = compute_mode_reward(p, g)

    # Trọng số thiết kế cho Vietnamese AMR (18k data, no sense-ID):
    #   smatch (35%)         — signal chính
    #   parse  (10%)         — giảm xuống, nhường chỗ cho completeness
    #   frame  (15%)         — ARG structure hợp lệ
    #   andor  ( 5%)         — giảm xuống (ít cặp and/or trong data)
    #   length_balance (20%) — TĂNG: ép model sinh graph đầy đủ hơn
    #   concept_quality (10%)— phạt garbled UTF-8 & concept rác
    #   reentrancy (3%)      — co-reference match
    #   mode (2%)            — phạt :mode amr-unknown khi graph mỏng
    total = (0.35 * r_smatch
           + 0.10 * r_parse
           + 0.15 * r_frame
           + 0.05 * r_andor
           + 0.20 * r_length_balance
           + 0.10 * r_concept_quality
           + 0.03 * r_reentrancy
           + 0.02 * r_mode)

    detail = {
        'smatch2':          r_smatch,
        'parse':            r_parse,
        'frame':            r_frame,
        'andor':            r_andor,
        'length_balance':   r_length_balance,
        'concept_quality':  r_concept_quality,
        'reentrancy':       r_reentrancy,
        'mode':             r_mode,
    }
    return total, detail


def compute_grpo_rewards(pred_strs, gold_strs):
    """
    Tính mức thưởng GRPO cho toàn bộ batch, chạy song song.
    """
    pairs = list(zip(pred_strs, gold_strs))
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_compute_single_reward, pairs))
    rewards = [r for r, _ in results]
    details = [d for _, d in results]
    return rewards, details
