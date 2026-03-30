import sys
import penman
import re
from io import StringIO
import smatch

def compute_smatch_reward(pred_amr_str: str, gold_amr_str: str) -> float:
    """
    Tính điểm Smatch F1 giữa Generated Graph và Gold Graph.
    Theo GRPO paper, kết quả được tính bình phương (F1 ** 2) để phạt nặng lỗi sai lớn.
    """
    try:
        pred_f = StringIO(pred_amr_str)
        gold_f = StringIO(gold_amr_str)
        score = None
        for score in smatch.score_amr_pairs(pred_f, gold_f):
            pass
        f1 = score[2] if score else 0.0
        return f1 ** 2
    except Exception as e:
        return 0.0

def validate_amr_structure(amr_str: str):
    """
    Đánh giá Cấu trúc đồ thị sinh ra để tính 3 reward tín hiệu:
    - Parsability (Hợp lệ chuỗi Penman)
    - Frame-Argument (ARG0-ARG5 hợp lệ đối với Node đóng vai trò là Predicate)
    - AND-OR logic (Toán tử op1, op2.. phải liên tiếp)
    """
    try:
        # Nếu Decode được bằng Penman -> Reward Parsability = 1
        graph = penman.decode(amr_str + " ") # pad to avoid parse error at end
        parsability = 1.0
    except Exception:
        # Nếu chuỗi mất ngoặc hoặc hỏng hoàn toàn thì trả 0 cho toàn bộ
        return 0.0, 0.0, 0.0
        
    triples = graph.triples
    
    total_frames = 0
    valid_frames = 0
    
    total_and_or = 0
    valid_and_or = 0
    
    # Thu thập Edges của từng biến Node
    node_edges = {}
    node_concepts = {}
    for src, rel, tgt in triples:
        if rel == ':instance':
            node_concepts[src] = tgt.lower() if isinstance(tgt, str) else tgt
        else:
            if src not in node_edges:
                node_edges[src] = []
            node_edges[src].append(rel)
            
    # Đánh giá luật
    for src, concept in node_concepts.items():
        edges = node_edges.get(src, [])
        
        # 1. Khung Argument Logic (Frame-Argument Correctness)
        if isinstance(concept, str) and re.match(r'.+-\d\d$', concept):
            total_frames += 1
            # Lọc các tham số bắt đầu bằng :ARG
            args = [int(r[4:]) for r in edges if r.startswith(':ARG') and r[4:].isdigit()]
            # Giả định: Hầu hết PropBank frames chỉ hỗ trợ :ARG0 đến :ARG5
            if any(a > 5 for a in args):
                pass
            else:
                valid_frames += 1
                
        # 2. AND-OR Logic
        if concept in ('and', 'or'):
            total_and_or += 1
            ops = [int(r[3:]) for r in edges if r.startswith(':op') and r[3:].isdigit()]
            if len(ops) == 0:
                pass
            elif len(ops) == 1 and ops[0] == 2:
                # Trường hợp đặc biệt ở AMR 3.0: AND/OR node ở root của mệnh đề tiếp nối chỉ có nhánh :op2
                valid_and_or += 1
            else:
                ops.sort()
                # Toán tử phải liên tiếp 1, 2, 3..
                if ops == list(range(1, len(ops) + 1)):
                    valid_and_or += 1
                    
    frame_reward = (valid_frames / total_frames) if total_frames > 0 else 1.0
    and_or_reward = (valid_and_or / total_and_or) if total_and_or > 0 else 1.0
    
    return parsability, frame_reward, and_or_reward

def compute_grpo_rewards(pred_strs, gold_strs):
    """
    Tính mức thưởng GRPO cho một batch kết quả. 
    Lưu ý pred_strs và gold_strs là list string đã decode từ token ids.
    """
    rewards = []
    details = []
    for p, g in zip(pred_strs, gold_strs):
        r_smatch = compute_smatch_reward(p, g)
        r_parse, r_frame, r_andor = validate_amr_structure(p)
        
        # Tổng hợp 4 hàm toán học chia trung bình (Trọng số đều)
        total = (r_smatch + r_parse + r_frame + r_andor) / 4.0
        rewards.append(total)
        
        details.append({
            'smatch2': r_smatch, 
            'parse': r_parse, 
            'frame': r_frame, 
            'andor': r_andor
        })
        
    return rewards, details
