import torch
import torch.nn as nn
from transformers import Seq2SeqTrainer
from rl_rewards import compute_grpo_rewards

class GRPOTrainer(Seq2SeqTrainer):
    """
    Kế thừa HF Seq2SeqTrainer để tính toán objective loss GRPO.
    """
    def __init__(self, *args, do_rl=False, rl_group_size=4, rl_alpha=0.5, rl_warmup_epochs=5, custom_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_rl = do_rl
        self.rl_group_size = rl_group_size
        # rl_alpha: Trọng số của Supervised Fine-Tuning. VD: 0.5 nghĩa là L = 0.5*CE + 0.5*RL
        self.rl_alpha = rl_alpha
        # rl_warmup_epochs: Số epoch chỉ train SFT thuần trước khi bật RL
        self.rl_warmup_epochs = rl_warmup_epochs
        self.custom_tokenizer = custom_tokenizer
        self._rl_activated_logged = False
        # Lưu lại num_beams gốc để khôi phục khi GRPO bắt đầu
        self._original_num_beams = None
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # ---- 1. Tính cấu phần Cross Entropy Loss (Supervised SFT) ----
        # Extract labels before calling super (because super().compute_loss pops it)
        labels = inputs.get("labels").clone()
        # Gọi thủ tục loss của bản thân base Seq2SeqTrainer để lấy CE_loss đã được fix loss dilution
        ce_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # Cờ an toàn, nếu config không bật RL thì chỉ trả về SFT loss như bình thường
        if not self.do_rl:
            return (ce_loss, outputs) if return_outputs else ce_loss

        # SFT Warm-up: chỉ train CE thuần trong N epoch đầu
        current_epoch = self.state.epoch if self.state else 0
        if current_epoch < self.rl_warmup_epochs:
            return (ce_loss, outputs) if return_outputs else ce_loss
        
        # Log khi RL bắt đầu kích hoạt (chỉ 1 lần)
        if not self._rl_activated_logged:
            print(f"\n🚀 GRPO RL activated at epoch {current_epoch:.1f} (warm-up = {self.rl_warmup_epochs} epochs)")
            self._rl_activated_logged = True

        # ---- 2. Tính cấu phần Reinforcement Learning Loss (GRPO) ----
        input_ids = inputs["input_ids"]
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Decode labels (bỏ qua id = -100 là padding trong seq2seq) để lấy Text Gold đo Smatch
        gold_ids = labels.clone()
        gold_ids[gold_ids == -100] = self.custom_tokenizer.pad_token_id
        gold_strs = self.custom_tokenizer.batch_decode(gold_ids, skip_special_tokens=True)
        
        # BƯỚC 2A. Sinh dữ liệu nhóm (Group Sampling)
        # Sử dụng torch.no_grad() để giải phóng biểu đồ tính toán (graph memory) khi text generation
        with torch.no_grad():
            sample_outputs = model.generate(
                input_ids=input_ids,
                max_length=self.args.generation_max_length if self.args.generation_max_length else 1024,
                num_return_sequences=self.rl_group_size,  # Generate G sequences per input
                num_beams=1, # BẮT BUỘC: Ép bằng 1 để dùng chuẩn Multinomial Sampling thay vì Beam Sampling (gây lỗi nan)
                do_sample=True,
                temperature=0.8, # Temperature khuyến nghị từ paper
                pad_token_id=self.custom_tokenizer.pad_token_id,
                eos_token_id=self.custom_tokenizer.eos_token_id,
                decoder_start_token_id=self.custom_tokenizer.amr_bos_token_id,
                 # Có thể tắt forced_bos_token_id nếu config gây lỗi
            )
            # Shape của sample_outputs: [batch_size * G, seq_len]
        
        # BƯỚC 2B. Đo đạc Reward Matrix 
        pred_strs = self.custom_tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
        
        # Duplicate mảng Gold Text G lần để mapping 1-1 với mảng Prediction Text
        gold_strs_repeated = []
        for g_str in gold_strs:
            gold_strs_repeated.extend([g_str] * self.rl_group_size)
            
        # Nạp vào module rewards của ta
        rewards, details = compute_grpo_rewards(pred_strs, gold_strs_repeated)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        # BƯỚC 2C. Chuyển đổi tính chuẩng lợi thế (Group Advantage Normalization)
        # Đây là Cốt lõi của GRPO, thay thế cho mạng Critic!!
        rewards_reshaped = rewards_tensor.view(batch_size, self.rl_group_size)
        mean_rewards = rewards_reshaped.mean(dim=1, keepdim=True)
        std_rewards = rewards_reshaped.std(dim=1, keepdim=True)
        
        # Normalization (A_i = (R_i - mean) / (std + 1e-8))
        advantages = (rewards_reshaped - mean_rewards) / (std_rewards + 1e-8)
        advantages = advantages.view(-1) # Kéo phẳng về lại shape [batch_size * G]
        
        # BƯỚC 2D. Lan truyền ngược Policy (Policy Gradient Calculation)
        # Chúng ta cần tính Log_prob của các output vừa sinh, LẦN NÀY LÀ CÓ LƯU GRADIENT TRÊN MODEL!
        # Do đó phải copy input gấp G lần.
        input_ids_repeated = input_ids.repeat_interleave(self.rl_group_size, dim=0)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask_repeated = attention_mask.repeat_interleave(self.rl_group_size, dim=0)
        else:
            attention_mask_repeated = None

        # Mô phỏng Auto-regressive offset (dịch phải 1 step để feed vào Decoder)
        decoder_input_ids = torch.cat([
            torch.full((sample_outputs.size(0), 1), self.custom_tokenizer.amr_bos_token_id, device=device),
            sample_outputs[:, :-1]
        ], dim=1)
        
        rl_outputs = model(
            input_ids=input_ids_repeated,
            attention_mask=attention_mask_repeated,
            decoder_input_ids=decoder_input_ids
        )
        logits = rl_outputs.logits # [batch_size * G, seq_len, vocab_size]
        
        # Tính Probability phân phối token và Extract log xác suất đúng của token thực tế được pick
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, sample_outputs.unsqueeze(-1)).squeeze(-1) # shape: [B*G, seq_len]
        
        # Không tính loss cho phần Padding token
        padding_mask = (sample_outputs != self.custom_tokenizer.pad_token_id).float()
        seq_log_probs = (token_log_probs * padding_mask).sum(dim=1) # Gom toàn thể prob của chuỗi, shape [B*G]
        
        # Ghi nhận policy gradient equation (dấu trừ vì là Loss cần Gradient Descent tối thiểu hóa)
        rl_loss = -(advantages * seq_log_probs).mean()
        
        # --- 3. Objective tổng hợp ---
        # Ngăn model thoái hóa cấu trúc tiếng Việt (SFT) để mix cùng F1-score optimization (RL)
        total_loss = self.rl_alpha * ce_loss + (1.0 - self.rl_alpha) * rl_loss
        
        # (Tuỳ chọn) Đẩy loss gốc lên logs của HF
        if hasattr(self, "log"):
            self.log({"ce_loss": ce_loss.item(), "rl_loss": rl_loss.item(), "reward_mean": rewards_tensor.mean().item()})

        return (total_loss, outputs) if return_outputs else total_loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **gen_kwargs):
        """
        Override evaluate để tự động tăng tốc eval trong SFT warm-up:
          - num_beams=1  (Greedy thay vì Beam Search)
          - generation_max_length=256 (thay vì 1024)
          - Chỉ eval 200 mẫu đại diện thay vì toàn bộ val set
        Khi GRPO bắt đầu → khôi phục toàn bộ cấu hình gốc để eval chính thức.
        """
        current_epoch = self.state.epoch if self.state else 0
        in_warmup = self.do_rl and (current_epoch < self.rl_warmup_epochs)

        # Lưu cấu hình gốc nếu chưa lưu
        if self._original_num_beams is None:
            self._original_num_beams = self.args.generation_num_beams
        
        # Lưu max_length gốc
        if not hasattr(self, '_original_max_length'):
            self._original_max_length = self.args.generation_max_length or 1024

        if in_warmup:
            # === CHẾ ĐỘ NHANH: SFT Warmup ===
            self.args.generation_num_beams = 1
            self.args.generation_max_length = 256   # Giảm max token sinh ra

            # Chỉ lấy 200 mẫu đại diện để eval nhanh
            _eval_dataset = eval_dataset or self.eval_dataset
            if _eval_dataset is not None and len(_eval_dataset) > 200:
                import random
                indices = random.sample(range(len(_eval_dataset)), 200)
                _eval_dataset = _eval_dataset.select(indices)
            
            n_samples = len(_eval_dataset) if _eval_dataset else "N/A"
            print(f"\n[Eval] SFT warm-up epoch {current_epoch:.1f}: "
                  f"Greedy(beams=1) + max_len=256 + {n_samples} mẫu → ~2-3 phút")

            result = super().evaluate(
                _eval_dataset, ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix, **gen_kwargs
            )
        else:
            # === CHẾ ĐỘ ĐẦY ĐỦ: GRPO Phase ===
            self.args.generation_num_beams = self._original_num_beams
            self.args.generation_max_length = self._original_max_length
            print(f"\n[Eval] GRPO phase: Full eval với beams={self._original_num_beams}, max_len={self._original_max_length}")
            
            result = super().evaluate(
                eval_dataset, ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix, **gen_kwargs
            )

        return result

