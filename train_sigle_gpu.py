import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from peft import get_peft_model, LoraConfig
from utils.set_seed import set_seed
from utils.set_logger import set_logger
from utils.functions import compute_acc_text, compute_metrics_from_stats, compute_metrics_text_binary_accumulate
from src.dataset import AudioDataset, collate_fn_qwen2audio
from transformers.modeling_outputs import BaseModelOutput
import gc


# ==========================
# 文本标签解析
# ==========================
def map_label_from_text(text: str):
    t = (text or "").strip()
    if ("抑郁" in t) or ("抑" in t):
        return "抑郁"
    if ("非抑郁" in t) or ("正常" in t) or ("非抑" in t):
        return "非抑郁"
    return t


# ==========================
# GPU 内存监控
# ==========================
def print_gpu_memory(desc=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{desc} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")


# ==========================
# Adapter 定义
# ==========================
class DepAdapter(nn.Module):
    def __init__(self, audio_dim, adapter_dim=512, dropout=0.1):
        super().__init__()
        self.down_proj = nn.Linear(audio_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, audio_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(audio_dim)
        
    def forward(self, audio_features):
        residual = audio_features
        x = self.down_proj(audio_features)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


def create_modified_qwen2audio_encoder(original_encoder, adapter_config, adapter_ckpt=None, freeze_adapter=True):
    """为 Qwen2Audio Encoder 添加 Adapter，并可加载训练好的权重"""
    audio_dim = original_encoder.config.d_model
    adapter = DepAdapter(
        audio_dim=audio_dim,
        adapter_dim=adapter_config.get("adapter_dim", 512),
        dropout=adapter_config.get("dropout", 0.1)
    )

    # ====== 加载预训练好的 Adapter 权重 ======
    if adapter_ckpt is not None and os.path.exists(adapter_ckpt):
        state = torch.load(adapter_ckpt, map_location="cpu")
        adapter.load_state_dict(state)
        print(f"成功加载训练好的 DepAdapter 权重: {adapter_ckpt}")
    else:
        print(f"未加载 DepAdapter 权重（路径无效或未提供）: {adapter_ckpt}")

    # 是否冻结 Adapter
    if freeze_adapter:
        for p in adapter.parameters():
            p.requires_grad = False
        print("已冻结 DepAdapter 参数（不再更新）")
    else:
        print("DepAdapter 参数将被训练！")

    # ====== 替换 Encoder 的 forward ======
    original_forward = original_encoder.forward

    def new_forward(
        self,
        input_features,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if return_dict is None:
            return_dict = self.config.use_return_dict

        outputs = original_forward(
            input_features=input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        audio_features = outputs.last_hidden_state
        with torch.no_grad() if freeze_adapter else torch.enable_grad():
            adapted_audio_features = adapter(audio_features)

        return BaseModelOutput(
            last_hidden_state=adapted_audio_features,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    # 绑定新的 forward
    original_encoder.forward = new_forward.__get__(original_encoder, type(original_encoder))
    original_encoder.audio_adapter = adapter
    return original_encoder


# ==========================
# 单卡训练
# ==========================
def train_single():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    model_path = "/root/autodl-tmp/Qwen/Qwen2-Audio-7B-Instruct"
    save_path = f"/root/autodl-tmp/Qwen2-Audio-finetune/output_model/single-{time.strftime('%H-%M-%S')}"
    adapter_ckpt_path = "/root/autodl-tmp/Qwen2-Audio-finetune/best_dep_adapter.pt"  # 训练好的adapter路径
    train_data_path = "/root/autodl-tmp/Qwen2-Audio-finetune/data/eatd/train"
    eval_data_path = "/root/autodl-tmp/Qwen2-Audio-finetune/data/eatd/test"
    train_prompt_path = "/root/autodl-tmp/Qwen2-Audio-finetune/data/eatd/train/eatd_multiprompt.jsonl"
    eval_prompt_path = "/root/autodl-tmp/Qwen2-Audio-finetune/data/eatd/test/eatd_multiprompt.jsonl"
    wav_type = "wav"

    os.makedirs(save_path, exist_ok=True)

    logger = set_logger(save_path)
    set_seed(1234)

    print_gpu_memory("初始状态")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    print_gpu_memory("模型加载后")

    # ====== 替换 audio_tower，加载预训练 adapter ======
    adapter_config = {"adapter_dim": 512, "dropout": 0.1}
    model.audio_tower = create_modified_qwen2audio_encoder(
        model.audio_tower,
        adapter_config,
        adapter_ckpt=adapter_ckpt_path,
        freeze_adapter=True,   # 固定不更新
    )

    # ====== 给 language_model 添加 LoRA ======
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj", "o_proj", "k_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model.language_model = get_peft_model(model.language_model, lora_config)

    # ====== 冻结除 projection + LoRA 外的参数 ======
    for name, p in model.named_parameters():
        p.requires_grad = False

    # multi_modal_projector 是 Qwen2-Audio 的音频-文本桥接层
    if hasattr(model, "multi_modal_projector"):
        for n, p in model.multi_modal_projector.named_parameters():
            p.requires_grad = True
            print(f"解冻 multi_modal_projector 参数: {n}")

    # LoRA 层参数（自动可训练）
    for name, p in model.named_parameters():
        if "lora" in name.lower() or "peft" in name.lower():
            p.requires_grad = True
            print(f"解冻 LoRA 参数: {name}")

    # 检查
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in trainable_params)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params_count:,}")
    print(f"可训练参数比例: {trainable_params_count/total_params*100:.4f}%")

    # 启用梯度检查点
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.to(device)
    print_gpu_memory("模型移动到设备后")

    # 优化器与调度器
    optim = torch.optim.AdamW(trainable_params, lr=1e-4)
    total_train_steps = 1000
    warmup_steps = 100
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda step: min(step / warmup_steps, 1.0)
        if step < warmup_steps
        else max(0.0, 1 - (step - warmup_steps) / (total_train_steps - warmup_steps))
    )

    # 数据加载
    train_dataset = AudioDataset(train_data_path, train_prompt_path, wav_type)
    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=4,
        pin_memory=True, collate_fn=partial(collate_fn_qwen2audio, processor=processor)
    )
    eval_dataset = AudioDataset(eval_data_path, eval_prompt_path, wav_type)
    eval_loader = DataLoader(
        eval_dataset, batch_size=2, shuffle=False, num_workers=4,
        pin_memory=True, collate_fn=partial(collate_fn_qwen2audio, processor=processor)
    )

    # ==========================
    # 训练循环
    # ==========================
    best_f1 = -float("inf")
    for epoch in range(10):
        model.train()
        train_bar = tqdm(train_loader, desc=f"[Train] epoch {epoch}")
        
        for step, batch in enumerate(train_bar):
            if step % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
                acc = compute_acc_text(processor, outputs.logits, batch["labels"])

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optim.step()
            scheduler.step()

            train_bar.set_description(f"[Train] epoch {epoch} | loss {loss:.3f} | acc {acc:.3f}")

        # ====== 验证 ======
        model.eval()
        eval_loss = 0
        global_stats = None  # 初始化全局统计
        
        with torch.no_grad():
            eval_bar = tqdm(eval_loader, desc="[Eval]")
            eval_results = []
            
            for batch_idx, batch in enumerate(eval_bar):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    # 累积统计信息
                    global_stats = compute_metrics_text_binary_accumulate(
                        processor, outputs.logits, batch["labels"], global_stats
                    )
                
                eval_loss += loss.item()
                
                # 进度条显示：基于当前累积的统计计算临时指标
                if global_stats and global_stats['total'] > 0:
                    temp_acc, temp_precision, temp_recall, temp_f1, temp_wf1 = compute_metrics_from_stats(global_stats)
                    eval_bar.set_description(f"[Eval] loss {loss:.3f} | acc {temp_acc:.4f} | posF1 {temp_f1:.4f} | wF1 {temp_wf1:.4f}")

                # ====== 解析并打印该批次的预测与真实标签（按标签位置解码） ======
                preds = torch.argmax(outputs.logits, dim=-1)  # [B, T_pred]
                labels = batch["labels"]  # [B, T_label]
                B = labels.size(0)
                for b in range(B):
                    # 对齐长度：使用标签有效位数与预测长度的最小值
                    mask = labels[b] != -100  # [T_label]
                    t_label_len = mask.sum().item()
                    if t_label_len == 0:
                        continue
                    seq_len = min(t_label_len, preds[b].size(0))
                    # 取标签的最后 seq_len 个有效token，与预测尾部对齐
                    idxs = mask.nonzero(as_tuple=False).squeeze(-1)
                    used_label_indices = idxs[-seq_len:]
                    preds_aligned = preds[b][-seq_len:]
                    true_ids = labels[b][used_label_indices]
                    pred_ids = preds_aligned

                    true_text = processor.tokenizer.decode(true_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    pred_text = processor.tokenizer.decode(pred_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    true_label = map_label_from_text(true_text)
                    pred_label = map_label_from_text(pred_text)
                    eval_results.append({"pred": pred_label, "true": true_label})

        # 计算最终指标（使用所有累积的统计）
        num_eval_batches = len(eval_loader)
        eval_loss /= num_eval_batches
        
        if global_stats and global_stats['total'] > 0:
            eval_acc, eval_precision, eval_recall, eval_f1, eval_wf1 = compute_metrics_from_stats(global_stats)
            
            logger.info(f"[Eval] epoch {epoch} | loss={eval_loss:.3f}, acc={eval_acc:.4f}, posF1={eval_f1:.4f}, posPre={eval_precision:.4f}, posRec={eval_recall:.4f}, wF1={eval_wf1:.4f}")
            
            # 输出混淆矩阵
            logger.info(f"[Confusion Matrix] TP={global_stats['tp']}, FP={global_stats['fp']}, FN={global_stats['fn']}, TN={global_stats['tn']}, Total={global_stats['total']}, Correct={global_stats['correct']}")
        else:
            logger.warning(f"[Eval] No valid samples for evaluation")
            eval_acc = eval_precision = eval_recall = eval_f1 = eval_wf1 = 0.0

        # 将本轮验证集的预测结果写入文件，便于事后检查
        preds_path = os.path.join(save_path, f"eval_preds_epoch{epoch}.jsonl")
        try:
            with open(preds_path, "w", encoding="utf-8") as f:
                for r in eval_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            logger.info(f"[Eval-Gen] saved predictions to {preds_path}")
        except Exception as e:
            logger.error(f"[Eval-Gen] failed to save predictions: {e}")

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            model.save_pretrained(save_path, safe_serialization=True)
            processor.save_pretrained(save_path)
            logger.info(f"[Saving Best Model] F1 improved to {best_f1:.3f} → {save_path}")

        torch.cuda.empty_cache()
        gc.collect()

    print(f"训练完成，最佳 F1 = {best_f1:.3f}")
    print_gpu_memory("训练完成后")


if __name__ == "__main__":
    train_single()
