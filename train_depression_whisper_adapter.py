import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from transformers import WhisperFeatureExtractor, WhisperModel, WhisperProcessor
from typing import List, Dict, Union

# ================= DepAdapter =================
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
        return self.layer_norm(x + residual)

# ================= 数据加载 =================
def load_scp_jsonl(scp_path, jsonl_path):
    scp_dict = {}
    with open(scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            key, wav_path = line.strip().split(maxsplit=1)
            scp_dict[key] = wav_path

    label_dict = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            key = item['key']
            target = item['target']
            if target in ["抑郁", "depressed"]:
                label = 1
            elif target in ["健康", "normal", "非抑郁"]:
                label = 0
            else:
                continue
            if key in scp_dict:
                label_dict[key] = (scp_dict[key], label)
    return label_dict

def merge_datasets(dataset_roots):
    merged_train, merged_val = {}, {}
    for root in dataset_roots:
        train_scp = os.path.join(root, "train.scp")
        train_jsonl = os.path.join(root, "train.jsonl")
        val_scp = os.path.join(root, "val.scp")
        val_jsonl = os.path.join(root, "val.jsonl")
        if not (os.path.exists(train_scp) and os.path.exists(train_jsonl)):
            print(f"跳过缺失文件的数据集：{root}")
            continue
        merged_train.update(load_scp_jsonl(train_scp, train_jsonl))
        merged_val.update(load_scp_jsonl(val_scp, val_jsonl))
    print(f"合并完成：训练集 {len(merged_train)} 条，验证集 {len(merged_val)} 条")
    return merged_train, merged_val

# ================= Dataset =================
class DepressionDataset(Dataset):
    def __init__(self, data_dict, processor):
        self.data = list(data_dict.items())
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key, (wav_path, label) = self.data[idx]

        try:
            # 用 librosa 加载音频
            waveform, sr = librosa.load(wav_path, sr=16000)
            
            # 使用特征提取器处理音频
            input_features = self.processor.feature_extractor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features  # [1, 80, 3000]
            
            # 移除batch维度
            input_features = input_features.squeeze(0)  # [80, 3000]
            
            return {
                "input_features": input_features,
                "labels": torch.tensor(label, dtype=torch.long)
            }
            
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            # 返回空特征
            input_features = torch.zeros(80, 3000)
            return {
                "input_features": input_features,
                "labels": torch.tensor(label, dtype=torch.long)
            }

# ================= Data Collator =================
class DataCollatorDepressionWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 处理音频特征
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # 处理标签
        labels = [feature["labels"] for feature in features]
        batch["labels"] = torch.stack(labels)
        
        return batch

# ================= 模型 =================
class DepressionDetectionModel(nn.Module):
    def __init__(self, whisper_model, adapter_dim=512, hidden_dim=512, num_classes=2):
        super().__init__()
        self.whisper = whisper_model
        whisper_dim = whisper_model.config.d_model

        # 冻结 Whisper 参数
        for p in self.whisper.parameters():
            p.requires_grad = False

        self.dep_adapter = DepAdapter(audio_dim=whisper_dim, adapter_dim=adapter_dim)
        self.attention = nn.Linear(whisper_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(whisper_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_features):
        # input_features: [B, 80, 3000]
        
        # 添加batch维度（如果需要）
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)
        
        # 通过Whisper编码器
        outputs = self.whisper.encoder(input_features=input_features)
        hidden_states = outputs.last_hidden_state  # [B, T, D]
        
        adapted = self.dep_adapter(hidden_states)
        attn_weights = torch.softmax(self.attention(adapted).squeeze(-1), dim=-1)
        pooled = torch.sum(adapted * attn_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(pooled)
        return logits

# ================= 训练函数 =================
def train_model(train_loader, val_loader, model, device, num_epochs=10, lr=1e-4, save_path="best_dep_adapter.pt"):
    criterion = nn.CrossEntropyLoss()
    
    # 只训练adapter和分类器，不训练Whisper
    trainable_params = []
    trainable_params.extend(model.dep_adapter.parameters())
    trainable_params.extend(model.classifier.parameters())
    trainable_params.extend(model.attention.parameters())
    
    optimizer = optim.AdamW(trainable_params, lr=lr)
    model.to(device)
    best_f1 = 0.0

    for epoch in range(num_epochs):
        # -------- 训练阶段 --------
        model.train()
        total_loss, total_correct = 0, 0
        for batch_idx, batch in enumerate(train_loader):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            
            # 调试：检查输入形状
            if batch_idx == 0:
                print(f"Batch input features shape: {input_features.shape}")
                print(f"Batch labels shape: {labels.shape}")
            
            optimizer.zero_grad()
            logits = model(input_features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()

        train_acc = total_correct / len(train_loader.dataset)

        # -------- 验证阶段 --------
        model.eval()
        TP = TN = FP = FN = 0
        with torch.no_grad():
            for batch in val_loader:
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_features)
                preds = torch.argmax(logits, dim=1)
                TP += ((preds == 1) & (labels == 1)).sum().item()
                TN += ((preds == 0) & (labels == 0)).sum().item()
                FP += ((preds == 1) & (labels == 0)).sum().item()
                FN += ((preds == 0) & (labels == 1)).sum().item()

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        val_acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"| Train Loss: {total_loss/len(train_loader.dataset):.4f} "
              f"| Train Acc: {train_acc:.4f} "
              f"| Val Acc: {val_acc:.4f} "
              f"| Precision: {precision:.4f} "
              f"| Recall: {recall:.4f} "
              f"| F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            # 只保存adapter模块的权重
            torch.save(model.dep_adapter.state_dict(), save_path)
            print(f"新最佳模型（F1={best_f1:.4f}）已保存到 {save_path}")

    print(f"训练完成，最高 F1={best_f1:.4f}")

# ================= 主函数 =================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset_paths = [
        "/root/autodl-tmp/Qwen2-Audio-finetune/data/finetune_adaptor/cmdc",
        "/root/autodl-tmp/Qwen2-Audio-finetune/data/finetune_adaptor/daic",
        "/root/autodl-tmp/Qwen2-Audio-finetune/data/finetune_adaptor/eatd"
    ]

    train_dict, val_dict = merge_datasets(dataset_paths)
    
    if len(train_dict) == 0:
        print("错误：训练数据为空！")
        exit(1)

    print(f"加载Whisper处理器...")
    processor = WhisperProcessor.from_pretrained("/root/autodl-tmp/AI-ModelScope/whisper-large-v3")
    print(f"加载Whisper模型...")
    whisper_model = WhisperModel.from_pretrained("/root/autodl-tmp/AI-ModelScope/whisper-large-v3")

    print(f"创建数据集...")
    train_dataset = DepressionDataset(train_dict, processor)
    val_dataset = DepressionDataset(val_dict, processor)

    # 创建data collator
    data_collator = DataCollatorDepressionWithPadding(processor)

    # 测试一个样本
    print("测试数据样本...")
    sample = train_dataset[0]
    print(f"样本输入形状: {sample['input_features'].shape}")
    print(f"样本标签: {sample['labels']}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=True,
        collate_fn=data_collator
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=128, 
        shuffle=False,
        collate_fn=data_collator
    )

    model = DepressionDetectionModel(whisper_model, adapter_dim=512)
    
    print("开始训练...")
    train_model(train_loader, val_loader, model, device, num_epochs=10, lr=1e-4)
