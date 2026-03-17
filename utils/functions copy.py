import torch
import numpy as np

def _decode_label_span(tokenizer, ids: torch.Tensor):
    """将标签token序列解码为文本。ids为一维tensor。"""
    if ids.numel() == 0:
        return ""
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)

def _sample_first_label_indices(labels: torch.Tensor):
    """返回每个样本第一个有效标签位置索引（labels != -100）。不存在则为 -1。"""
    B, T = labels.shape
    first_idx = torch.full((B,), -1, dtype=torch.long, device=labels.device)
    for b in range(B):
        idxs = (labels[b] != -100).nonzero(as_tuple=False).squeeze(-1)
        if idxs.numel() > 0:
            first_idx[b] = idxs[0]
    return first_idx

def compute_acc_text(processor, logits, labels):
    """按样本解码标签片段并映射到二分类，计算准确率。
    说明：labels中有效位置(!= -100)视为标签片段，预测从logits取argmax并对齐到相同长度进行解码。
    """
    preds = torch.argmax(logits, dim=-1)  # [B, T_pred]
    B = labels.size(0)
    correct = 0
    total = 0
    for b in range(B):
        mask = labels[b] != -100  # [T_label]
        t_label_len = mask.sum().item()
        if t_label_len == 0:
            continue
        # 对齐预测长度到标签长度（取序列尾部）
        t_pred_len = min(t_label_len, preds[b].size(0))
        # 找到标签的实际索引范围
        idxs = mask.nonzero(as_tuple=False).squeeze(-1)
        # 只使用最后t_pred_len个标签位与预测尾部对齐
        used_label_indices = idxs[-t_pred_len:]
        true_ids = labels[b][used_label_indices]
        pred_ids = preds[b][-t_pred_len:]

        true_text = _decode_label_span(processor.tokenizer, true_ids)
        pred_text = _decode_label_span(processor.tokenizer, pred_ids)

        # print(f"Sample {b}: true_text: {true_text}, pred_text: {pred_text}")

        def map_text(t: str):
            t = (t or "").strip()
            if ("健康" in t):
                return 0  # 健康
            if ("抑郁" in t) or ("抑" in t):
                return 1  # 抑郁
            return -1

        y_true = map_text(true_text)
        y_pred = map_text(pred_text)
        if y_true != -1 and y_pred != -1:
            correct += int(y_true == y_pred)
            total += 1
    if total == 0:
        return torch.tensor(0.0, device=labels.device)
    return torch.tensor(correct / total, device=labels.device)

# ==========================
# 修改后的指标计算函数（只累积统计，不计算指标）
# ==========================
def compute_metrics_text_binary_accumulate(processor, logits, labels, global_stats=None):
    """只累积统计信息，不计算指标"""
    preds = torch.argmax(logits, dim=-1)
    device = labels.device
    
    # 如果是第一次调用，初始化全局统计
    if global_stats is None:
        global_stats = {
            'tp': 0,  # 抑郁预测抑郁
            'fp': 0,  # 健康预测抑郁
            'fn': 0,  # 抑郁预测健康
            'tn': 0,  # 健康预测健康
            'total': 0,
            'correct': 0
        }
    
    B = labels.size(0)
    
    for b in range(B):
        mask = labels[b] != -100
        t_label_len = mask.sum().item()
        if t_label_len == 0:
            continue
        t_pred_len = min(t_label_len, preds[b].size(0))
        idxs = mask.nonzero(as_tuple=False).squeeze(-1)
        used_label_indices = idxs[-t_pred_len:]
        true_ids = labels[b][used_label_indices]
        pred_ids = preds[b][-t_pred_len:]

        true_text = processor.tokenizer.decode(true_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        pred_text = processor.tokenizer.decode(pred_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # print(f"true_text: {true_text}, pred_text: {pred_text}")
        
        def map_text(t: str):
            t = (t or "").strip()
            # 更精确的映射
            if t == "健康":
                return 0
            if t == "抑郁":  # 只匹配完整的"抑郁"
                return 1
            return -1

        yt = map_text(true_text)
        yp = map_text(pred_text)
        
        if yt != -1:
            global_stats['total'] += 1
            if yp != -1:
                if yt == yp:
                    global_stats['correct'] += 1
                    if yt == 1:
                        global_stats['tp'] += 1
                    else:
                        global_stats['tn'] += 1
                else:
                    if yt == 1 and yp == 0:  # 真实抑郁但预测健康
                        global_stats['fn'] += 1
                    elif yt == 0 and yp == 1:  # 真实健康但预测抑郁
                        global_stats['fp'] += 1
            else:
                # 预测无效，视为错误
                if yt == 1:
                    global_stats['fn'] += 1
                else:
                    global_stats['fp'] += 1
    
    # 返回更新后的统计信息
    return global_stats


# ==========================
# 根据累积统计计算指标的函数
# ==========================
def compute_metrics_from_stats(global_stats):
    """根据累积的统计信息计算指标"""
    if not global_stats or global_stats['total'] == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    total = global_stats['total']
    correct = global_stats['correct']
    tp = global_stats['tp']
    fp = global_stats['fp']
    fn = global_stats['fn']
    tn = global_stats['tn']
    
    # 准确率
    accuracy = correct / total if total > 0 else 0.0
    
    # 正类（抑郁）指标
    pos_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pos_f1 = (2 * pos_precision * pos_recall / (pos_precision + pos_recall)) if (pos_precision + pos_recall) > 0 else 0.0
    
    # 负类（健康）指标
    neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    neg_f1 = (2 * neg_precision * neg_recall / (neg_precision + neg_recall)) if (neg_precision + neg_recall) > 0 else 0.0
    
    # 加权F1
    pos_weight = (tp + fn) / total if total > 0 else 0.0
    neg_weight = (tn + fp) / total if total > 0 else 0.0
    f1_weighted = pos_weight * pos_f1 + neg_weight * neg_f1
    
    return accuracy, pos_precision, pos_recall, pos_f1, f1_weighted
    

def compute_metrics_text_binary(processor, logits, labels, global_stats=None):
    """使用全局累积统计"""
    preds = torch.argmax(logits, dim=-1)
    device = labels.device
    
    if global_stats is None:
        global_stats = {
            'tp': torch.tensor(0, device=device, dtype=torch.float),
            'fp': torch.tensor(0, device=device, dtype=torch.float),
            'fn': torch.tensor(0, device=device, dtype=torch.float),
            'tn': torch.tensor(0, device=device, dtype=torch.float),
            'total': torch.tensor(0, device=device, dtype=torch.float),
            'correct': torch.tensor(0, device=device, dtype=torch.float)
        }
    
    B = labels.size(0)
    for b in range(B):
        mask = labels[b] != -100
        t_label_len = mask.sum().item()
        if t_label_len == 0:
            continue
        t_pred_len = min(t_label_len, preds[b].size(0))
        idxs = mask.nonzero(as_tuple=False).squeeze(-1)
        used_label_indices = idxs[-t_pred_len:]
        true_ids = labels[b][used_label_indices]
        pred_ids = preds[b][-t_pred_len:]

        true_text = _decode_label_span(processor.tokenizer, true_ids)
        pred_text = _decode_label_span(processor.tokenizer, pred_ids)

        print(f"true_text: {true_text}, pred_text: {pred_text}")
        
        def map_text(t: str):
            t = (t or "").strip()
            # 更精确的映射
            if t == "健康":
                return 0
            if t == "抑郁":  # 只匹配完整的"抑郁"
                return 1
            return -1  # 无效

        yt = map_text(true_text)
        yp = map_text(pred_text)
        
        if yt != -1:
            global_stats['total'] += 1
            if yp != -1:
                if yt == yp:
                    global_stats['correct'] += 1
                    if yt == 1:
                        global_stats['tp'] += 1
                    else:
                        global_stats['tn'] += 1
                else:
                    if yt == 1 and yp == 0:
                        global_stats['fn'] += 1
                    elif yt == 0 and yp == 1:
                        global_stats['fp'] += 1
            else:
                # 预测无效，视为错误
                if yt == 1:
                    global_stats['fn'] += 1
                else:
                    global_stats['fp'] += 1
    
    # 计算当前批次基于累积统计的指标
    total = global_stats['total']
    if total > 0:
        accuracy = global_stats['correct'] / total
        
        # 正类指标
        tp = global_stats['tp']
        fp = global_stats['fp']
        fn = global_stats['fn']
        
        pos_precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0, device=device)
        pos_recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, device=device)
        pos_f1 = (2 * pos_precision * pos_recall / (pos_precision + pos_recall)) if (pos_precision + pos_recall) > 0 else torch.tensor(0.0, device=device)
        
        # weighted F1
        tn = global_stats['tn']
        # 负类指标
        neg_precision = tn / (tn + fn) if (tn + fn) > 0 else torch.tensor(0.0, device=device)
        neg_recall = tn / (tn + fp) if (tn + fp) > 0 else torch.tensor(0.0, device=device)
        neg_f1 = (2 * neg_precision * neg_recall / (neg_precision + neg_recall)) if (neg_precision + neg_recall) > 0 else torch.tensor(0.0, device=device)
        
        # 加权F1
        pos_weight = (tp + fn) / total
        neg_weight = (tn + fp) / total
        f1_weighted = pos_weight * pos_f1 + neg_weight * neg_f1
        
        return accuracy, pos_precision, pos_recall, pos_f1, f1_weighted, global_stats
    else:
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero, zero, zero, global_stats


# import torch
# from sklearn.metrics import precision_score, recall_score, f1_score
# import numpy as np
# def compute_acc(logits,labels):
#     _,labels_len = labels.shape
#     preds = torch.argmax(logits,dim=-1)
#     labels_indices = labels != -100 
#     acc = torch.sum(preds[:,-labels_len-1:-1][labels_indices] == labels[labels_indices]).float() /torch.sum(labels_indices).float()
#     return acc

# def compute_metrics(logits, labels):
#     _, labels_len = labels.shape
#     preds = torch.argmax(logits, dim=-1)
#     labels_indices = labels != -100
    
#     preds_aligned = preds[:, -labels_len-1:-1]  # 从倒数第labels_len+1到倒数第1个
#     labels_aligned = labels
#     labels_indices_aligned = labels_indices
    
#     # 确保形状匹配
#     seq_len = min(preds_aligned.size(1), labels_aligned.size(1))
#     preds_aligned = preds_aligned[:, -seq_len:]
#     labels_aligned = labels_aligned[:, -seq_len:]
#     labels_indices_aligned = labels_indices_aligned[:, -seq_len:]
    
#     valid_preds = preds_aligned[labels_indices_aligned]
#     valid_labels = labels_aligned[labels_indices_aligned]
    
#     device = labels.device
    
#     if len(valid_labels) == 0:
#         return (torch.tensor(0.0, device=device), 
#                 torch.tensor(0.0, device=device), 
#                 torch.tensor(0.0, device=device), 
#                 torch.tensor(0.0, device=device))
    
#     # 计算准确率
#     accuracy = torch.sum(valid_preds == valid_labels).float() / torch.sum(labels_indices_aligned).float()
    
#     # 获取所有类别
#     all_classes = torch.unique(torch.cat([valid_preds, valid_labels]))
    
#     # 计算每个类别的指标
#     precision_list = []
#     recall_list = []
    
#     for class_id in all_classes:
#         tp = torch.sum((valid_preds == class_id) & (valid_labels == class_id)).float()
#         fp = torch.sum((valid_preds == class_id) & (valid_labels != class_id)).float()
#         fn = torch.sum((valid_preds != class_id) & (valid_labels == class_id)).float()
        
#         precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0, device=device)
#         recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, device=device)
        
#         precision_list.append(precision)
#         recall_list.append(recall)
    
#     # 宏平均
#     if precision_list:
#         precision_macro = torch.mean(torch.stack(precision_list))
#         recall_macro = torch.mean(torch.stack(recall_list))
#     else:
#         precision_macro = torch.tensor(0.0, device=device)
#         recall_macro = torch.tensor(0.0, device=device)
    
#     # 计算F1
#     if precision_macro + recall_macro > 0:
#         f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro)
#     else:
#         f1_macro = torch.tensor(0.0, device=device)
    
#     return accuracy, precision_macro, recall_macro, f1_macro
