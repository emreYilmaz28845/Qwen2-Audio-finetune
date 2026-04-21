import torch
import numpy as np
import os

def _env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


_ENABLE_METRIC_DEBUG = _env_flag("AUDIOLLM_ENABLE_METRIC_DEBUG", "0")
_METRIC_DEBUG_LIMIT = int(os.environ.get("AUDIOLLM_METRIC_DEBUG_LIMIT", "12"))
_metric_debug_count = 0


def _is_main_rank():
    return int(os.environ.get("RANK", "0")) == 0

# ========================== NEW VERSION (with global statistics accumulation) ==========================
# This version accumulates true positives, false positives, true negatives, false negatives, total samples, and correct predictions across batches. 
# The compute_metrics_text_binary function updates these statistics and computes metrics based on the accumulated values.
# The compute_metrics_from_stats function can be called at the end of an epoch to compute final metrics from the accumulated statistics.

# converting token ids to text for the label span
def _decode_label_span(tokenizer, ids: torch.Tensor):
    """Decode the label token sequence into text. ids is a 1D tensor."""
    if ids.numel() == 0:
        return ""
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)



# not used in the current version ??
def _sample_first_label_indices(labels: torch.Tensor):
    """Return the index of the first valid label position (labels != -100) for each sample. If none exists, return -1."""
    B, T = labels.shape
    # create a result tensor of shape [B]
    # initialize every sample’s result as -1
    # -1 means “no valid label found”
    first_idx = torch.full((B,), -1, dtype=torch.long, device=labels.device)
    
    for b in range(B):
        # build a boolean mask for valid label positions in sample b
        # get the indices where label is valid
        # idxs becomes something like [15, 16, 17]
        idxs = (labels[b] != -100).nonzero(as_tuple=False).squeeze(-1)
        
        # if there are valid label positions, take the first one and store it in the result tensor
        if idxs.numel() > 0:
            first_idx[b] = idxs[0]
    return first_idx




def compute_acc_text(processor, logits, labels):
    """Decode label spans per sample, map them to binary classes, and compute accuracy.
    Note: valid positions in labels (!= -100) are treated as the label span, 
    and predictions are obtained by argmax over logits and decoded after alignment to the same length.
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
        # get the exact positions of the label tokens
        idxs = mask.nonzero(as_tuple=False).squeeze(-1)
        # collect the true token IDs at those positions
        true_ids = labels[b][idxs]
        # FIX: In causal LMs, logits[t] predicts token at position t+1,
        # so for label at position i, the prediction is at preds[i-1].
        # CHECKPOINT
        pred_indices = idxs - 1
        pred_indices = pred_indices.clamp(min=0)  # safety clamp
        pred_ids = preds[b][pred_indices]

        true_text = _decode_label_span(processor.tokenizer, true_ids)
        pred_text = _decode_label_span(processor.tokenizer, pred_ids)

        # print(f"Sample {b}: true_text: {true_text}, pred_text: {pred_text}")
        # mapping depressed/healthy labels to binary classes (1 for depressed, 0 for healthy, -1 for invalid)
        def map_text(t: str):
            t = (t or "").strip()
            # Check "非抑郁" BEFORE "抑郁" to avoid substring false match
            if ("非抑郁" in t) or ("健康" in t):
                return 0  # Non-depressed / healthy
            if ("抑郁" in t) or ("抑" in t):
                return 1  # Depressed
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
# Modified metric computation function (only accumulates statistics, does not compute metrics)
# ==========================
def compute_metrics_text_binary_accumulate(processor, logits, labels, global_stats=None):
    """Only accumulate statistics, do not compute metrics."""
    global _metric_debug_count

    preds = torch.argmax(logits, dim=-1)
    device = labels.device
    
    # Initialize global statistics on the first call
    if global_stats is None:
        global_stats = {
            'tp': 0,  # Depressed predicted as depressed
            'fp': 0,  # Healthy predicted as depressed
            'fn': 0,  # Depressed predicted as healthy
            'tn': 0,  # Healthy predicted as healthy
            'total': 0,
            'correct': 0
        }
    
    B = labels.size(0)
    
    for b in range(B):
        mask = labels[b] != -100
        t_label_len = mask.sum().item()
        if t_label_len == 0:
            continue
        idxs = mask.nonzero(as_tuple=False).squeeze(-1)
        true_ids = labels[b][idxs]
        # FIX: In causal LMs, logits[t] predicts token at position t+1,
        # so for label at position i, the prediction is at preds[i-1].
        pred_indices = idxs - 1
        pred_indices = pred_indices.clamp(min=0)  # safety clamp
        pred_ids = preds[b][pred_indices]

        true_text = processor.tokenizer.decode(true_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        pred_text = processor.tokenizer.decode(pred_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # print(f"true_text: {true_text}, pred_text: {pred_text}")
        
        def map_text(t: str):
            t = (t or "").strip()
            # Check "非抑郁" BEFORE "抑郁" to avoid substring false match
            if t == "健康" or t == "非抑郁" or "非抑郁" in t or "健康" in t:
                return 0
            if t == "抑郁" or "抑郁" in t:  # Match "depressed"
                return 1
            return -1

        yt = map_text(true_text)
        yp = map_text(pred_text)

        if _ENABLE_METRIC_DEBUG and _is_main_rank() and _metric_debug_count < _METRIC_DEBUG_LIMIT:
            print("[DEBUG compute_metrics_text_binary_accumulate]")
            print(f"sample_index={_metric_debug_count + 1}")
            print(f"true_text={true_text!r}")
            print(f"pred_text={pred_text!r}")
            print(f"mapped_true={yt}")
            print(f"mapped_pred={yp}")
            print(f"current_stats_before={{'tp': {global_stats['tp']}, 'fp': {global_stats['fp']}, 'fn': {global_stats['fn']}, 'tn': {global_stats['tn']}, 'total': {global_stats['total']}, 'correct': {global_stats['correct']}}}")
            print("=" * 80)
            _metric_debug_count += 1
        
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
                    if yt == 1 and yp == 0:  # Actually depressed but predicted healthy
                        global_stats['fn'] += 1
                    elif yt == 0 and yp == 1:  # Actually healthy but predicted depressed
                        global_stats['fp'] += 1
            else:
                # Invalid prediction, count as an error
                if yt == 1:
                    global_stats['fn'] += 1
                else:
                    global_stats['fp'] += 1
    
    # Return the updated statistics
    return global_stats


# ==========================
# Function to compute metrics from accumulated statistics
# ==========================
def compute_metrics_from_stats(global_stats):
    """Compute metrics from accumulated statistics."""
    if not global_stats or global_stats['total'] == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    total = global_stats['total']
    correct = global_stats['correct']
    tp = global_stats['tp']
    fp = global_stats['fp']
    fn = global_stats['fn']
    tn = global_stats['tn']
    
    # Accuracy
    accuracy = correct / total if total > 0 else 0.0
    
    # Positive class (depressed) metrics
    pos_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pos_f1 = (2 * pos_precision * pos_recall / (pos_precision + pos_recall)) if (pos_precision + pos_recall) > 0 else 0.0
    
    # Negative class (healthy) metrics
    neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    neg_f1 = (2 * neg_precision * neg_recall / (neg_precision + neg_recall)) if (neg_precision + neg_recall) > 0 else 0.0
    
    # Weighted F1
    pos_weight = (tp + fn) / total if total > 0 else 0.0
    neg_weight = (tn + fp) / total if total > 0 else 0.0
    f1_weighted = pos_weight * pos_f1 + neg_weight * neg_f1
    
    return accuracy, pos_precision, pos_recall, pos_f1, f1_weighted
    

# It is not used in the current version, more like a combination of the above two functions, where it both updates the global statistics and computes metrics based on the updated statistics.
def compute_metrics_text_binary(processor, logits, labels, global_stats=None):
    """Use globally accumulated statistics."""
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
        idxs = mask.nonzero(as_tuple=False).squeeze(-1)
        true_ids = labels[b][idxs]
        # FIX: In causal LMs, logits[t] predicts token at position t+1,
        # so for label at position i, the prediction is at preds[i-1].
        pred_indices = idxs - 1
        pred_indices = pred_indices.clamp(min=0)  # safety clamp
        pred_ids = preds[b][pred_indices]

        true_text = _decode_label_span(processor.tokenizer, true_ids)
        pred_text = _decode_label_span(processor.tokenizer, pred_ids)

        # print(f"true_text: {true_text}, pred_text: {pred_text}")  # Commented out for cleaner output
        
        def map_text(t: str):
            t = (t or "").strip()
            # Check "非抑郁" BEFORE "抑郁" to avoid substring false match
            if t == "健康" or t == "非抑郁" or "非抑郁" in t or "健康" in t:
                return 0
            if t == "抑郁" or "抑郁" in t:  # Match "depressed"
                return 1
            return -1  # Invalid

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
                # Invalid prediction, count as an error
                if yt == 1:
                    global_stats['fn'] += 1
                else:
                    global_stats['fp'] += 1
    
    # Compute metrics for the current batch based on accumulated statistics
    total = global_stats['total']
    if total > 0:
        accuracy = global_stats['correct'] / total
        
        # Positive class metrics
        tp = global_stats['tp']
        fp = global_stats['fp']
        fn = global_stats['fn']
        
        pos_precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0, device=device)
        pos_recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, device=device)
        pos_f1 = (2 * pos_precision * pos_recall / (pos_precision + pos_recall)) if (pos_precision + pos_recall) > 0 else torch.tensor(0.0, device=device)
        
        # weighted F1
        tn = global_stats['tn']
        # Negative class metrics
        neg_precision = tn / (tn + fn) if (tn + fn) > 0 else torch.tensor(0.0, device=device)
        neg_recall = tn / (tn + fp) if (tn + fp) > 0 else torch.tensor(0.0, device=device)
        neg_f1 = (2 * neg_precision * neg_recall / (neg_precision + neg_recall)) if (neg_precision + neg_recall) > 0 else torch.tensor(0.0, device=device)
        
        # Weighted F1
        pos_weight = (tp + fn) / total
        neg_weight = (tn + fp) / total
        f1_weighted = pos_weight * pos_f1 + neg_weight * neg_f1
        
        return accuracy, pos_precision, pos_recall, pos_f1, f1_weighted, global_stats
    else:
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero, zero, zero, global_stats



# ========================== OLD VERSION (computes metrics directly without accumulation) ==========================
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
# this is literally token accuracy, not the same as text-level binary classification metrics
def compute_acc(logits,labels):
    # logits: [B, T, V], labels: [B, T]
    _,labels_len = labels.shape
    #getting the highest score index from the logits as prediction
    preds = torch.argmax(logits,dim=-1)
    # returns false if the position is a padding token, true otherwise
    labels_indices = labels != -100 
    #computing token level accuracy only on the valid label positions (labels != -100)
    # CHECKPOINT ??
    acc = torch.sum(preds[:,-labels_len-1:-1][labels_indices] == labels[labels_indices]).float() /torch.sum(labels_indices).float()
    return acc

def compute_metrics(logits, labels):
    _, labels_len = labels.shape
    preds = torch.argmax(logits, dim=-1)
    labels_indices = labels != -100
    
    preds_aligned = preds[:, -labels_len-1:-1]  # From the -(labels_len+1)-th token to the second-to-last token
    labels_aligned = labels
    labels_indices_aligned = labels_indices
    
    # Ensure the shapes match
    seq_len = min(preds_aligned.size(1), labels_aligned.size(1))
    preds_aligned = preds_aligned[:, -seq_len:]
    labels_aligned = labels_aligned[:, -seq_len:]
    labels_indices_aligned = labels_indices_aligned[:, -seq_len:]
    
    valid_preds = preds_aligned[labels_indices_aligned]
    valid_labels = labels_aligned[labels_indices_aligned]
    
    device = labels.device
    
    if len(valid_labels) == 0:
        return (torch.tensor(0.0, device=device), 
                torch.tensor(0.0, device=device), 
                torch.tensor(0.0, device=device), 
                torch.tensor(0.0, device=device))
    
    # Compute token accuracy
    accuracy = torch.sum(valid_preds == valid_labels).float() / torch.sum(labels_indices_aligned).float()
    
    # Get all classes
    all_classes = torch.unique(torch.cat([valid_preds, valid_labels]))
    
    # Compute metrics for each class
    precision_list = []
    recall_list = []

    # valid_preds and valid_labels are token IDs, not decoded class labels
    # all_classes is the set of token IDs appearing in predictions/labels
    # the loop computes TP/FP/FN for each token ID as if each token were a class

    for class_id in all_classes:
        tp = torch.sum((valid_preds == class_id) & (valid_labels == class_id)).float()
        fp = torch.sum((valid_preds == class_id) & (valid_labels != class_id)).float()
        fn = torch.sum((valid_preds != class_id) & (valid_labels == class_id)).float()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0, device=device)
        recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, device=device)
        
        precision_list.append(precision)
        recall_list.append(recall)
    
    # Macro average
    if precision_list:
        precision_macro = torch.mean(torch.stack(precision_list))
        recall_macro = torch.mean(torch.stack(recall_list))
    else:
        precision_macro = torch.tensor(0.0, device=device)
        recall_macro = torch.tensor(0.0, device=device)
    
    # Compute F1
    if precision_macro + recall_macro > 0:
        f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro)
    else:
        f1_macro = torch.tensor(0.0, device=device)
    
    return accuracy, precision_macro, recall_macro, f1_macro
