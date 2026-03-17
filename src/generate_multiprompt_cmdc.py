import os
import json

def read_existing_fold_jsonl(fold_dir, fold_num, split):
    """读取指定折的现有JSONL文件"""
    jsonl_file = os.path.join(fold_dir, f"fold{fold_num}_{split}_multitask.jsonl")
    task_mapping = {}
    if os.path.exists(jsonl_file):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                key = data.get("key", "")
                task = data.get("task", "")
                task_mapping[key] = task
        print(f"  从 {jsonl_file} 加载了 {len(task_mapping)} 个task映射")
    else:
        print(f"  警告: 找不到文件 {jsonl_file}")
    return task_mapping

def read_emotion_labels(label_file):
    """读取情感标签文件"""
    emotion_data = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                audio_path = parts[0]  # MDD/MDD10/Q1.wav
                emotion = parts[1]
                # 从路径提取key: MDD/MDD10/Q1.wav -> MDD10_Q1
                basename = os.path.basename(audio_path)  # Q1.wav
                filename = os.path.splitext(basename)[0]  # Q1
                dirname = os.path.basename(os.path.dirname(audio_path))  # MDD10
                key = f"{dirname}_{filename}"
                emotion_data[key] = emotion
    return emotion_data

def read_text_file(text_path):
    """读取文本转录文件"""
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except:
        return ""

def get_text_path_from_key(text_base_dir, key):
    """根据key构造文本文件路径"""
    # key格式: MDD10_Q1 或 HC10_Q1
    parts = key.split('_')
    if len(parts) >= 2:
        dir_name = parts[0]  # MDD10 或 HC10
        file_name = parts[1]  # Q1
        
        # 确定类别文件夹 (MDD 或 HC)
        if dir_name.startswith('MDD'):
            category = 'MDD'
        elif dir_name.startswith('HC'):
            category = 'HC'
        else:
            category = dir_name[:3]  # 默认取前3个字符
        
        # 构造完整路径: /root/autodl-tmp/CMDC_text/MDD/MDD10/Q1.txt
        text_path = os.path.join(text_base_dir, category, dir_name, f"{file_name}.txt")
        return text_path
    return ""

def generate_single_fold_split_jsonl(existing_fold_dir, text_base_dir, label_file, output_dir, fold_num, split):
    """为单个折的单个split（train/test）生成JSONL文件"""
    
    # 读取现有的task映射
    task_mapping = read_existing_fold_jsonl(existing_fold_dir, fold_num, split)
    if not task_mapping:
        print(f"  折{fold_num}_{split}: 没有找到task映射，跳过")
        return
    
    # 读取情感标签
    emotion_data = read_emotion_labels(label_file)
    print(f"  折{fold_num}_{split}: 情感标签 {len(emotion_data)} 个样本")
    
    # 生成JSONL文件
    output_file = os.path.join(output_dir, f"fold{fold_num}_{split}.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        processed = 0
        missing_text = 0
        missing_task = 0
        
        for key, emotion_label in emotion_data.items():
            # 获取task名称
            task_name = task_mapping.get(key)
            if not task_name:
                missing_task += 1
                continue
            
            # 构造文本文件路径
            text_path = get_text_path_from_key(text_base_dir, key)
            
            # 读取文本转录
            text_transcription = read_text_file(text_path)
            if not text_transcription:
                text_transcription = "文本转录未找到"
                missing_text += 1
            
            # 构建prompt
            prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>请根据这段语音、其对应的文本转录和情感描述判断该说话人是抑郁还是非抑郁\n情感描述: {emotion_label}\n文本转录: {text_transcription}"
            
            # 构建JSON对象
            json_obj = {
                "task": task_name,
                "prompt": prompt
            }
            
            f_out.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
            processed += 1
        
        print(f"  折{fold_num}_{split}: 成功处理 {processed} 个样本")
        print(f"     - 缺少文本转录: {missing_text} 个")
        print(f"     - 缺少task映射: {missing_task} 个")

def generate_5_fold_jsonl(existing_fold_dir, text_base_dir, emotion_labels_dir, output_dir):
    """生成5折交叉验证的JSONL文件"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"开始生成5折交叉验证数据...")
    print(f"现有JSONL目录: {existing_fold_dir}")
    print(f"文本目录: {text_base_dir}")
    print(f"情感标签目录: {emotion_labels_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    for fold_num in range(1, 6):
        print(f"正在处理第{fold_num}折...")
        
        # 处理训练集
        train_label_file = os.path.join(emotion_labels_dir, f"fold{fold_num}_train_multiprompt.txt")
        if os.path.exists(train_label_file):
            generate_single_fold_split_jsonl(
                existing_fold_dir, 
                text_base_dir, 
                train_label_file, 
                output_dir, 
                fold_num, 
                "train"
            )
        else:
            print(f"  警告: 找不到训练集标签文件 {train_label_file}")
        
        # 处理测试集
        test_label_file = os.path.join(emotion_labels_dir, f"fold{fold_num}_test_multiprompt.txt")
        if os.path.exists(test_label_file):
            generate_single_fold_split_jsonl(
                existing_fold_dir, 
                text_base_dir, 
                test_label_file, 
                output_dir, 
                fold_num, 
                "test"
            )
        else:
            print(f"  警告: 找不到测试集标签文件 {test_label_file}")
        
        print(f"第{fold_num}折处理完成")
        print("-" * 30)

if __name__ == "__main__":
    existing_fold_directory = "/root/autodl-tmp/Qwen2-Audio-finetune/data/cmdc/jsonl"  
    text_base_directory = "/root/autodl-tmp/CMDC_text"  
    emotion_labels_directory = "/root/autodl-tmp/CMDC_emotion"  
    output_directory = "/root/autodl-tmp/Qwen2-Audio-finetune/data/cmdc" 
    
    generate_5_fold_jsonl(
        existing_fold_dir=existing_fold_directory,
        text_base_dir=text_base_directory,
        emotion_labels_dir=emotion_labels_directory,
        output_dir=output_directory
    )
    
    print("5折交叉验证JSONL文件生成完成！")
    print(f"输出文件保存在: {output_directory}")