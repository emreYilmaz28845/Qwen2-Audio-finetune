import os
import json
import glob

def load_emotion_labels(emotion_file_path):
    """
    加载情感描述文件，构建字典
    文件格式：t_1/neutral_out.wav	['心酸肠疼，伤心难过']
    """
    emotion_dict = {}
    if os.path.exists(emotion_file_path):
        with open(emotion_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        audio_path = parts[0].strip()
                        emotion = parts[1].strip()
                        # 提取key：子文件夹_文件名（不带_out）
                        folder_name = os.path.dirname(audio_path)
                        file_name = os.path.basename(audio_path).replace('_out.wav', '')
                        key = f"{folder_name}_{file_name}"
                        emotion_dict[key] = emotion
    return emotion_dict

def load_transcription(subdir_path, audio_file):
    """
    加载对应音频的文本转录
    audio_file: positive_out.wav -> 读取positive.txt
    """
    # 获取基础文件名（不带_out.wav）
    base_name = audio_file.replace('_out.wav', '')
    txt_file = os.path.join(subdir_path, f"{base_name}.txt")
    
    if os.path.exists(txt_file):
        with open(txt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        print(f"警告: 找不到转录文件 {txt_file}")
        return ""

def generate_detailed_jsonl_files(dataset_path, emotion_file_path, output_dir):
    """
    生成包含完整prompt的jsonl文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载情感描述字典
    emotion_dict = load_emotion_labels(emotion_file_path)
    print(f"加载了 {len(emotion_dict)} 条情感描述")
    
    train_output = os.path.join(output_dir, "train_detailed.jsonl")
    test_output = os.path.join(output_dir, "test_detailed.jsonl")
    
    with open(train_output, 'w', encoding='utf-8') as train_f, \
         open(test_output, 'w', encoding='utf-8') as test_f:
        
        for subdir in os.listdir(dataset_path):
            subdir_path = os.path.join(dataset_path, subdir)
            
            if not os.path.isdir(subdir_path):
                continue
                
            # 选择对应的输出文件
            if 't' in subdir:  # 训练集
                output_file = train_f
            elif 'v' in subdir:  # 测试集
                output_file = test_f
            else:
                continue
                
            # 读取标签文件
            label_file = os.path.join(subdir_path, "new_label.txt")
            if not os.path.exists(label_file):
                print(f"跳过文件夹 {subdir} (没有找到new_label.txt)")
                continue
                
            try:
                with open(label_file, 'r', encoding='utf-8') as lf:
                    label_value = float(lf.read().strip())
                target = "抑郁" if label_value >= 53.0 else "非抑郁"
            except Exception as e:
                print(f"处理文件夹 {subdir} 时出错: {e}")
                continue
            
            # 处理音频文件
            for file in os.listdir(subdir_path):
                if file.endswith('_out.wav'):
                    key = f"{subdir}_{file.replace('_out.wav', '')}"
                    task = f"{key}_抑郁症识别"
                    
                    # 获取情感描述
                    emotion = emotion_dict.get(key, "['情感描述未找到']")
                    
                    # 获取文本转录
                    transcription = load_transcription(subdir_path, file)
                    
                    # 构建完整的prompt
                    prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>请根据这段语音、其对应的文本转录和情感描述判断该说话人是抑郁还是非抑郁\n情感描述: {emotion}\n文本转录: {transcription}"
                    
                    json_obj = {
                        "task": task,
                        "prompt": prompt
                    }
                    
                    output_file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

def generate_combined_jsonl(dataset_path, emotion_file_path, output_file):
    """
    生成合并的详细jsonl文件
    """
    # 加载情感描述字典
    emotion_dict = load_emotion_labels(emotion_file_path)
    print(f"加载了 {len(emotion_dict)} 条情感描述")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for subdir in os.listdir(dataset_path):
            subdir_path = os.path.join(dataset_path, subdir)
            
            if not os.path.isdir(subdir_path):
                continue
                
            # 只处理包含t或v的文件夹
            if not ('t' in subdir or 'v' in subdir):
                continue
                
            # 处理音频文件
            for file in os.listdir(subdir_path):
                if file.endswith('_out.wav'):
                    key = f"{subdir}_{file.replace('_out.wav', '')}"
                    task = f"{key}_抑郁症识别"
                    
                    # 获取情感描述
                    emotion = emotion_dict.get(key, "['情感描述未找到']")
                    
                    # 获取文本转录
                    transcription = load_transcription(subdir_path, file)
                    
                    # 构建完整的prompt
                    prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>请根据这段语音、其对应的文本转录和情感描述判断该说话人是抑郁还是非抑郁\n情感描述: {emotion}\n文本转录: {transcription}"
                    
                    json_obj = {
                        "task": task,
                        "prompt": prompt
                    }
                    
                    f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    dataset_path = "/root/autodl-tmp/EATD-Corpus" 
    emotion_file_path = "/root/autodl-tmp/EATD_emotion/inference_results.txt" 
    output_dir = "/root/autodl-tmp/Qwen2-Audio-finetune/data/eatd"
    
    # 生成分离的训练集和测试集
    generate_detailed_jsonl_files(dataset_path, emotion_file_path, output_dir)
    print(f"详细jsonl文件已生成在目录: {output_dir}")
    print("训练集文件: train_detailed.jsonl")
    print("测试集文件: test_detailed.jsonl")
    
    # 生成合并的文件
    generate_combined_jsonl(dataset_path, emotion_file_path, "all_detailed.jsonl")
    print("合并的详细jsonl文件已生成: all_detailed.jsonl")